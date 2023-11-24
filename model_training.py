import os
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
import utils
from sklearn.model_selection import train_test_split
from skimage.util import montage
from skimage.segmentation import mark_boundaries
from unet import Unet
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras import models, layers
import keras.backend as K
from keras.optimizers.legacy import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import json


BATCH_SIZE = 48
EDGE_CROP = 16
GAUSSIAN_NOISE = 0.1
UPSAMPLE_MODE = 'SIMPLE'
NET_SCALING = (1, 1)
IMG_SCALING = (3, 3)
VALID_IMG_COUNT = 450
MAX_TRAIN_STEPS = 10
MAX_TRAIN_EPOCHS = 20
AUGMENT_BRIGHTNESS = False

class Preprocessor():
    def __init__(self):
        self.train_image_dir = os.path.join('airbus-ship-detection', 'train_v2')
        self.test_image_dir = os.path.join('airbus-ship-detection', 'test_v2')
        self.train_raw = pd.read_csv('airbus-ship-detection/train_ship_segmentations_v2.csv')
    
    def sample_ships(self, in_df, base_rep_val=1500):
        if in_df['ships'].values[0]==0:
            return in_df.sample(base_rep_val//3) # even more strongly undersample no ships
        else:
            return in_df.sample(base_rep_val, replace=(in_df.shape[0]<base_rep_val))
        
    def create_aug_gen(self, in_gen, dg_args, seed = None):        
        if AUGMENT_BRIGHTNESS:
            dg_args['brightness_range'] = [0.5, 1.5]
        image_gen = ImageDataGenerator(**dg_args)

        if AUGMENT_BRIGHTNESS:
            dg_args.pop('brightness_range')
        label_gen = ImageDataGenerator(**dg_args)
        
        np.random.seed(seed if seed is not None else np.random.choice(range(9999)))
        for in_x, in_y in in_gen:
            seed = np.random.choice(range(9999))
            g_x = image_gen.flow(255*in_x, 
                                batch_size = in_x.shape[0], 
                                seed = seed, 
                                shuffle=True)
            g_y = label_gen.flow(in_y, 
                                batch_size = in_x.shape[0], 
                                seed = seed, 
                                shuffle=True)

            yield next(g_x)/255.0, np.float32(next(g_y))
   
    def get_params(self): 
        """
        Returns:
            dict: train_x, train_y, valid_x, valid_y, step_count, aug_gen
        """       
        masks = pd.read_csv(r'airbus-ship-detection\train_ship_segmentations_v2.csv')        
        
        masks['ships'] = masks['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
        unique_img_ids = masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
        unique_img_ids['has_ship'] = unique_img_ids['ships'].map(lambda x: 1.0 if x>0 else 0.0)
        unique_img_ids['has_ship_vec'] = unique_img_ids['has_ship'].map(lambda x: [x])
        
        imageid = lambda c_img_id: os.stat(os.path.join(self.train_image_dir, c_img_id)).st_size/1024
        unique_img_ids['file_size_kb'] = unique_img_ids['ImageId'].map(imageid)
        unique_img_ids = unique_img_ids[unique_img_ids['file_size_kb'] > 50]        
        masks = masks.drop(['ships'], axis=1)

        train_ids, valid_ids = train_test_split(unique_img_ids, 
                                                test_size = 0.3,
                                                stratify = unique_img_ids['ships'])
        
        train_ids.to_csv('train_ids.csv')
        valid_ids.to_csv('valid_ids.csv')
        
        train_df = pd.merge(masks, train_ids)
        valid_df = pd.merge(masks, valid_ids)        
        train_df['grouped_ship_count'] = train_df['ships'].map(lambda x: (x+1)//2).clip(0, 7)            
            
        dg_args = dict(featurewise_center = False, 
                  samplewise_center = False,
                  rotation_range = 15, 
                  width_shift_range = 0.1, 
                  height_shift_range = 0.1, 
                  shear_range = 0.01,
                  zoom_range = [0.9, 1.25],  
                  horizontal_flip = True, 
                  vertical_flip = True,
                  fill_mode = 'reflect',
                  data_format = 'channels_last')
        
        balanced_train_df = train_df.groupby('grouped_ship_count').apply(self.sample_ships)        
        train_gen = utils.make_image_gen(balanced_train_df,BATCH_SIZE, IMG_SCALING, self.train_image_dir)
        train_x, train_y = next(train_gen)  
        step_count = min(MAX_TRAIN_STEPS, balanced_train_df.shape[0]//BATCH_SIZE)        
        aug_gen = self.create_aug_gen(utils.make_image_gen(balanced_train_df,BATCH_SIZE, IMG_SCALING, self.train_image_dir), 
                                      dg_args)              
        
        valid_x, valid_y = next(utils.make_image_gen(valid_df, VALID_IMG_COUNT, IMG_SCALING, self.train_image_dir))  
        
        params = {'train': (train_x, train_y),
                  'valid': (valid_x, valid_y),
                  'step_count': step_count,
                  'aug_gen': aug_gen}
                
        return params
    

class Unet:  
    def __init__(self):
        self.model = None
    
    def conv_block(self, inputs, num_filters, w):
        x = layers.Conv2D(num_filters, w, activation='relu', padding="same")(inputs)
    #     x = layers.BatchNormalization()(x)        
        x = layers.Conv2D(num_filters, w, activation='relu',padding="same")(x)
    #     x = layers.BatchNormalization()(x)

        return x

    def encoder_block(self, inputs, num_filters, w):
        x = self.conv_block(inputs, num_filters, w)
        p = layers.MaxPool2D((1,1))(x)
        p = layers.Dropout(0.5)(p)
        return x, p

    def decoder_block(self, inputs, skip, num_filters, w):
        x = layers.Conv2DTranspose(num_filters, (1,1), strides=1, padding="same")(inputs)
        x = layers.Concatenate()([x, skip])
        x = layers.Dropout(0.5)(x)
        x = self.conv_block(x, num_filters, w)
        return x

    def build_unet(self, input_shape):
        inputs = layers.Input(input_shape)
        w = 3
        # TODO: change repetetive inputs to p2,p3...
        s1, p1 = self.encoder_block(inputs, 4, w)        
        s2, p2 = self.encoder_block(inputs, 4, w)       
        s3, p3 = self.encoder_block(inputs, 8, w)        
        s4, p4 = self.encoder_block(inputs, 16, w)        
        s5, p5 = self.encoder_block(inputs, 16, w)        
        
        b1 = self.conv_block(p5, 32, w) 
                                         
        d1 = self.decoder_block(b1, s5, 16, w)
        d2 = self.decoder_block(d1, s4, 16, w)
        d3 = self.decoder_block(d2, s3, 8, w)
        d4 = self.decoder_block(d3, s2, 4, w)
        d5 = self.decoder_block(d4, s1, 4, w)        

        outputs = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(d5)
        self.model = models.Model(inputs, outputs, name="UNET")
        return self.model
        
        # d = layers.Conv2D(1, (1, 1), activation='sigmoid') (c9)
        # # d = layers.Cropping2D((EDGE_CROP, EDGE_CROP))(d)
        # # d = layers.ZeroPadding2D((EDGE_CROP, EDGE_CROP))(d)
        # if NET_SCALING is not None:
        #     d = layers.UpSampling2D(NET_SCALING)(d)
                
    def create_callbacks(self):        
        checkpoint = ModelCheckpoint("seg_model_weights.best.hdf5", 
                                     monitor='val_dice_coef',
                                     verbose=1, 
                                     save_best_only=True, 
                                     mode='max', 
                                     save_weights_only=True)

        reduceLROnPlat = ReduceLROnPlateau(monitor='val_dice_coef', 
                                           factor=0.5, 
                                           patience=3, 
                                           verbose=1, 
                                           mode='max',
                                           min_delta=0.0001, 
                                           cooldown=2, 
                                           min_lr=1e-6)
        
        early = EarlyStopping(monitor="val_dice_coef", 
                              mode="max", 
                              patience=15)
        
        return [checkpoint, early, reduceLROnPlat]
    
    def fit(self, aug_gen, valid_x, valid_y, step_count):       
        def dice_coef(y_true, y_pred, smooth=1):
            intersection = K.sum(y_true * y_pred, axis=[1,2,3])
            union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
            return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

        def dice_loss(in_gt, in_pred):
            return 1 - dice_coef(in_gt, in_pred)         
        
        self.model.compile(optimizer=Adam(1e-4, decay=1e-6), 
                           loss=dice_loss, 
                           metrics=[dice_coef, 'binary_accuracy'])
                
        callbacks_list = self.create_callbacks()    
        loss_history = [self.model.fit_generator(aug_gen, 
                                  steps_per_epoch=step_count, 
                                  epochs=MAX_TRAIN_EPOCHS, 
                                  validation_data=(valid_x, valid_y),
                                  callbacks=callbacks_list,
                                  workers=1)]   
    
if __name__ == '__main__':
    prep = Preprocessor()
    params = prep.get_params()    
    
    model = Unet()
    model.build_unet((256, 256, 3))
    
    valid_x, valid_y = params['valid']
    model.fit(params['aug_gen'], 
              valid_x, 
              valid_y, 
              params['step_count'])
    
    model.model.save('seg_model.h5')
    if IMG_SCALING is not None:
        fullres_model = models.Sequential()
        fullres_model.add(layers.AvgPool2D(IMG_SCALING, input_shape = (None, None, 3)))
        fullres_model.add(model.model)
        fullres_model.add(layers.UpSampling2D(IMG_SCALING))
    else:
        fullres_model = model.model
    fullres_model.save('fullres_model.h5')
        