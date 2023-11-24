import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread
from matplotlib.cm import get_cmap
from matplotlib import colormaps
from skimage.morphology import binary_opening, disk, label
from model_training import Unet
import os
import numpy as np
from keras import layers, models
import utils
import json


class ModelInference():
    IMG_SCALING = (3, 3)
    test_image_dir = os.path.join('airbus-ship-detection', 'test_v2')
    train_image_dir = os.path.join('airbus-ship-detection', 'train_v2')
    
    def __init__(self):
        self.fullres_model = None

    def load_fullres_model(self, seg_model_path="model_params/seg_model_weights.best.hdf5",
                           fullres_path='model_params/fullres_model.h5'):
        seg_model = Unet().build_unet((256, 256, 3))    
        seg_model.load_weights(seg_model_path)
        if self.IMG_SCALING is not None:
            fullres_model = models.Sequential()
            fullres_model.add(layers.AvgPool2D(self.IMG_SCALING, input_shape = (None, None, 3)))
            fullres_model.add(seg_model)
            fullres_model.add(layers.UpSampling2D(self.IMG_SCALING))
        else:
            fullres_model = seg_model
        fullres_model.load_weights(fullres_path)
        
        return fullres_model

    def masks_as_color(self, in_mask_list):
        # Take the individual ship masks and create a color mask array for each ships
        all_masks = np.zeros((768, 768), dtype = np.float32)
        scale = lambda x: (len(in_mask_list)+x+1) / (len(in_mask_list)*2) ## scale the heatmap image to shift 
        for i,mask in enumerate(in_mask_list):
            if isinstance(mask, str):
                all_masks[:,:] += scale(i) * utils.rle_decode(mask)
        return all_masks

    def raw_prediction(self, img, c_img_name, path=test_image_dir):
        c_img = imread(os.path.join(path, c_img_name))
        c_img = np.expand_dims(c_img, 0)/255.0
        cur_seg = self.fullres_model.predict(c_img)[0]
        return cur_seg, c_img[0]

    def smooth(self, cur_seg):
        return binary_opening(cur_seg>0.99, np.expand_dims(disk(2), -1))

    def predict(self, img, path=test_image_dir):
        cur_seg, c_img = self.raw_prediction(img, path=path)
        return self.smooth(cur_seg), c_img
    
    def show_preds(self, valid_df, masks):
        samples = valid_df.groupby('ships').apply(lambda x: x.sample(1))
        fig, m_axs = plt.subplots(samples.shape[0], 4, figsize = (15, samples.shape[0]*4))
        [c_ax.axis('off') for c_ax in m_axs.flatten()]

        for (ax1, ax2, ax3, ax4), c_img_name in zip(m_axs, samples.ImageId.values):
            first_seg, first_img = self.raw_prediction(c_img_name, c_img_name, self.train_image_dir)
            ax1.imshow(first_img)
            ax1.set_title('Image: ' + c_img_name)
            # ax2.imshow(first_seg[:, :, 0], cmap=colormaps('jet'))
            ax2.imshow(first_seg)
            ax2.set_title('Model Prediction')
            reencoded = self.masks_as_color(utils.multi_rle_encode(self.smooth(first_seg)))
            ax3.imshow(reencoded)
            ax3.set_title('Prediction Masks')
            ground_truth = self.masks_as_color(masks.query('ImageId=="{}"'.format(c_img_name))['EncodedPixels'])
            ax4.imshow(ground_truth)
            ax4.set_title('Ground Truth')
        # fig.show() # for some reason it closes immediately
        fig.savefig('test_predictions.jpg')

    def get_data(self):
        masks = pd.read_csv(r'airbus-ship-detection\train_ship_segmentations_v2.csv')
        masks['ships'] = masks['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
        unique_img_ids = masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
        unique_img_ids['has_ship'] = unique_img_ids['ships'].map(lambda x: 1.0 if x>0 else 0.0)
        unique_img_ids['has_ship_vec'] = unique_img_ids['has_ship'].map(lambda x: [x])
        
        unique_img_ids['file_size_kb'] = unique_img_ids['ImageId'].map(lambda c_img_id: 
                                                                    os.stat(os.path.join(self.train_image_dir, 
                                                                                            c_img_id)).st_size/1024)
        unique_img_ids = unique_img_ids[unique_img_ids['file_size_kb'] > 50]
        unique_img_ids['file_size_kb'].hist()
        masks = masks.drop(['ships'], axis=1)
            
        valid_ids = pd.read_csv('valid_ids.csv')
        
        return masks, pd.merge(masks, valid_ids)
        
def main():
    inference = ModelInference()
    masks, valid_df = inference.get_data()
    fullres_model = inference.load_fullres_model()
    inference.fullres_model = fullres_model
    inference.show_preds(valid_df, masks, fullres_model)

if __name__=='__main__':
    main()