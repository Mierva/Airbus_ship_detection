## Dataset for training and validation 
https://www.kaggle.com/c/airbus-ship-detection/data
<br></br>

## model_training.py 
trains data on a randomly selected dataset from train_v2 kaggle dataset and saves the indeces into **train_ids.csv**, the same goes for **valid_ids.csv**.  
Also saves weights for model trained on a 256x256 pictures and full-resolution model (model_params folder, **seg_model_weights.best.hdf5** and **fullres_model.h5** respectively)

## model_inference.py
shows predictions on a test data using weights from previously trained model on the same achetecture **model_training.py** has, saving results to **test_predictions.jpg**
<br></br>
The whole project is built using python 3.9.7
