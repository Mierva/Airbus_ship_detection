<H2>model_training.py</H2>
trains data on a randomly selected dataset from train_v2 kaggle dataset and saves the indeces into train_ids.csv, the same goes for valid_ids.csv
also saves weights for model trained on a 256x256 pictures and full-resolution model (model_params folder, seg_model_weights.best.hdf5 and fullres_model.h5 respectively)

<H2>model_inference.py</H2>
shows predictions on a test data using weights from previously trained model on the same achetecture model_training.py has, saving results to test_predictions.jpg 
<br></br>
The whole project is built on python 3.9.7
