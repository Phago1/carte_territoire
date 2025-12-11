=======================================================

------------------carte et territoire------------------

This package is to train a deep-learning model based on semantic segmentation.
The goal is to classify aerial images of earth based on 
Institut National des Informations Géographique (IGN) dataset.

Give an image of a region, that has at least
256x256 pixels, to the model and it will return a territory's classification.
It can be used to record the evolution of the surface type (man made's construction, agriculture, deforestation...)

The dataset:
IGN has a database of satellite ORTHO-Images (Near-Infrared - Red - Green) of various french
territories. We selected the dataset from the Belfort aerea.

The app:
The app is online with Streamlit, the code is in another repository carte-territoire-website
created by A. Pareux (one of the contributors)

files:
- dl_logic/preprocessor.py
- dl_logic/data.py
- dl_logic/labels.py
- dl_logic/model.py
- dl_logic/registry.py
- interface/main.py
- interface/load_chunks_to_GCP.py
- interface/workflow.py
- api/fast.py
- params.py

---- data.py ----
tiles_viz : function to visualize the image, the labelised image and an overlay of both
label_tiles_info : 

---- labels.py ----
flair_class_data : is the original classification from IGN
REDUCED_7 : is the classification we define
COSIA16_TO_REDUCED7 : mapping from original to new classification
count_class : gives the distribution of class of one image in a dictionary
merge_counts : merges one dict of class distribution to a new one
compute_dataset_class_stats :

---- model.py ----
we initialize three models with different architectures, cnn, U-net and U-net+.
The models are built with the Tenserflow library
initialize_cnn_model
initialize_unet_model
initialize_unet_plus_model
compile_model : the learning rate can be either a float or exponential. The target class is
either reduce to seven classes or not. There is 16 classes originally.
train_model

---- preprocessor.py ----

----registry.py ----
The functions to save the metrics and the model's parameters and another to save the model.

-------------------------

---- main.py ----

---- load_chunks_to_GCP.py ----

---- workflow.py ----

-------------------------

---- fast.py ----

-------------------------

---- params.py ----
