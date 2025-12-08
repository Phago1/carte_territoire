=======================================================
------------------carte et territoire------------------

This package is to classify aerial images of earth based on 
Institut National des Informations Géographique (IGN)  dataset

Goal: is to give an image of a territory that has at least
256x256 pixels and our model will return a classification
of that territory

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

---- preprocessor.py ----

----registry.py ----

-------------------------

---- main.py ----

---- load_chunks_to_GCP.py ----

---- workflow.py ----

-------------------------

---- fast.py ----

-------------------------

---- params.py ----
