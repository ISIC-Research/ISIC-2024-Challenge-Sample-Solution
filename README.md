
# ISIC 2024 Sample Solution Using the Monet Model

This sample solution uses the Monet Model is an image-text based model trained on dermatology data. This project should be used as a starting point for participants of the
ISIC 2024 - Skin Cancer Detection with 3D-TBP Challenge hosted on Kaggle. 

## Accessing the Dataset 

The data can be downloaded from Kaggle using the CLI Itool. See the example in Downloading_Data.py. 


## Creating Image Descriptions from Metadata. 

Model has the ability to process text along with images. Captions for images can be generated using metadata fields to supplement each image. 


See Creating_Metadata_Descriptions.py for a tutorial on creating these captions. 


## Extracting Image and Text Features 

Once captions for images have been created, the next step is to create the image and text feature embeddings. These embeddings will be used to train a classification head to identify benign and malignant images. 

An example of how to create these embeddings using the Monet model may be found in Vision_Text_Encoder.py. 

## Training 

After creating image-text embeddings, one can begin training a binary classifier to detect benign and malignant images. An example of how data may be split to train and evaluate an algorithm van be found in Monet_Classifier_training_FV_Challenge2024.py. 

## Citation

This sample solution was created by Maura Gillis and Kivanc Kose at Memorial Sloan Kettering Cancer Center, Dermatology Service. 

Kim, C., Gadgil, S.U., DeGrave, A.J. et al. Transparent medical image AI via an image–text foundation model grounded in medical literature. Nat Med 30, 1154–1165 (2024). https://doi.org/10.1038/s41591-024-02887-x