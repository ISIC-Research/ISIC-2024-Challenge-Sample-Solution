
# ISIC 2024 Sample Solution Using the Monet Model

This sample solution uses the Monet Model, an image-text based model trained on dermatology data. This project should be used as a starting point for participants of the
ISIC 2024 - Skin Cancer Detection with 3D-TBP Challenge hosted on Kaggle. 

https://www.kaggle.com/competitions/isic-2024-challenge/overview

## Accessing the Dataset 

The data can be downloaded from Kaggle using the CLI tool. See the example in Downloading_Data.py. 

## Instructions to Acces the Monet Model

Instructions to download the Monet Model can be found here: https://github.com/suinleelab/MONET.

Monet License: Attribution-NonCommercial-ShareAlike 4.0 International

## Creating Image Descriptions from Metadata 

The Monet Model has the ability to process text along with images. Captions can be generated using metadata fields to supplement each image. 

See Creating_Metadata_Descriptions.py for a tutorial on creating these captions. 

## Extracting Image and Text Features 

Once captions for images have been created, the next step is to create the image and text feature embeddings. These embeddings will be used to train a classification head to identify benign and malignant images. 

An example of how to create these embeddings using the Monet Model may be found in Vision_Text_Encoder.py. 

## Training 

After creating image-text embeddings, one can begin training a binary classifier to detect benign and malignant images. An example of how data may be split to train and evaluate an algorithm, as well as the training process itself can be found in Monet_Classifier_training_FV_Challenge2024.py. 

## Citation

Kim, C., Gadgil, S.U., DeGrave, A.J. et al. Transparent medical image AI via an image–text foundation model grounded in medical literature. Nat Med 30, 1154–1165 (2024). https://doi.org/10.1038/s41591-024-02887-x

This sample solution was created by Maura Gillis and Kivanc Kose at Memorial Sloan Kettering Cancer Center, Dermatology Service. 