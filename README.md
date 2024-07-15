
# ISIC 2024 Sample Solution Using the Monet Model

This sample solution uses the MONET Model, an image-text based model trained on dermatology data. This project can be used as a starting point for participants in the
[ISIC 2024 - Skin Cancer Detection with 3D-TBP Challenge](https://www.kaggle.com/competitions/isic-2024-challenge) hosted on Kaggle. 

https://www.kaggle.com/competitions/isic-2024-challenge/overview

## Accessing the Dataset 

The data can be downloaded from Kaggle using the CLI tool. See the example in Downloading_Data.py. 

## Instructions to Access the Monet Model

Instructions on downloading the MONET Model can be found here: https://github.com/suinleelab/MONET.

**!!REMINDER!!**: MONET License: Attribution-NonCommercial-ShareAlike 4.0 International. Therefore, [ISIC 2024 - Skin Cancer Detection with 3D-TBP Challenge](https://www.kaggle.com/competitions/isic-2024-challenge) submissions using the MONET model can not be considered for prize award, but will be considered in the leaderboards.

## Creating Image Descriptions from Metadata 

MONET is a Vision-Language model that can process text along with images. In this work, we generated captions (text information) using metadata fields to supplement each image. 

See Creating_Metadata_Descriptions.py for a tutorial on creating these captions. 

## Extracting Image and Text Features 

Once image captions have been created, the next step is to create the image and text feature embeddings. These embeddings will be used to train a classification head to identify benign and malignant images. 

An example of how to create these embeddings using the MONET Model may be found in [Vision_Text_Encoder.py](./Vision_Text_Encoder.py). 

## Training 

After creating image & text embeddings, one can begin training a binary classifier to detect benign and malignant images. Examplar code on how data may be split to train and evaluate an algorithm, as well as the training process itself, can be found in [Monet_Classifier_training_FV_Challenge2024.py](./Monet_Classifier_training_FV_Challenge2024.py). 

## Citation

Kim, C., Gadgil, S.U., DeGrave, A.J. et al. Transparent medical image AI via an image–text foundation model grounded in medical literature. Nat Med 30, 1154–1165 (2024). https://doi.org/10.1038/s41591-024-02887-x

This sample solution was created by Maura Gillis and Kivanc Kose at Memorial Sloan Kettering Cancer Center, Dermatology Service. 
