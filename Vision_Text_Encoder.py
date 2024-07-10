import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
import h5py, io
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm

#Connect to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Define batch size 
batch_size = 64

#Monet model
processor = AutoProcessor.from_pretrained("suinleelab/monet")
model = AutoModelForZeroShotImageClassification.from_pretrained("suinleelab/monet")
model.to(device)
model.eval()

#Import data 
data=pd.read_csv("Data/train_metadata_with_descriptions.csv", low_memory=False)
imagedata = "Data/Dataset_Images.hdf5"


#Define custom dataset 
class CustomDataset(Dataset):
    def __init__(self, data,imagedata,processor):
        self.data = data
        self.imagedata = h5py.File(imagedata, 'r')
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #Read the Image Data
        fname = self.data.iloc[idx]['isic_id']
        image = np.array(self.imagedata[fname]) #opens image
        image = [Image.open(io.BytesIO(image))]
        image = self.processor.image_processor(image)['pixel_values'][0]

        #Read the Text Data
        description= self.data.iloc[idx]['description']
        tokens=self.processor.tokenizer(description,truncation = True)['input_ids']

        return image,torch.tensor(tokens)


custom_data=CustomDataset(data,imagedata,processor)
next(iter(custom_data))
data_loader=DataLoader(custom_data, batch_size=batch_size, shuffle=False, num_workers=0)

Embeddings=torch.empty((0,1536))


#Extract image and text feature vectors 
with torch.no_grad():
    for inputs in tqdm(data_loader):
        Image_Embeddings = model.get_image_features(inputs[0].to(device))
        Text_Embeddings  = model.get_text_features (inputs[1].to(device))
        Embeddings = torch.concat((Embeddings,torch.concat((Image_Embeddings.cpu(),Text_Embeddings.cpu()),dim = -1)),dim = 0)
embeddings_np=Embeddings.detach().numpy()

#Save the image and text feature vectors as a numpy array 
np.save("Data/train_vision_text_embeddings.npy", embeddings_np)

print("All done!")