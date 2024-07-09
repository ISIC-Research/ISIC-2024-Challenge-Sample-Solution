



#Packages##########################################################################################################
import pandas as pd
import numpy as np
from tqdm import tqdm
#from utils import HDFDataset,get_transform
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
####################################################################################################################

#Preliminary Things#################################################################################################
    
#Connect to device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Define batch size 
batch_size = 128

test_features=np.load("Data/test_vision_text_embeddings.npy")

data=pd.read_csv("Data/test_metadata_with_descriptions.csv", low_memory=False, header=0, index_col=0)


ground_truth=pd.read_csv("Data/test-gt.csv").set_index("isic_id")

ground_truth['label'] = ground_truth['label'].map({True: 1, False: 0})

#data_test=data.set_index('isic_id')
test_data=data.join(ground_truth["label"])


num_features = test_features.shape[1]


classification_head = nn.Sequential(
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 1)
)


class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


# Create instances of the dataset for training and validation data
test_dataset = CustomDataset(test_features, test_data['label'].values)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


model_temp=classification_head
state_dict=torch.load("Outputs/Monet_BestModel_FV.pth")
model_temp.load_state_dict(state_dict)
model= model_temp.to(device)

model.eval()

#Test loop
predictions = []
true_labels = []


with torch.no_grad():
    for feature, label in tqdm(test_loader):
        feature, label = feature.to(device), label.to(device)
        outputs = model(feature)
        predictions += list(nn.Sigmoid()(outputs).cpu().numpy())
        true_labels += list(label.cpu().numpy())

    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    test_roc_auc = roc_auc_score(true_labels, predictions)


    with open(f"MONET_Kaggle_Sample_Solution_TestAUC.txt", "a") as file:
        file.write(f"{test_roc_auc:.4f}\n")


# Evaluate the model
predictions = np.array(predictions)
test_data_new=data.copy()
test_data_new["target"]=predictions


submission=test_data_new[["target"]]

submission.index.name = 'isic_id'

submission_path="MONET_Kaggle_Sample_Submission_Inference.csv"
submission.to_csv(submission_path)

print(submission.head())

