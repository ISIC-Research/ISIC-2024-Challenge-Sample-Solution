#Training classifier head with Monet features

#Packages##########################################################################################################
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
import random
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc
####################################################################################################################

    
#Connect to device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Define batch size 
batch_size = 128

#Define Train vs Validation ratio (default 0.7 Train and 0.3 Val)
train_val_fraction = 0.7


####################################################################################################################

#Data File Paths#####################################################################################################

#Define the metadata for the labels 

data = pd.read_csv("Data/train_metadata_with_descriptions.csv", low_memory=False, header=0, index_col=0)
data = data.reset_index(drop=True)


#Load feature vectors 
features=np.load("Data/train_vision_text_embeddings.npy")
####################################################################################################################


####################################################################################################################

#Data Splitting######################################################################################################


#Split Training Data to Training and Validation Data 

#Patient-wise splitting. Each patients with at least 1 malignant lesions and mark them as Malignant. 
#                        Then we sample between Malignant and Benign patients with the given ratio
mal_ben = np.clip(data.groupby('patient_id')['target'].sum(),0,1)

# Find the patients and shuffle their order for randomization
ben = list(mal_ben[mal_ben==0].keys())
mal = list(mal_ben[mal_ben==1].keys())
random.shuffle(ben),random.shuffle(mal)

# Sample with given ration
patients_train = ben[:int(train_val_fraction*len(ben))]+mal[:int(train_val_fraction*len(mal))]
patients_val   = ben[int(train_val_fraction*len(ben)):]+mal[int(train_val_fraction*len(mal)):]

# Add another columm to the dataframe to indicate if a sample is in Training (0) or Validation (1) Set
data['train_val'] = 0

for pts in tqdm(patients_val):
    data.loc[data['patient_id']==pts,'train_val'] = 1
    
train_data = data[data['train_val']==0] #selects images in training set
val_data  = data[data['train_val']==1] #selects images in the validation set


train_data_index = train_data.index.tolist()
val_data_index = val_data.index.tolist()

train_features = features[train_data_index]
val_features = features[val_data_index]


#Defining the Custom Dataset ##########################################################################################


# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        target = self.targets[idx]
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

# Create instances of the dataset for training and validation data
train_dataset = CustomDataset(train_features, train_data['target'].values)
val_dataset = CustomDataset(val_features, val_data['target'].values)

# Create DataLoader for batching
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


####################################################################################################################

num_features = train_features.shape[1]

classification_head = nn.Sequential(
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 1)
)


classification_head = classification_head.to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([100]).cuda()) 
optimizer = optim.Adam(classification_head.parameters(), lr=0.0001, weight_decay=1e-5)

# Training loop
num_epochs = 100 
scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10, verbose=True) 
best_val_roc_auc=0 
best_epoch = 0

min_tpr = 0.88
max_fpr = 1 - min_tpr

for epoch in range(num_epochs):
    classification_head.train()  
    running_loss = 0.0 
    train_labels=[]
    train_preds=[]

    for feature, target in tqdm(train_loader): 
        feature, target = feature.view(feature.size(0),-1).to(device).to(torch.float), target.to(device)  
        optimizer.zero_grad() 
        outputs = classification_head(feature)
        loss = criterion(outputs, target.float().view(-1,1)) 
        loss.backward()   
        optimizer.step() 
        running_loss += loss.item()
        train_labels += [l for l in target.cpu().numpy()]
        train_preds += [l for l in nn.Sigmoid()(outputs).cpu().detach().numpy()]
    train_auc = roc_auc_score(train_labels, train_preds)

    #Calculate the pAUC using https://github.com/ISIC-Research/Challenge-2024-Metrics/blob/main/PrimaryMetric-pAUC.py
    gt = abs(np.asarray(train_labels) - 1)
    pred = -1.0 * np.asarray(train_preds)

    fpr, tpr, _ = roc_curve(gt, pred, sample_weight=None)
    if max_fpr is None or max_fpr == 1:
        partial_auc = auc(fpr, tpr)
    elif max_fpr <= 0 or max_fpr > 1:
        raise ValueError("Expected min_tpr in range [0, 1), got: %r" % min_tpr)
    else:
        # Add a single point at max_fpr by linear interpolation
        stop = np.searchsorted(fpr, max_fpr, "right")
        x_interp = [fpr[stop - 1], fpr[stop]]
        y_interp = [tpr[stop - 1], tpr[stop]]
        tpr = np.append(tpr[:stop], np.interp(max_fpr, x_interp, y_interp))
        fpr = np.append(fpr[:stop], max_fpr)
        partial_auc = auc(fpr, tpr)


    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")
    print(f"Training ROC AUC: {train_auc}")
    print(f"Training pAUC at TPR 0.88: {partial_auc}")

    with open("Monet_TrainAUC.txt", "a") as file:
        file.write(f"Epoch {epoch + 1}, AUC: {train_auc:.4f}, pAUC@TPR0.88: {partial_auc:.4f}\n")


    #Validation loop
    classification_head.eval()
    val_loss = 0.0
    predictions = []
    true_labels = []
    

    with torch.no_grad():
        for feature, target in tqdm(val_loader):
            feature, target = feature.view(feature.size(0),-1).to(device).to(torch.float), target.to(device)
            outputs = classification_head(feature)
            loss = criterion(outputs, target.float().view(-1, 1))
            val_loss += loss.item()
            predictions += list(nn.Sigmoid()(outputs).cpu().detach().numpy())
            true_labels += list(target.cpu().numpy())

    val_loss /= len(val_loader)

    # Evaluate the model
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    threshold = 0.5
    binary_predictions = (predictions > threshold).astype(int)

    #Print the confusion matrix
    epoch_conf_matrix = confusion_matrix(true_labels, binary_predictions)
    print(f'\nEpoch: {epoch + 1}, Confusion Matrix:\n{epoch_conf_matrix}')

    #Calculate the ROC
    roc_auc = roc_auc_score(true_labels, predictions)
    
    #Calculate the pAUC
    v_gt = abs(true_labels - 1)
    v_pred = -1.0 * predictions

    fpr, tpr, _ = roc_curve(v_gt, v_pred, sample_weight=None)
    if max_fpr is None or max_fpr == 1:
        partial_auc = auc(fpr, tpr)
    elif max_fpr <= 0 or max_fpr > 1:
        raise ValueError("Expected min_tpr in range [0, 1), got: %r" % min_tpr)
    else:
        # Add a single point at max_fpr by linear interpolation
        stop = np.searchsorted(fpr, max_fpr, "right")
        x_interp = [fpr[stop - 1], fpr[stop]]
        y_interp = [tpr[stop - 1], tpr[stop]]
        tpr = np.append(tpr[:stop], np.interp(max_fpr, x_interp, y_interp))
        fpr = np.append(fpr[:stop], max_fpr)
        partial_auc = auc(fpr, tpr)

    scheduler.step(roc_auc)

    print("Validation ROC AUC:", roc_auc)
    print("Validation pAUC at TPR 0.88:", partial_auc)

    with open(f"Monet_ValAUC.txt", "a") as file:
        file.write(f"Epoch {epoch + 1}, {roc_auc:.4f}, pAUC@TPR0.88: {partial_auc:.4f}\n")


    if best_val_roc_auc<roc_auc:
        best_val_roc_auc = roc_auc
        best_epoch=epoch
        torch.save(classification_head.state_dict(), "Monet_BestModel.pth")

    print('*'*55)
    print(f"Best Validation ROC AUC: {best_val_roc_auc} at Epoch:{best_epoch}")
    print('*'*55)