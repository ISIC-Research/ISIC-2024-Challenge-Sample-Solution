

#Download the data from Kaggle 

import os
import zipfile

# Create a directory to store the downloaded data
download_path = 'Data/'
os.makedirs(download_path, exist_ok=True)

# Download the ISIC 2024 dataset
os.system(f'kaggle competitions download -c isic-2024-challenge -p {download_path}')

# Unzip all downloaded files
for file in os.listdir(download_path):
    if file.endswith('.zip'):
        with zipfile.ZipFile(os.path.join(download_path, file), 'r') as zip_ref:
            zip_ref.extractall(download_path)
            print(f'Extracted {file}')