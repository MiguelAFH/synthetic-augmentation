import pandas as pd
from datasets import load_dataset, Image

ds_name = "alkzar90/NIH-Chest-X-ray-dataset"

# Load dataset with the specified data directory
dataset_with_image_data = load_dataset(ds_name, 'image-classification', data_dir='./data')
dataset = load_dataset(ds_name, 'image-classification', data_dir='./data').cast_column('image', Image(decode=False))

# Convert dataset to DataFrames
# train_data = pd.DataFrame(dataset['train'])
# test_data = pd.DataFrame(dataset['test'])

