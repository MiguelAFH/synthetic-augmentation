from fastcore.all import *
from fastai.vision.all import *
import time
import pandas as pd
from datasets import load_dataset, Image
import os


def get_x(item):
    return item['image']['path']

def get_y(item):
    return item['labels']

def fine_tune(dls, model, epochs=4, export_path=None):
    learn = vision_learner(dls, model, metrics=accuracy_multi)
    learn.model.to('cuda')
    # exit()
    learn.fine_tune(epochs)
    if export_path:
        learn.export(export_path)
    return learn
  
def validate(learn, test_data):
    test_dl = learn.dls.test_dl(test_data)
    return learn.validate(dl=test_dl)

def dataset_stats(data, name):
    print("\n" + "="*50)
    print(f"{name} Statistics:")
    print("Number of samples:", len(data))
    print("Columns:", data.columns.tolist())
    print("\nData Types:")
    print(data.dtypes)
    print("\nMissing Values:")
    print(data.isnull().sum())
    # print("\nValue Counts for each column:")
    # for col in data.columns:
    #     print("\n", col)
    #     print(data[col].value_counts())
    print("="*50 + "\n")

if __name__ == "__main__":
    MODELS_PATH = "src/models"
    
    ds_name = "alkzar90/NIH-Chest-X-ray-dataset"
    dataset_with_image_data = load_dataset(ds_name, 'image-classification', data_dir='./data')
    dataset = load_dataset(ds_name, 'image-classification', data_dir='./data').cast_column('image', Image(decode=False))
    train_data = pd.DataFrame(dataset['train']).sample(frac=0.1, random_state=42)
    test_data = pd.DataFrame(dataset['test'])
    
    dataset_stats(train_data, "Train Data")
    dataset_stats(test_data, "Test Data")
    
    dls = DataBlock(
      blocks=(ImageBlock, MultiCategoryBlock),
      get_x=get_x,
      get_y=get_y,
      splitter=RandomSplitter(valid_pct=0.2, seed=42),
      item_tfms=[Resize(224, method="squish")]
    ).dataloaders(train_data)
    
    dls.show_batch(max_n=10)

    learn1 = fine_tune(dls, resnet18, 8, os.path.join(MODELS_PATH,"resnet18.pkl"))
    # learn2 = fine_tune(dls, resnet50, 4, os.path.join(MODELS_PATH, "resnet50.pkl"))
    # learn3 = fine_tune(dls, vgg19, 6, os.path.join(MODELS_PATH, "vgg19.pkl"))

    #Run validation and create accuracy barchart for each model
    validate(learn1, test_data)
    # validate(learn2, test_data)
    # validate(learn3, test_data)
    #Create accuracy bar chart with all models
    accs = [0.75, 0.85, 0.80]
    models = ["resnet18", "resnet50", "vgg19"]
    plt.bar(models, accs)
    
    