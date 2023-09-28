from Dataset import LoadDataset
from New_Dataset import LoadDataset
import os
from pathlib import Path
import torch
import tqdm
import cv2

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint...\n")
    model.load_state_dict(checkpoint["state_dict"])
    
def get_loaders(
    train_dir,
    train_maskdir,
    test_dir,
    test_maskdir,
    train_transform,
    test_transform,

):
    train_ds = LoadDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    test_ds = LoadDataset(
        image_dir=test_dir,
        mask_dir=test_maskdir,
        transform=test_transform,
    )
    print(f'Before training,')
    print(f'The number of training is {len(train_ds)} dataset')
    print(f'The number of testing is {len(test_ds)} dataset')
    print('-'*30)
    return train_ds, test_ds
def save_BestNumber(epoch, fold_number):
    # Create the txt to record the parameters
    # Here need to mention
    path = os.getcwd()
    if not os.path.isdir(os.path.join(path,"Best.txt")):
            Path('Best.txt').touch()
    f = open("Best.txt", "w+")
    f.write("Now the best epoch number is %d"%epoch)
    f.write("Best kfold number is %d"%fold_number)
    f.close()


