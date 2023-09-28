import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch.optim as optim
from ResUNet import ResUNet
from UNet import UNet
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    val_loss,
    save_predictions_as_imgs,
)
from tqdm import tqdm
import os
import numpy as np

# Hyperparameters etc.
LEARNING_RATE = 1e-3  # 1e-4
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# print(torch.cuda.device_count())
# print(torch.cuda.get_device_name())
BATCH_SIZE = 10
NUM_EPOCHS = 1500
IMAGE_HEIGHT = 200
IMAGE_WIDTH = 200
LOAD_MODEL = False
TRAIN_IMG_DIR = "data/train_images/"
TRAIN_MASK_DIR = "data/train_masks/"
VAL_IMG_DIR = "data/val_images/"
VAL_MASK_DIR = "data/val_masks/"
TEST_IMG_DIR = "data/testing/test_images/"
TEST_MASK_DIR = "data/testing/test_masks/"
PIN_MEMORY = True
NUM_WORKERS = 4
Initial_train_loss = 10000000 # Lucky number
Initial_val_loss = 10000000 # Lucky number