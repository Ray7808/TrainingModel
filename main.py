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

def train_fn(loader, Epoch, model, optimizer, loss_fn):
    loop = tqdm(loader)
    LossData=[]
    path = os.getcwd()
    loss_Train = 0

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = (targets.float().unsqueeze(1))
        targets = (targets).to(device=DEVICE)

        # forward
        predictions = model(data)
        #predictions = (predictions/predictions.max())

        loss = loss_fn(predictions, targets)
        loss_Train += loss
        # backward
        optimizer.zero_grad() # Set the gradient to zero
        loss.backward()  # Back-propogation
        optimizer.step()
        
        # update tqdm loop
        loop.set_postfix(loss=loss.item())  # Show on the terminal
    loss_np = loss_Train.cpu().detach().numpy()
    loss_np = loss_np / len(loader)
    
    LossData.append(loss_np)
    print('Epoch '+str(batch_idx)+' : ',' LOSS ='+str(LossData))
    Loss0 = np.array(LossData)
    np.save(path+'/loss_train'+'/epoch_{}'.format(Epoch),Loss0) 
    return Loss0[0]


def main(LEARNING_RATE):
    # Data augmentation
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.Normalize(
                mean=[0.0],
                std=[1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0],
                std=[1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    # model = UNet().to(DEVICE)
    model = ResUNet().to(DEVICE)

    loss_fn = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) 

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    scaler = torch.cuda.amp.GradScaler()
    test = 1
    for epoch in range(NUM_EPOCHS):
        print(f"Now the number of epoches is {epoch + 1}!\n")
        New_train_loss = train_fn(train_loader,epoch, model, optimizer, loss_fn)

        if int(epoch % 300) == test and epoch > 300:
            LEARNING_RATE = LEARNING_RATE * 0.9
            test = test + 1
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) 

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        global Initial_train_loss 
        global Initial_val_loss
        print(f"Now the previous training loss is {Initial_train_loss}, and the new loss is {New_train_loss}")
        if New_train_loss <= Initial_train_loss:
            print("Obtain better model parameters")
            save_checkpoint(checkpoint, epoch)
            Initial_train_loss = New_train_loss

        # load model and check val_loss
        validation_loss = val_loss(val_loader, epoch, model, loss_fn, device=DEVICE)
        print(f"Now the validation loss is {Initial_val_loss}, and the new loss is {validation_loss}")

        if validation_loss <= Initial_val_loss:
            Initial_val_loss = validation_loss
        save_predictions_as_imgs(
            val_loader, epoch, BATCH_SIZE, model, folder="saved_images_", device=DEVICE
        )

        print('Saved the predicted images\n')

def Get_Image():
        # Data augmentation
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.Normalize(
                mean=[0.0],
                std=[1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0],
                std=[1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    # model = UNet().to(DEVICE)
    model = ResUNet().to(DEVICE)

    loss_fn = nn.MSELoss()
    # Not used# loss_fn = nn.CrossEntropyLoss()  # Softmax + CrossEntropy
    # Not used# loss_fn = nn.L1Loss()  # MAE loss

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) 
    # Not used# optimizer = optim.SGD(model.parameters()

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        TEST_IMG_DIR,
        TEST_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
    scaler = torch.cuda.amp.GradScaler()

    save_predictions_as_imgs(
            val_loader, 1, BATCH_SIZE, model, folder="test_images_", device=DEVICE
    )

    print('Saved the predicted images\n')

def Get_Train_Image():
        # Data augmentation
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.Normalize(
                mean=[0.0],
                std=[1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0],
                std=[1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    # model = UNet().to(DEVICE)
    model = ResUNet().to(DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) 

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
    scaler = torch.cuda.amp.GradScaler()

    save_predictions_as_imgs(
            val_loader, 1, BATCH_SIZE, model, folder="trained_images_", device=DEVICE
    )

    print('Saved the predicted images\n')


if __name__ == "__main__":
    main(LEARNING_RATE)
