import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch.optim as optim
from ResUNet import ResUNet
from New_utils import (
    load_checkpoint,
    get_loaders,
    save_BestNumber
)
from tqdm.auto import tqdm
from sklearn.model_selection import KFold
import numpy as np
import cv2
import os

# Hyperparameters etc.
LEARNING_RATE = 1e-3  # 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(torch.cuda.device_count())
# print(torch.cuda.get_device_name())
BATCH_SIZE = 5
NUM_EPOCHS = 1000
IMAGE_HEIGHT = 200
IMAGE_WIDTH = 200
LOAD_MODEL = False
TRAIN_IMG_DIR = "data/train_images/"
TRAIN_MASK_DIR = "data/train_masks/"
TEST_IMG_DIR = "data/test_images/"
TEST_MASK_DIR = "data/test_masks/"
TEST_IMG_DIR_ADD = "data/Testing_additional/test_images/"
TEST_MASK_DIR_ADD = "data/Testing_additional/test_masks/"
Initial_train_loss = 10000000 # Lucky number
Initial_val_loss = 10000000 # Lucky number
seed = 100 # Lucky number
Kfold_num = 5

# TEST_additional_IMG_DIR = "data/testing/test_images/"
# TEST_additional_MASK_DIR = "data/testing/test_masks/"
# PIN_MEMORY = True
# NUM_WORKERS = 4

torch.manual_seed(seed)  # torch+CPU
torch.cuda.manual_seed(seed)  # torch+GPU
print(f"Now we are using {DEVICE}")
print('-'*30)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset) # number of samples
    num_batches = len(dataloader) # batches per epoch

    model.train() # Sets the model in training mode.
    epoch_loss, epoch_correct = 0, 0

    # for batch_i, (x, y) in enumerate(tqdm(dataloader, leave=False)):
    for batch_i, (x, y) in enumerate(tqdm(dataloader)):

        x, y = x.to(DEVICE), y.float().unsqueeze(1).to(DEVICE) # move data to GPU

        # Compute prediction loss
        pred = model(x)
        loss = loss_fn(pred, y)

        # Optimization by gradients
        optimizer.zero_grad() # set prevision gradient to 0
        loss.backward() # backpropagation to compute gradients
        optimizer.step() # update model params

        # write to logs
        epoch_loss += loss.item()

    # return avg loss of epoch
    return epoch_loss/num_batches

def validation(dataloader, model, loss_fn):
    size = len(dataloader.dataset) # number of samples
    num_batches = len(dataloader) # batches per epoch

    model.eval() # Sets the model in test mode.
    epoch_loss = 0

    # No training for test data
    with torch.no_grad():
        for batch_i, (x, y) in enumerate(tqdm(dataloader, leave=False)):
            x, y = x.to(DEVICE), y.float().unsqueeze(1).to(DEVICE)

            pred = model(x)
            loss = loss_fn(pred, y)

            epoch_loss += loss.item()
    return epoch_loss/num_batches

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset) # number of samples
    num_batches = len(dataloader) # batches per epoch

    model.eval() # Sets the model in test mode.
    epoch_loss = 0

    # No training for test data
    with torch.no_grad():
        for batch_i, (x, y) in enumerate(tqdm(dataloader, leave=False)):
            x, y = x.to(DEVICE), y.float().unsqueeze(1).to(DEVICE)

            pred = model(x)
            loss = loss_fn(pred, y)

            epoch_loss += loss.item()
    return epoch_loss/num_batches

def save_images(dataloader, model, DEVICE, fold_i, epoch, folder='saved_'):
    model.eval() # Sets the model in test mode.

    # No training for test data
    with torch.no_grad():
        for batch_i, (x, y) in enumerate(dataloader):
            x, y = x.to(DEVICE), y.float().unsqueeze(1).to(DEVICE)
            preds = model(x)
            # print(f"Now x size is {x.size()}")
            # print(f"Now y size is {y.size()}")
            # print(f"Now pred size is {pred.size()}")
            Every_size = list(preds.size())
            for i in range(Every_size[0]):
                new_preds = preds.squeeze(dim=1)
                # print(f"Now preds size is {new_preds.size()}")
                new_preds = (new_preds[i]/(new_preds[i].max())).mul(255).detach().cpu().numpy()

                new_y = y.squeeze(dim=1)
                # print(f"Now y size is {new_y.size()}")
                new_y = (new_y[i]/(new_y[i].max())).mul(255).detach().cpu().numpy()

                new_x=x.squeeze(dim=1)
                # print(f"Now x size is {new_x.size()}")
                new_x = (new_x[i]/(new_x[i].max())).mul(255).detach().cpu().numpy()

                # print(f'After move to cpu/numpy')
                # print(f'The size of x is {new_x.shape}')
                # print(f'The size of y is {new_y.shape}')
                # print(f'The size of pred is {new_preds.shape}')

                filename_input = (str(batch_i) + "_" + str(i) + "_input.jpg")
                filename_y = (str(batch_i) + "_" + str(i) + ".jpg")
                filename_preds = (str(batch_i) + "_" + str(i) + "_pred.jpg")

                path = os.getcwd()
                if not os.path.isdir(os.path.join(path,str(folder)+"fold_"+str(fold_i)+"_epoch_"+str(epoch))):
                    os.mkdir(os.path.join(path,str(folder)+"fold_"+str(fold_i)+"_epoch_"+str(epoch)))
                
                Final_x = (os.path.join(path,str(folder)+"fold_"+str(fold_i)+"_epoch_"+str(epoch)) +"\\"+ filename_input)
                Final_y = (os.path.join(path,str(folder)+"fold_"+str(fold_i)+"_epoch_"+str(epoch)) +"\\"+ filename_y)
                Final_pred = (os.path.join(path,str(folder)+"fold_"+str(fold_i)+"_epoch_"+str(epoch)) +"\\"+ filename_preds)
                # print(Final_x)
                # print(Final_y)
                # print(Final_pred)
                cv2.imwrite(Final_x, new_x)
                cv2.imwrite(Final_y, new_y)
                cv2.imwrite(Final_pred ,new_preds)
def SaveTrainAndTestLoss(epoch, kfold,loss, pathName):
    loss = np.array(loss)
    path = os.getcwd()
    # np.save(path+'/loss_train'+'/fold_{}'.format(kfold)+'epoch_{}'.format(epoch),loss) 
    np.save(path+pathName+'/fold_{}'.format(kfold)+'epoch_{}'.format(epoch),loss) 



def main(Initial_train_loss,Initial_val_loss):
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

    test_transform = A.Compose(
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

    model = ResUNet().to(DEVICE)
    loss_fn = nn.MSELoss()
    # Not used# loss_fn = nn.L1Loss()  # MAE loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) 

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
        
    train_ds, test_ds = get_loaders(TRAIN_IMG_DIR, TRAIN_MASK_DIR, TEST_IMG_DIR, TEST_MASK_DIR, train_transform, test_transform)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE)
    kfold = KFold(n_splits=Kfold_num)
    torch.save(model.state_dict(), "my_checkpoint.pth.tar")

    fold_losses = []    


    for fold_i, (train_ids, val_ids) in enumerate(kfold.split(train_ds)):
        print(f'Now is kfold number {fold_i}')
        print(f'Sample of training: {len(train_ids)}, Sample of validation: {len(val_ids)}')
        # Reset model parameters
        model.load_state_dict(torch.load("my_checkpoint.pth.tar"))

        # Sample elements from selected ids
        train_sampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_ids)

        # Use sampler to select data for training and validation
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=val_sampler)
        optimizer = torch.optim.Adam(params=model.parameters())

        for epoch in tqdm(range(NUM_EPOCHS), leave=False):
            train_loss = train(train_loader, model, loss_fn, optimizer)
            val_loss = validation(val_loader, model, loss_fn)
            if val_loss <= Initial_val_loss:
                Initial_val_loss = val_loss
                print("\nObtain better parameter!")
                torch.save(model.state_dict(), "my_checkpoint.pth.tar")
                save_BestNumber(epoch,fold_i)
            print("-"*30)
            print("Save the training loss and validation loss")
            SaveTrainAndTestLoss(epoch, fold_i, train_loss, "/loss_train")
            SaveTrainAndTestLoss(epoch, fold_i, val_loss, "/loss_val")

            if epoch == 299:
                print("-"*30)
                print("Now save the validation images of every Epoch")
                save_images(val_loader, model, DEVICE, fold_i, epoch)


        # Test
        print("-"*30)
        print("Now save the testing images of different kfold")
        test_loss = test(test_loader, model, loss_fn)
        SaveTrainAndTestLoss(epoch, fold_i, val_loss, "/loss_test")
        save_images(test_loader, model, DEVICE, fold_i, epoch, folder='Test_')


        fold_losses.append(test_loss)
    print(f"Loss: mean {np.mean(fold_losses):.3f}, std: {np.std(fold_losses):.3f}")

def main_v2():
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

    test_transform = A.Compose(
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

    model = ResUNet().to(DEVICE)
    loss_fn = nn.MSELoss()
    # Not used# loss_fn = nn.L1Loss()  # MAE loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) 

    model.load_state_dict(torch.load("my_checkpoint.pth.tar"))
        
    train_ds, test_ds = get_loaders(TRAIN_IMG_DIR, TRAIN_MASK_DIR, TEST_IMG_DIR_ADD, TEST_MASK_DIR_ADD, train_transform, test_transform)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE)
     # Test
    print("-"*30)
    print("Now save the additional testing images")
    test_loss = test(test_loader, model, loss_fn)
    save_images(test_loader, model, DEVICE, 100, 100, folder='Test_additional_')

if __name__ == "__main__":
    # main(Initial_train_loss,Initial_val_loss)
    #  Run additional test
    main_v2()