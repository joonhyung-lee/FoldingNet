#%%
import wandb
import argparse
import os
import time
from datetime import datetime
import numpy as np

import torch
import torch.optim as optim
from datasets import ShapeNetPartDataset
from model import AutoEncoder
# from chamfer_distance.chamfer_distance import ChamferDistance
from loss import ChamferLoss
from utils import show_point_cloud, show_point_cloud_with_pyvista, subsample_arrays
#%%

def get_runname():
    now = datetime.now()
    format = "%m%d:%H%M"
    runname = now.strftime(format)
    return runname

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='./dataset/seed_4_all')
parser.add_argument('--npoints', type=int, default=2048)
parser.add_argument('--runname', type=str, default='stable_ae_01')
parser.add_argument('--name', type=str, default='stable')
parser.add_argument('--mpoints', type=int, default=2025)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-6)
parser.add_argument('--epochs', type=int, default=500000)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--log_dir', type=str, default='./weights/stable')
parser.add_argument('--WANDB', type=bool, default=False)

args = parser.parse_args(args=[])

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

# Set logger 
runname = get_runname() if args.runname=='None' else args.runname
if args.WANDB:
    wandb.init(project = args.name)
    wandb.run.name = runname   

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
len_data = len(os.listdir(args.root))
print(f'Number of scenes: {len_data}')
pcds_list = []
for idx in range(1, len_data + 1):
    pcds_load = np.load(os.path.join(f'{args.root}/all_data_{idx:03d}.npz'))['pointcloud']
    # pcds_load = np.load(os.path.join(f'{args.root}/all_data_{idx:03d}.npz'))['stable_pcds']
    pcds_list.append(pcds_load)

# Get min and max of shape
pcds_list_shape = [arr.shape for arr in pcds_list]
min_shape = np.min(pcds_list_shape, axis=0)
max_shape = np.max(pcds_list_shape, axis=0)
print(f'Min shape: {min_shape}, Max shape: {max_shape}')
pcds_subsampled = subsample_arrays(pcds_list, target_size=args.npoints)
shapes_of_subsampled_arrays = [arr.shape for arr in pcds_subsampled]
print(f'Subsampled shapes: {np.array(shapes_of_subsampled_arrays).shape}')
print(f'Length of pcds_list: {len(pcds_list)}')
print(f'Batch size: {args.batch_size}')
pcds_torch = torch.tensor(np.array(pcds_subsampled)).float()
print(f"pcds_torch.shape: {pcds_torch.shape}")

from torch.utils.data import DataLoader, random_split, Dataset

model = AutoEncoder()
device = torch.device('cpu')
model.to(device)
cd_loss = ChamferLoss()

class CustomPointCloudDataset(Dataset):
    def __init__(self, root_dir, npoints=2048, data_name="pointcloud"):
        self.root_dir = root_dir
        self.npoints = npoints
        self.data_name = data_name
        self.file_list = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.file_list[idx])
        data_dict = np.load(file_path, allow_pickle=True)

        if self.data_name not in data_dict:
            raise KeyError(f"Data dictionary must contain '{self.data_name}' key.")
        point_cloud = data_dict[f"{self.data_name}"]
        if len(point_cloud) > self.npoints:
            indices = np.random.choice(len(point_cloud), self.npoints, replace=False)
            point_cloud = point_cloud[indices]

        return torch.tensor(point_cloud, dtype=torch.float)

# Load the entire dataset
full_dataset = CustomPointCloudDataset(root_dir='./dataset/seed_4_all', npoints=2048, data_name="pointcloud")
full_dataset = CustomPointCloudDataset(root_dir='./dataset/seed_4_all', npoints=2048, data_name="stable_pcds")

# Split the dataset into train and test sets (80% train, 20% test)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

for idx, data in enumerate(train_dataset):
    print(f"idx: {idx}, data.shape: {data.shape}")
    break

#%%
for batch_idx, data in enumerate(train_dataset):
    print(f"Batch {batch_idx}, data.shape: {data.shape}")
    break

#%%
# model
autoendocer = AutoEncoder()
autoendocer.to(device)

# loss function
# cd_loss = ChamferDistance()
cd_loss = ChamferLoss()
# optimizer
optimizer = optim.Adam(autoendocer.parameters(), lr=args.lr, betas=[0.9, 0.999], weight_decay=args.weight_decay)

batches = int(len(train_dataset) / args.batch_size + 0.5)

min_cd_loss = 1e3
best_epoch = -1
#%%
print("Test shape error")
for idx, data in enumerate(train_dataset):
    data = data.unsqueeze(0)
    point_clouds = data.to(device).permute(0, 2, 1)
    recons = autoendocer(point_clouds)
    ls = cd_loss(point_clouds.permute(0, 2, 1), recons.permute(0, 2, 1))
    print(f"idx: {idx}, data.shape: {data.shape}")
    break

#%%
print('\033[31mBegin Training...\033[0m')
for epoch in range(1, args.epochs + 1):
    # training
    start = time.time()
    autoendocer.train()
    for i, data in enumerate(train_dataset):
        point_clouds = data.unsqueeze(0)
        point_clouds = point_clouds.permute(0, 2, 1)
        point_clouds = point_clouds.to(device)
        recons = autoendocer(point_clouds)
        ls = cd_loss(point_clouds.permute(0, 2, 1), recons.permute(0, 2, 1))
        # ls = cd_loss(point_clouds.permute(0, 2, 1), recons.permute(0, 2, 1))
        # show_point_cloud(point_clouds.cpu().numpy())
        # show_point_cloud(recons.detach().cpu().numpy())
        print(f"ls: {ls}")
        
        optimizer.zero_grad()
        ls.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch {}/{} with iteration {}/{}: CD loss is {}.'.format(epoch, args.epochs, i + 1, batches, ls.item() / len(point_clouds)))
            # Log training loss
            wandb.log({"Chamfer Loss": ls.item() / len(point_clouds)})

    # evaluation
    autoendocer.eval()
    total_cd_loss = 0
    with torch.no_grad():
        for data in test_dataset:
            point_clouds = data.unsqueeze(0)
            point_clouds = point_clouds.permute(0, 2, 1)
            point_clouds = point_clouds.to(device)
            recons = autoendocer(point_clouds)
            ls = cd_loss(point_clouds.permute(0, 2, 1), recons.permute(0, 2, 1))
            # ls = cd_loss(point_clouds.permute(0, 2, 1), recons.permute(0, 2, 1))
            total_cd_loss += ls.item()
    
    # calculate the mean cd loss
    mean_cd_loss = total_cd_loss / len(test_dataset)

    # records the best model and epoch
    if mean_cd_loss < min_cd_loss:
        min_cd_loss = mean_cd_loss
        best_epoch = epoch
        torch.save(autoendocer.state_dict(), os.path.join(args.log_dir, f'model_{args.name}_lowest_cd_loss.pth'))
    
    # save the model every 100 epochs
    if (epoch) % 100 == 0 or epoch == 1:
        torch.save(autoendocer.state_dict(), os.path.join(args.log_dir, f'model_{args.name}_epoch_{epoch}.pth'))
    
    end = time.time()
    cost = end - start

    print('\033[32mEpoch {}/{}: reconstructed Chamfer Distance is {}. Minimum cd loss is {} in epoch {}.\033[0m'.format(
        epoch, args.epochs, mean_cd_loss, min_cd_loss, best_epoch))
    print('\033[31mCost {} minutes and {} seconds\033[0m'.format(int(cost // 60), int(cost % 60)))

    if args.WANDB:

        custom_chart = wandb.plot.line_series(
            xs=[epoch for epoch in range(1, args.epochs + 1)],
            ys=[[mean_cd_loss], [mean_cd_loss]],  # Training and validation loss
            keys=["Training Loss", "Validation Loss"],
            title="Loss Over Epochs",
            xname="Epoch"
        )

        # Log training loss
        wandb.log({"Training Loss": mean_cd_loss})
        # Log evaluation metrics
        wandb.log({"Validation Loss": mean_cd_loss})
        # Log the custom chart
        wandb.log({"Loss Over Epochs": custom_chart})

# %%
