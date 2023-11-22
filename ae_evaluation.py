#%%
import torch
from model import AutoEncoder
from utils import show_point_cloud
from loss import ChamferLoss

model = AutoEncoder()
model.load_state_dict(torch.load('log/model_lowest_cd_loss.pth'))
device = torch.device('cuda')
model.to(device)
cd_loss = ChamferLoss()

#%%
import os
import torch
import numpy as np
from torch.utils.data import Dataset

class CustomPointCloudDataset(Dataset):
    def __init__(self, root_dir, npoints=2048, data_augmentation=False):
        self.root_dir = root_dir
        self.npoints = npoints
        self.data_augmentation = data_augmentation
        self.file_list = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.file_list[idx])
        point_cloud = np.load(file_path)  # Assuming the point cloud files are NumPy arrays
        # Downsample or process the point cloud as needed
        # Apply data augmentation if required
        return torch.tensor(point_cloud[:self.npoints], dtype=torch.float)

# Usage
test_dataset = CustomPointCloudDataset(root_dir='./dataset/pcds_3', npoints=2048, data_augmentation=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

#%%
# evaluation
model.eval()
total_cd_loss = 0
with torch.no_grad():
    for data in test_dataloader:
        point_clouds = data
        point_clouds = point_clouds.permute(0, 2, 1)
        point_clouds = point_clouds.to(device)
        recons = model(point_clouds)
        ls = cd_loss(point_clouds.permute(0, 2, 1), recons.permute(0, 2, 1))
        total_cd_loss += ls.item()
        show_point_cloud(point_clouds.cpu().numpy())
        show_point_cloud(recons.cpu().numpy())
# calculate the mean cd loss
mean_cd_loss = total_cd_loss / len(test_dataset)
print('Mean Chamfer Distance of all Point Clouds:', mean_cd_loss)

# %%

# %%
show_point_cloud(recons.cpu().numpy())
