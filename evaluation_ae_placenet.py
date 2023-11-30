#%%
import torch
from model import AutoEncoder
from utils import show_point_cloud, show_point_clouds
from utils import show_point_cloud, show_point_cloud_with_pyvista, subsample_arrays
from loss import ChamferLoss
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import random_split, Dataset

model = AutoEncoder()
# model.load_state_dict(torch.load('log/model_lowest_cd_loss.pth'))
model.load_state_dict(torch.load('./weights/scene/model_scene_epoch_1300.pth'))
# model.load_state_dict(torch.load('./weights/stable/model_stable_epoch_1700.pth'))
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
# full_dataset = CustomPointCloudDataset(root_dir='./dataset/seed_4_all', npoints=2048, data_name="stable_pcds")

# Split the dataset into train and test sets (80% train, 20% test)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

for idx, data in enumerate(test_dataset):
    print(f"idx: {idx}, data.shape: {data.shape}")
    break

#%%
# Set the model to evaluation mode
model.eval()
total_cd_loss = 0
with torch.no_grad():
    for data in test_dataset:  # Iterate over batches in the test dataloader
        point_clouds = data.unsqueeze(0)  # Add a batch dimension
        print(f"point_clouds.shape: {point_clouds.shape}")
        point_clouds = point_clouds.permute(0, 2, 1)  # Permute dimensions
        point_clouds = point_clouds.to(device)  # Move data to the device (e.g., GPU)
        print(f"point_clouds.shape: {point_clouds.shape}")
        recons = model(point_clouds)
        ls = cd_loss(point_clouds.permute(0, 2, 1), recons.permute(0, 2, 1))
        # ls = cd_loss(point_clouds, recons)
        total_cd_loss += ls.item()
        
        show_point_cloud(point_clouds.cpu().numpy(), title="Input Point Cloud")
        show_point_cloud(recons.cpu().numpy(), title="Reconstructed Point Cloud")

# Calculate the mean CD loss
mean_cd_loss = total_cd_loss / len(test_dataset)
print('Mean Chamfer Distance of all Point Clouds:', mean_cd_loss)

# %%
show_point_cloud(recons.cpu().numpy())

#%%
from utils import passthrough_filter

pcd_masked = passthrough_filter(point_clouds[0].permute([1,0]).cpu().numpy(), axis=0, interval=[0.4, 1.6])
pcd_masked = passthrough_filter(pcd_masked, axis=1, interval=[-0.5, 0.5])
pcd_masked = passthrough_filter(pcd_masked, axis=2, interval=[0.5, 1.3])

show_point_cloud(pcd_masked)
show_point_cloud(point_clouds.cpu().numpy())
# %%
recons = model(torch.from_numpy(pcd_masked).permute([1,0]).unsqueeze(0).to(device))
show_point_cloud(recons.cpu().detach().numpy())

# %%
point_clouds = []
point_clouds.append(pcd_masked)
point_clouds.append(recons.cpu().detach().numpy()[0])

show_point_clouds(point_clouds)
# %%
show_point_cloud(pcd_masked)
show_point_cloud(recons.cpu().detach().numpy())

# %%
import pyvista as pv

pv_plotter = pv.Plotter()
pv_plotter.add_mesh(pv.PolyData(pcd_masked), color='red')
pv_plotter.add_mesh(pv.PolyData(recons.permute([0,2,1]).cpu().detach().numpy()[0]), color='blue')
pv_plotter.show()
# %%

#%%
#%%
# root = './dataset/seed_4_all'
# len_data = len(os.listdir(root))
# print(f'Number of scenes: {len_data}')
# pcds_list = []
# for idx in range(1, len_data + 1):
#     pcds_load = np.load(os.path.join(f'{root}/all_data_{idx:03d}.npz'))['pointcloud']
#     pcds_list.append(pcds_load)

# # Get min and max of shape
# batch_size = 1
# pcds_list_shape = [arr.shape for arr in pcds_list]
# min_shape = np.min(pcds_list_shape, axis=0)
# max_shape = np.max(pcds_list_shape, axis=0)
# print(f'Min shape: {min_shape}, Max shape: {max_shape}')
# pcds_subsampled = subsample_arrays(pcds_list, target_size=args.npoints)
# shapes_of_subsampled_arrays = [arr.shape for arr in pcds_subsampled]
# print(f'Subsampled shapes: {np.array(shapes_of_subsampled_arrays).shape}')
# print(f'Length of pcds_list: {len(pcds_list)}')
# print(f'Batch size: {batch_size}')
# pcds_torch = torch.tensor(np.array(pcds_subsampled)).float()
# print(f"pcds_torch.shape: {pcds_torch.shape}")

# from torch.utils.data import DataLoader, random_split
# # Split with ratio args.ratio.
# split_ratio = 0.8
# train_size = int(split_ratio * len(pcds_torch))
# test_size = len(pcds_torch) - train_size
# train_dataset, test_dataset = random_split(pcds_torch, [train_size, test_size])
# train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
# test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)