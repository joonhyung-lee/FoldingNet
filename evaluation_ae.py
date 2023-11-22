#%%
import torch
from model import AutoEncoder
from utils import show_point_cloud, show_point_clouds
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
