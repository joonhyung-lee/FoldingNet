#%%
import random
import torch
from datasets import ShapeNetPartDataset
from model import AutoEncoder
from loss import ChamferLoss
# from chamfer_distance.chamfer_distance import ChamferDistance
from utils import show_point_cloud, show_point_cloud_with_pyvista


ae = AutoEncoder()
# ae.load_state_dict(torch.load('./log/model_lowest_cd_loss.pth'))
ae.load_state_dict(torch.load('./log/model_epoch_2590.pth'))
ae.eval()

DATASET_PATH = './dataset/shapenetcore_partanno_segmentation_benchmark_v0'
test_dataset = ShapeNetPartDataset(root=DATASET_PATH, npoints=1024, split='train', classification=False, data_augmentation=False, class_choice='Table')
# Randomly select a point cloud from the test dataset
for _ in range(10):
    input_pc = test_dataset[random.randint(0, len(test_dataset))][0]
    show_point_cloud(input_pc, title="Input Point Cloud")
    # show_point_cloud_with_pyvista(input_pc.numpy())

    input_tensor = input_pc.unsqueeze(0).permute(0, 2, 1)
    output_tensor = ae(input_tensor)
    reconstructed_pc = output_tensor.permute(0, 2, 1).squeeze().detach().numpy()

    show_point_cloud(reconstructed_pc, title="Reconstructed Point Cloud")
    # show_point_cloud_with_pyvista(reconstructed_pc)

#%%
cd_loss = ChamferLoss()
print(cd_loss(input_tensor, output_tensor))

# %%
