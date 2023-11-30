import random

import numpy as np
import torch
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D


import pyvista as pv

def show_point_cloud_with_pyvista(point_cloud):
    """
    Visualize a point cloud using PyVista.

    Args:
        point_cloud (np.ndarray): The coordinates of the point cloud.
    """
    # Create a PyVista plotter
    plotter = pv.Plotter()

    # Create a point cloud mesh from the numpy array
    cloud_mesh = pv.PolyData(point_cloud)

    # Add the point cloud mesh to the plotter
    plotter.add_mesh(cloud_mesh, point_size=5)

    # Show the plot
    plotter.show()



def show_point_cloud(point_cloud, axis=False, title='Point Cloud', xlabel='X-axis', ylabel='Y-axis', zlabel='Z-axis'):
    """
    Visualize a point cloud.

    Args:
        point_cloud (np.ndarray): The coordinates of the point cloud.
        axis (bool, optional): Hide the coordinate of the matplotlib. Defaults to False.
        title (str, optional): Title of the plot. Defaults to 'Point Cloud'.
        xlabel (str, optional): Label for the X-axis. Defaults to 'X-axis'.
        ylabel (str, optional): Label for the Y-axis. Defaults to 'Y-axis'.
        zlabel (str, optional): Label for the Z-axis. Defaults to 'Z-axis'.
    """
    ax = plt.figure().add_subplot(projection='3d')
    ax._axis3don = axis
    ax.scatter(xs=point_cloud[:, 0], ys=point_cloud[:, 1], zs=point_cloud[:, 2], s=5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.show()

def show_point_clouds(point_clouds, axis=False, device='cuda'):
    """visual a point cloud
    Args:
        point_cloud (np.ndarray): the coordinates of point cloud
        axis (bool, optional): Hid the coordinate of the matplotlib. Defaults to False.
    """
    ax = plt.figure().add_subplot(projection='3d')
    for idx, point_cloud in enumerate(point_clouds):
        pcd_np = np.array(point_cloud)
        pcd_torch = torch.from_numpy(pcd_np).permute([1,0]).unsqueeze(0).to(device)
        ax.scatter(xs=pcd_torch.cpu().detach().numpy()[0, :, 0], ys=pcd_torch.cpu().detach().numpy()[0, :, 1], zs=pcd_torch.cpu().detach().numpy()[0, :, 2], s=5)
    ax._axis3don = False
    plt.show()

def setup_seed(seed):
    """
    Set the random seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# passthrough filter about specific axis
def passthrough_filter(pcd, axis, interval):
    mask = (pcd[:, axis] > interval[0]) & (pcd[:, axis] < interval[1])
    return pcd[mask]

def index_points(point_clouds, index):
    """
    Given a batch of tensor and index, select sub-tensor.

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, N, k]
    Return:
        new_points:, indexed points data, [B, N, k, C]
    """
    device = point_clouds.device
    batch_size = point_clouds.shape[0]
    view_shape = list(index.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(index.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    new_points = point_clouds[batch_indices, index, :]
    return new_points


def knn(x, k):
    """
    K nearest neighborhood.

    Parameters
    ----------
        x: a tensor with size of (B, C, N)
        k: the number of nearest neighborhoods
    
    Returns
    -------
        idx: indices of the k nearest neighborhoods with size of (B, N, k)
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  # (B, N, N)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)  # (B, 1, N)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  # (B, 1, N), (B, N, N), (B, N, 1) -> (B, N, N)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (B, N, k)
    return idx


def to_one_hots(y, categories):
    """
    Encode the labels into one-hot coding.

    :param y: labels for a batch data with size (B,)
    :param categories: total number of kinds for the label in the dataset
    :return: (B, categories)
    """
    y_ = torch.eye(categories)[y.data.cpu().numpy()]
    if y.is_cuda:
        y_ = y_.cuda()
    return y_

def subsample_arrays(arrays_list, target_size=1024):
    """
    Subsamples each array in the list to a fixed number of rows (target_size).
    Arrays with fewer rows than target_size are excluded from the result.
    """
    arrays_list_shape = [arr.shape for arr in arrays_list]
    min_shape = np.min(arrays_list_shape, axis=0)
    max_shape = np.max(arrays_list_shape, axis=0)
    print(min_shape, max_shape)
    # if min_shape[0] < target_size:
    #     target_size = min_shape[0]
    subsampled_list = []
    for arr in arrays_list:
        if arr.shape[0] > target_size:
            # Select target_size rows randomly without replacement
            indices = np.random.choice(arr.shape[0], target_size, replace=False)
            subsampled_arr = arr[indices]
            subsampled_list.append(subsampled_arr)
    return subsampled_list


if __name__ == '__main__':
    pcs = torch.rand(32, 3, 1024)
    knn_index = knn(pcs, 16)
    print(knn_index.size())
    knn_pcs = index_points(pcs.permute(0, 2, 1), knn_index)
    print(knn_pcs.size())

