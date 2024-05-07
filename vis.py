import numpy as np
import open3d as o3d
import laspy 
import os 
import torch
from pointnet import PointnetSeg
from pointnet2 import Pointnet2Seg
from dataset import Dales
import torch.nn as nn
from torch.utils.data import DataLoader
from train import test_loop
from sklearn.metrics import classification_report
np.random.seed(42)


colors = np.random.rand(8,3)
def visualize(data, label):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    color = np.zeros((len(data), 3))
    for j in range(8):
        color[label == j] += colors[j]

    pcd.colors = o3d.utility.Vector3dVector(color)

    return pcd

def visualize_model(model_name):
    loader = DataLoader(Dales('cuda', 25, 4096, partition='test',not_norm=True), batch_size = 8)
    tiles = np.load(os.path.join("data", "Dales" , 'test', f"not_norm_25_4096.npz"))
    data = tiles['x'].reshape(-1, 3)
    label = tiles['y'].reshape(-1)
    model = Pointnet2Seg().to('cuda')

    model.load_state_dict(torch.load(os.path.join("models", "best", f"{model_name}.pt")))
    loss_fn = nn.CrossEntropyLoss()
    _,acc,bal_acc,preds = test_loop(loader, loss_fn, model, 'cuda')
    print(f'{acc=}')
    print(f'{bal_acc=}')
    targets_names = ['ground', 'vegetation', 'cars', 'trucks', 'power_lines', 'fences', 'poles', 'buildings']
    preds = np.asarray(preds).reshape(-1)
    print(classification_report(label, preds,target_names=targets_names))
    pcd = visualize(data, preds)

    return pcd

if __name__ == "__main__":

    pcd = visualize_model('pointnet2_25_4096_3')

    o3d.visualization.draw_geometries([pcd])


    # label = tiles['y']
    # import glob
    # for fl in glob.glob(os.path.join("data", "Dales", "test","*.las")):
    #     las = laspy.read(fl)


    # gt_mesh = o3d.io.read_triangle_mesh()
    # gt_mesh.points = o3d.utility.Vector3dVector(las.xyz)
    # gt_mesh.estimate_normals()
    # gt_mesh.compute_vertex_normals()

    # pcd = gt_mesh.sample_points_poisson_disk(3000)
    # o3d.visualization.draw_geometries([pcd])
    # print(data)







