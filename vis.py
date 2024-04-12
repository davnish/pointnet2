import numpy as np
import open3d as o3d
import laspy 
import os 
import torch
from model import PointTransformerSeg
from dataset import Dales
from torch.utils.data import DataLoader
from main import test_loop
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
import torch.nn as nn

def visualize_model(model_name):
    loader = DataLoader(Dales('cuda', 25, 4096, partition='test',not_norm=True), batch_size = 8)
    tiles = np.load(os.path.join("data", "Dales" , 'test', f"not_norm_25_4096.npz"))
    data = tiles['x'].reshape(-1, 3)
    label = tiles['y'].reshape(-1)
    model = PointTransformerSeg().to('cuda')

    model.load_state_dict(torch.load(os.path.join("model", "checkpoint", f"{model_name}.pt")))
    loss_fn = nn.CrossEntropyLoss()
    _,acc,bal_acc,preds = test_loop(loader, loss_fn, model, 'cuda')
    print(f'{acc=}')
    print(f'{bal_acc=}')
    targets_names = ['ground', 'vegetation', 'cars', 'trucks', 'power_lines', 'fences', 'poles', 'buildings']
    preds = np.asarray(preds).reshape(-1)
    print(classification_report(label, preds,target_names=targets_names))
    pcd = visualize(data, preds)

    return pcd


pcd = visualize_model(11)

o3d.visualization.draw_geometries([pcd])







