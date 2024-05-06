import torch
import torch.nn as nn
from .utils import sample_and_group, Local_op, PointNetFeaturePropagation
import torch.nn.functional as F
    
class Pointnet2Seg(nn.Module):
    def __init__(self, in_channels = 3, labels=8, radius = 1):
        super(Pointnet2Seg, self).__init__()
        self.radius = radius
        self.emb1 = nn.Conv1d(in_channels, 64, kernel_size=1, bias=False)
        self.emb2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.emb1_bn = nn.BatchNorm1d(64)
        self.emb2_bn = nn.BatchNorm1d(64)

        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        self.featureProp_0 = PointNetFeaturePropagation(in_channel=256+128, mlp=[256, 512], drp_add=True) # With skip connections
        self.featureProp_1 = PointNetFeaturePropagation(in_channel=512+64, mlp=[512, 1024], drp_add=True) # With skip connections


        self.conv1 = nn.Conv1d(1024, 512, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(512)
        
        self.conv2 = nn.Conv1d(512, 128, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(128)

        self.output = nn.Conv1d(128, labels, kernel_size=1)
 
    
    def forward(self, x):
        xyz = x
        x = x.permute(0,2,1)

        x = F.relu(self.emb1_bn(self.emb1(x)))
        x = F.relu(self.emb2_bn(self.emb2(x)))

        features0 = x
        xyz1, features1 = sample_and_group(npoint=2048, nsample=32, xyz = xyz, points=x.permute(0, 2, 1), radius = self.radius)
        features1 = self.gather_local_0(features1)
        x = F.dropout(x, p=0.4)
        
        xyz2, features2 = sample_and_group(npoint=1024, nsample=32, xyz = xyz1, points=features1.permute(0, 2, 1), radius = self.radius)
        x = self.gather_local_1(features2)
        x = F.dropout(x, p=0.4)
        
        x = self.featureProp_0(xyz1 = xyz1.transpose(1,2), xyz2 = xyz2.transpose(1,2), points1=features1, points2=x)
        x = self.featureProp_1(xyz1 = xyz.transpose(1,2), xyz2 = xyz1.transpose(1,2), points1=features0, points2=x)
       
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.dropout(x, p=0.4)
        x = F.relu(self.bn2(self.conv2(x)))
        
        logits = self.output(x)
        return logits.permute(0,2,1)
    
if __name__ == "__main__":
    
    x = torch.rand(2,2048,3)

    model = Pointnet2Seg()
    y = model(x)
    print(y.size())