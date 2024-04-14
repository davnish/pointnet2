import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from pointnet_util import sample_and_group, Local_op, PointNetFeaturePropagation
torch.manual_seed(42)


class PointnetCls(nn.Module):
    def __init__(self, n_embd, dropout):
        _input = 3
        classes = 40
        super(PointnetCls, self).__init__()
        self.conv1 = nn.Conv1d(_input, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)

        self.ll1 = nn.Linear(1024, 512)
        self.ll2 = nn.Linear(512, 256)
        self.output = nn.Linear(256, classes) # This 40 is for modelnet40

        self.t1 = T1(dropout)
        self.t2 = T2(dropout)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    
    def forward(self, x):
        
        t1 = self.t1(x)
        x =  torch.bmm(t1, x)

        x = F.relu(self.bn1(self.conv1(x)))
       
        t2 = self.t2(x)
        x = torch.bmm(t2, x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.adaptive_max_pool1d(x, 1).view(x.size(0), -1)
        
        x = F.relu(self.bn4(self.ll1(x)))
        x = F.relu(self.bn5(self.ll2(x)))
        x = self.output(x)
        return x
    
class PointnetSeg(nn.Module):
    def __init__(self, n_embd=64, dropout=0.2):
        super(PointnetSeg, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)

        self.conv4 = nn.Conv1d(1088, 512, kernel_size=1)
        self.conv5 = nn.Conv1d(512, 256, kernel_size=1)
        self.conv6 = nn.Conv1d(256, 128, kernel_size=1)

        self.output = nn.Conv1d(128, 8, kernel_size=1) # This 8 is for Dales

        self.t1 = T1(dropout)
        self.t2 = T2(dropout)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.bn6 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        n_points = x.size()[2]
        t1 = self.t1(x)
        x =  torch.bmm(t1, x)

        x = F.relu(self.bn1(self.conv1(x)))
       
        t2 = self.t2(x)
        feature_transformed = torch.bmm(t2, x)

        x = F.relu(self.bn2(self.conv2(feature_transformed)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.adaptive_max_pool1d(x, 1).view(x.size(0), -1)
        x = torch.cat((feature_transformed, x.unsqueeze(-1).repeat(1, 1, n_points)), dim = 1)

        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))

        x = self.output(x)
        x = x.permute(0, 2, 1)
        return x

class T1(nn.Module):
    def __init__(self, dropout):
        super(T1, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)

        self.ll1 = nn.Linear(1024, 512)
        self.ll2 = nn.Linear(512, 256)

        self.t1 = nn.Linear(256, 9)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = F.adaptive_max_pool1d(x, 1)

        x = F.relu(self.bn4(self.ll1(x.view(x.size(0), -1))))
        x = F.relu(self.bn5(self.ll2(x)))
        x = self.t1(x) 
        iden = Variable(torch.from_numpy(np.eye(3).flatten().astype(np.float32)))
        if x.is_cuda:
            iden = iden.to('cuda')
        elif x.is_mps:
            iden = iden.to('mps')
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

class T2(nn.Module):
    def __init__(self, dropout):
        super(T2, self).__init__()
        self.conv1 = nn.Conv1d(64, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)

        self.ll1 = nn.Linear(1024, 512)
        self.ll2 = nn.Linear(512, 256)

        self.t2 = nn.Linear(256, 4096)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = F.adaptive_max_pool1d(x, 1)

        x = F.relu(self.bn4(self.ll1(x.view(x.size(0), -1))))
        x = F.relu(self.bn5(self.ll2(x)))
        x = self.t2(x) 
        iden = Variable(torch.from_numpy(np.eye(64).flatten().astype(np.float32)))
        if x.is_cuda:
            iden = iden.to('cuda')
        elif x.is_mps:
            iden = iden.to('mps')
        x = x + iden
        x = x.view(-1, 64, 64)
        return x

class Pointnet2Cls(nn.Module):
    def __init__(self, n_embd, dropout):
        super(Pointnet2Cls, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)

        self.ll1 = nn.Linear(1024, 512)
        self.ll2 = nn.Linear(512, 256)
        self.output = nn.Linear(256, 40) # This 40 is for modelnet40

        self.t1 = T1(dropout)
        self.t2 = T2(dropout)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
    
    def forward(self, x):
        xyz = x
        
        t1 = self.t1(x)
        x =  torch.bmm(t1, x)

        x = sample_and_group(npoint=1024, nsample=32, xyz = xyz, points=x)

        x = F.relu(self.bn1(self.conv1(x)))
       
        t2 = self.t2(x)
        x = torch.bmm(t2, x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.adaptive_max_pool1d(x, 1).view(x.size(0), -1)
        
        x = F.relu(self.bn4(self.ll1(x)))
        x = F.relu(self.bn5(self.ll2(x)))
        x = self.output(x)
        return x
    
class Pointnet2Seg(nn.Module):
    def __init__(self, in_channels = 3, labels=8):
        super(Pointnet2Seg, self).__init__()
        self.emb1 = nn.Conv1d(in_channels, 64, kernel_size=1)
        self.emb2 = nn.Conv1d(64, 64, kernel_size=1)
        self.emb1_bn = nn.BatchNorm1d(64)
        self.emb2_bn = nn.BatchNorm1d(64)

        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        self.featureProp_0 = PointNetFeaturePropagation(in_channel=256+128, mlp=[256, 512]) # With skip connections
        self.featureProp_1 = PointNetFeaturePropagation(in_channel=512+64, mlp=[512, 1024]) # With skip connections


        self.conv1 = nn.Conv1d(1024, 512, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        
        self.conv2 = nn.Conv1d(512, 128, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(128)

        self.output = nn.Conv1d(128, labels, kernel_size=1)
 
    
    def forward(self, x):
        xyz = x
        x = x.permute(0,2,1)

        x = F.relu(self.emb1_bn(self.emb1(x)))
        x = F.relu(self.emb2_bn(self.emb2(x)))
        features0 = x
        xyz1, features1 = sample_and_group(npoint=2048, nsample=32, xyz = xyz, points=x.permute(0, 2, 1))
        features1 = self.gather_local_0(features1)

        xyz2, features2 = sample_and_group(npoint=1024, nsample=32, xyz = xyz1, points=features1.permute(0, 2, 1))
        x = self.gather_local_1(features2)

        x = self.featureProp_0(xyz1 = xyz1.transpose(1,2), xyz2 = xyz2.transpose(1,2), points1=features1, points2=x)
        x = self.featureProp_1(xyz1 = xyz.transpose(1,2), xyz2 = xyz1.transpose(1,2), points1=features0, points2=x)
       
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        logits = self.output(x)
        return logits.permute(0,2,1)
    
if __name__ == "__main__":
    
    model = Pointnet2Seg()
    x = torch.rand(2,2048,3)

    x = model(x)

    print(x.size())