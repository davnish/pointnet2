import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
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
    def __init__(self, n_embd=64, dropout=0.5):
        super(PointnetSeg, self).__init__()
        self.dp = dropout
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
        x = torch.bmm(t1, x)
        
        x = F.dropout(x, p=self.dp) 
        
        x = F.relu(self.bn1(self.conv1(x)))

        x = F.dropout(x, p=self.dp) 
       
        t2 = self.t2(x)
        feature_transformed = torch.bmm(t2, x)

        x = F.dropout(feature_transformed, p=self.dp)
        
        x = F.relu(self.bn2(self.conv2(feature_transformed)))

        x = F.dropout(x, p=self.dp) 

        x = F.relu(self.bn3(self.conv3(x)))

        x = F.dropout(x, p=self.dp) 

        x = F.adaptive_max_pool1d(x, 1).view(x.size(0), -1)
        x = torch.cat((feature_transformed, x.unsqueeze(-1).repeat(1, 1, n_points)), dim = 1)

        x = F.relu(self.bn4(self.conv4(x)))
        x = F.dropout(x, p=self.dp) 

        x = F.relu(self.bn5(self.conv5(x)))

        x = F.dropout(x, p=self.dp) 

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
    
if __name__ == "__main__":
    
    model = PointnetSeg()
    x = torch.rand(2,2048,3)
    x = model(x)

    print(x.size())