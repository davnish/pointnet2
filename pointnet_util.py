import torch
import numpy as np
import torch.nn as nn

class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU()

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 128]) 
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        batch_size, _, N = x.size()
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x))) # B, D, N
        x = torch.max(x, 2)[0] # Getting the global just like in pointnet
        x = x.view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x
    
    
def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def sample_and_group(npoint, nsample, xyz, points, radius=2):
    """
    inputs:
    npoint: no of points to sample, int
    nsample: no of points to consider while considering the distance
    xyz: points coords, [B, N, C(cords)]
    points: Embeddings [B, N, C(embdding shape)]

    output:
    new_xyz: [B, npoint, C(coords shape)]
    new_points: [B, npoint, nsample, 2C]
    """
    B, N, C = xyz.shape
    S = npoint 
    
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint]

    new_xyz = index_points(xyz, fps_idx) # Gives the coords of fps_idx [B, S, C] where is the sampled points S=npoint
    new_points = index_points(points, fps_idx) # B, npoint, C(embedding shape)

    ###### This can be replaced by query ball
    dists = square_distance(new_xyz, xyz)  # B x npoint x N
    idx = dists.argsort()[:, :, :nsample]  # B x npoint x K
    ######

    # idx = query_ball_point(radius=radius, nsample=nsample, xyz=xyz, new_xyz=new_xyz)

    grouped_points = index_points(points, idx) # B, npoint, K, C(embedding shape)
    grouped_points_norm = grouped_points - new_points.view(B, S, 1, -1)
    # print(grouped_points.size())
    new_points = torch.cat([grouped_points_norm, new_points.view(B, S, 1, -1).repeat(1, 1, nsample, 1)], dim=-1)
    return new_xyz, new_points

def sample_and_group_all(nsample, xyz, points):
    B, N, C = xyz.shape
    S = N

    dists = square_distance(xyz, xyz)  # B x npoint x N
    idx = dists.argsort()[:, :, :nsample]  # B x npoint x K

    grouped_points = index_points(points, idx)
    grouped_points_norm = grouped_points - points.view(B, S, 1, -1)

    new_points = torch.cat([grouped_points_norm, points.view(B, S, 1, -1).repeat(1, 1, nsample, 1)], dim=-1)
    return xyz, new_points

def fps_numpy(points, num_points):
    points = np.asarray(points)
    sampled_pnts = np.zeros(num_points, dtype='int') # Sampled points
    pnts_left = np.arange(len(points), dtype='int') # points not sampled 
    dist = np.ones_like(pnts_left) * float('inf') # dist array

    selected = np.random.randint(0, len(points)) # Selected current point
    sampled_pnts[0] = selected
    pnts_left = np.delete(pnts_left, selected)
    # dist = np.linalg.norm(points[pnts_left] - points[selected], ord = 2)


    for i in range(1, num_points):
        

        selected_dist = np.linalg.norm(points[pnts_left] - points[selected], ord = 2)

        # temp = np.linalg.norm(points[pnts_left] - points[selected], ord = 2)

        dist[pnts_left] = np.minimum(dist[pnts_left], selected_dist)
        # print(dist)
        selected = np.argmax(dist[pnts_left], axis = 0)
        # print(selected)
        sampled_pnts[i] = pnts_left[selected]

        pnts_left = np.delete(pnts_left, selected)

    return sampled_pnts

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp, drp_add = True):
        super(PointNetFeaturePropagation, self).__init__()

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.mlp_drp = nn.ModuleList()
        self.drp_add = drp_add
        last_channel = in_channel

        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))

            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            if self.drp_add:
                self.mlp_drp.append(nn.Dropout(p=0.5))
            last_channel = out_channel
        self.relu = nn.ReLU()
        

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Inputs:
            xyz1: input points position data, [B, C, N], these are the points from the previous layers
            xyz2: sampled input points position data, [B, C, S], these are the points from the current sample layer
            points1: input points data, [B, D, N]
            points2: input points data, [B, D', S]
        Return:
            new_points: upsampled points data, [B, D' + D, N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2) # cal distance of every xyz1 array with every point of xyz2
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2) # takes every point from points2[idx]
        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = self.relu(bn(conv(new_points)))
            if self.drp_add:
                new_points = self.mlp_drp[i](new_points)
                
        return new_points


if __name__ == "__main__":
    a = torch.rand(2, 400, 10) # B, C, N
    axyz = torch.rand(2, 3, 10)# B, C, N
    b = torch.rand(2,200, 5) # B, C', N
    bxyz = torch.rand(2, 3, 5)

    fp = PointNetFeaturePropagation(in_channel=600, mlp=[300]) # in_channel = C+C'
    xyz, points = fp(axyz, bxyz, a, b)
    print(xyz.shape, points.shape)

    grouped_idx, grouped_points = sample_and_group(npoint=5, nsample=3, xyz=axyz.permute(0,2,1), points=a.permute(0,2,1))
    print(grouped_idx.size(), grouped_points.size())