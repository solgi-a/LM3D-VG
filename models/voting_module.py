
import torch
import torch.nn as nn
import torch.nn.functional as F

class VotingModule(nn.Module):
    def __init__(self, vote_factor, seed_feature_dim):
        super().__init__()
        self.vote_factor = vote_factor
        self.in_dim = seed_feature_dim
        self.out_dim = self.in_dim
        self.conv1 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv2 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv3 = torch.nn.Conv1d(self.in_dim, (3+self.out_dim) * self.vote_factor, 1)
        self.bn1 = torch.nn.BatchNorm1d(self.in_dim)
        self.bn2 = torch.nn.BatchNorm1d(self.in_dim)
        
    def forward(self, seed_xyz, seed_features):
        batch_size = seed_xyz.shape[0]
        num_seed = seed_xyz.shape[1]
        num_vote = num_seed*self.vote_factor
        net = F.relu(self.bn1(self.conv1(seed_features))) 
        net = F.relu(self.bn2(self.conv2(net))) 
        net = self.conv3(net)
                
        net = net.transpose(2,1).view(batch_size, num_seed, self.vote_factor, 3+self.out_dim).contiguous()
        offset = net[:,:,:,0:3]
        vote_xyz = seed_xyz.unsqueeze(2) + offset
        vote_xyz = vote_xyz.contiguous().view(batch_size, num_vote, 3)
        
        residual_features = net[:,:,:,3:]
        vote_features = seed_features.transpose(2,1).unsqueeze(2).contiguous() + residual_features
        vote_features = vote_features.contiguous().view(batch_size, num_vote, self.out_dim)
        vote_features = vote_features.transpose(2,1).contiguous()
        
        return vote_xyz, vote_features
 
