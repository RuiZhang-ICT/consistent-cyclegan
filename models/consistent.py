import torch
import torch.nn as nn

from collections import namedtuple
from torchvision import models

def DiffFuncAveL1(fea1, fea2):
    # fea1, fea2: [b, num_fea, num_pair, fea_dim]
    tmp_diff = torch.abs(fea1 - fea2)
    tmp_norm = torch.mean(torch.abs(fea1)) + torch.mean(torch.abs(fea2))
    tmp_diff = torch.mean(tmp_diff, dim=-1) / tmp_norm
    return tmp_diff # [b, num_fea, num_pair]

def DiffFuncCosine(fea1, fea2):
    # fea1, fea2: [b, num_fea, num_pair, fea_dim]
    tmp_diff = nn.functional.cosine_similarity(fea1, fea2, dim=-1)
    return 1- tmp_diff # [b, num_fea, num_pair]

class ConsistentLoss(nn.Module):
    def __init__(self, sample_rate, threshold, diff_func='avel1'):
        super(ConsistentLoss, self).__init__()
        self.sample_rate = sample_rate
        self.threshold = threshold
        if diff_func == 'avel1':
            self.diff_func = DiffFuncAveL1
        elif diff_func == 'cosine':
            self.diff_func = DiffFuncCosine
        else:
            self.diff_func = None
            print "undefined similarity function"
    def gather_diff(self, fea, idx):
        # fea: [b, num_fea, fea_dim]
        # idx: [b, num_fea, num_pair]
        tmp_fea = fea.unsqueeze(2) # [b, num_fea, 1, fea_dim]
        tmp_fea_gather = tmp_fea.repeat(1, 1, fea.shape[1], 1) # [b, num_fea, num_fea, fea_dim]
        tmp_fea_gather = tmp_fea_gather.permute(0,2,1,3) # [b, num_fea, num_fea, fea_dim]
        tmp_idx = idx.unsqueeze(-1) # [b, num_fea, num_pair, 1]
        tmp_idx = tmp_idx.repeat(1, 1, 1, fea.shape[2]) # [b, num_fea, num_pair, fea_dim]
        tmp_gather = tmp_fea_gather.gather(2, tmp_idx) # [b, num_fea, num_pair, fea_dim]
        tmp_fea_repeat = tmp_fea.repeat(1, 1, idx.shape[2], 1) # [b, num_fea, num_pair, fea_dim]
        tmp_diff = self.diff_func(tmp_gather, tmp_fea_repeat)
        return tmp_diff # [b, num_fea, num_pair]
    #def gather_similarity_loop(self, fea, idx):
        # fea: [b, num_fea, fea_dim]
        # idx: [b, num_fea, num_pair]
    #    b, num_fea, fea_dim = fea.shape
    #    num_pair = idx.shape[-1]
    #    tmp_diff = torch.zeros((b, num_fea, num_pair), device=torch.device(fea.device))
    #    for bidx in range(b):
    #        print bidx
    #        for fidx in range(num_fea):
    #            tmp_fea = fea[bidx][fidx] # [fea_dim]
    #            tmp_fea = tmp_fea.unsqueeze(0) # [1, fea_dim]
    #            tmp_fea_repeat = tmp_fea.repeat(num_pair, 1) # [num_pair, fea_dim]
    #            tmp_idx = idx[bidx][fidx] # [num_pair]
    #            tmp_idx = tmp_idx.unsqueeze(-1) # [num_pair, 1]
    #            tmp_idx = tmp_idx.repeat(1, fea_dim) # [num_pair, fea_dim]
    #            tmp_gather = fea[bidx].gather(0, tmp_idx) # [num_pair, fea_dim]
    #            tmp_diff[bidx][fidx][...] = self.similarity_func(tmp_gather, tmp_fea_repeat)
    #    return tmp_diff
    def forward(self, fea_in, fea_out):
        in_batch, in_channel, in_height, in_width = fea_in.shape
        num_fea  = in_height * in_width
        num_pair = int(num_fea * self.sample_rate)
        fea_mat_in  = fea_in.reshape((in_batch, in_channel, num_fea))
        fea_mat_out = fea_out.reshape((in_batch, in_channel, num_fea))
        fea_mat_in  = fea_mat_in.permute([0, 2, 1])  # [b, num_fea, fea_dim]
        fea_mat_out = fea_mat_out.permute([0, 2, 1]) # [b, num_fea, fea_dim]
        pair_idx = torch.randint(num_fea, (in_batch, num_fea, num_pair), device=torch.device(fea_in.device)).long() # [b, num_fea, num_pair]
        diff_in  = self.gather_diff(fea_mat_in, pair_idx)  # [b, num_fea, num_pair]
        diff_out = self.gather_diff(fea_mat_out, pair_idx) # [b, num_fea, num_pair]
        diff_map = diff_in <= self.threshold
        #print torch.sum(diff_map), diff_map.shape
        diff_res = diff_out[diff_map]
        return torch.mean(diff_res)


class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, gpu_ids=None):
        super(Vgg16, self).__init__()
        self.gpu_ids = gpu_ids
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
    def forward(self, X):
        #h = self.slice1(X)
        h = nn.parallel.data_parallel(self.slice1, X, self.gpu_ids)
        h_relu1_2 = h
        #h = self.slice2(h)
        h = nn.parallel.data_parallel(self.slice2, h, self.gpu_ids)
        h_relu2_2 = h
        #h = self.slice3(h)
        h = nn.parallel.data_parallel(self.slice3, h, self.gpu_ids)
        h_relu3_3 = h
        #h = self.slice4(h)
        h = nn.parallel.data_parallel(self.slice4, h, self.gpu_ids)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out
