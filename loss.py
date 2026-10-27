import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
import math
device = torch.device('cuda:0')

class AutomaticWeightedLoss(torch.nn.Module):
    """automatically weighted multi-task loss
    Params:
        num: int,the number of loss
        x: multi-task loss
    Examples:
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for (i, loss) in enumerate(x):
            loss_sum += 0.5 / self.params[i] ** 2 * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum

class ContrastiveWeight(torch.nn.Module):

    def __init__(self, device, positive_nums, temperature):
        super(ContrastiveWeight, self).__init__()
        self.temperature = temperature
        self.device = device
        self.bce = torch.nn.BCELoss()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)
        self.kl = torch.nn.KLDivLoss(reduction='batchmean')
        self.positive_nums = positive_nums

    def get_positive_and_negative_mask(self, similarity_matrix, cur_batch_size):
        diag = np.eye(cur_batch_size)
        mask = torch.from_numpy(diag)
        mask = mask.type(torch.bool)
        oral_batch_size = cur_batch_size // (self.positive_nums + 1)
        positives_mask = np.zeros(similarity_matrix.size())
        for i in range(self.positive_nums + 1):
            ll = np.eye(cur_batch_size, cur_batch_size, k=oral_batch_size * i)
            lr = np.eye(cur_batch_size, cur_batch_size, k=-oral_batch_size * i)
            positives_mask += ll
            positives_mask += lr
        positives_mask = torch.from_numpy(positives_mask).to(similarity_matrix.device)
        positives_mask[mask] = 0
        negatives_mask = 1 - positives_mask
        negatives_mask[mask] = 0
        return (positives_mask.type(torch.bool), negatives_mask.type(torch.bool))

    def forward(self, batch_emb_om):
        cur_batch_shape = batch_emb_om.shape
        norm_emb = F.normalize(batch_emb_om, dim=1)
        similarity_matrix = torch.matmul(norm_emb, norm_emb.transpose(0, 1))
        (positives_mask, negatives_mask) = self.get_positive_and_negative_mask(similarity_matrix, cur_batch_shape[0])
        positives = similarity_matrix[positives_mask].view(cur_batch_shape[0], -1)
        negatives = similarity_matrix[negatives_mask].view(cur_batch_shape[0], -1)
        logits = torch.cat((positives, negatives), dim=-1)
        y_true = torch.cat((torch.ones(cur_batch_shape[0], positives.shape[-1]) / positives.shape[-1], torch.zeros(cur_batch_shape[0], negatives.shape[-1])), dim=-1).to(self.device).float()
        predict = self.log_softmax(logits / self.temperature)
        loss = self.kl(predict, y_true)
        return (loss, similarity_matrix, logits, positives_mask)

class CenterLoss(torch.nn.Module):

    def __init__(self, num_classes, feat_dim):
        super().__init__()
        self.centers = torch.nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, features, labels):
        batch_centers = self.centers[labels]
        loss = ((features - batch_centers) ** 2).sum(dim=1).mean()
        return loss

class CrossDomainContrastiveWeight(nn.Module):
    """
    Multi-positive Cross-Domain Contrastive Loss (InfoNCE)

    Input:
        batch_emb_om: [num_domains * batch_size, feature_dim]
        embeddings are stacked by domain blocks
    """

    def __init__(self, feature_dim, proj_dim=256, temperature=0.1, device='cuda'):
        super().__init__()
        self.temperature = temperature
        self.device = device
        self.projector = nn.Sequential(nn.Linear(feature_dim, proj_dim), nn.ReLU(), nn.Linear(proj_dim, feature_dim)).to(device)

    @staticmethod
    def build_positive_mask(batch_size, num_domains, device):
        """
        Positive pairs: same sample index, different domains
        """
        total = batch_size * num_domains
        idx = torch.arange(total, device=device)
        sample_id = idx % batch_size
        pos_mask = sample_id.unsqueeze(0) == sample_id.unsqueeze(1)
        pos_mask.fill_diagonal_(False)
        return pos_mask

    def forward(self, batch_emb_om, batch_size=64, num_domains=8):
        """
        batch_emb_om: [num_domains * batch_size, feature_dim]
        """
        assert batch_emb_om.size(0) == batch_size * num_domains
        z = F.normalize(batch_emb_om, dim=1)
        z = self.projector(z)
        z = F.normalize(z, dim=1)
        sim = torch.matmul(z, z.T) / self.temperature
        sim = sim - sim.max(dim=1, keepdim=True)[0]
        pos_mask = self.build_positive_mask(batch_size, num_domains, z.device)
        self_mask = torch.eye(batch_size * num_domains, dtype=torch.bool, device=z.device)
        exp_sim = torch.exp(sim) * ~self_mask
        pos_exp = exp_sim * pos_mask
        loss = -torch.log(pos_exp.sum(dim=1) / exp_sim.sum(dim=1))
        return loss.mean()

def supervised_contrastive_loss(features, labels, temperature=0.1):
    """
    features: [B, D]
    labels:   [B]
    """
    features = F.normalize(features, p=2, dim=1)
    batch_size = features.size(0)
    similarity = torch.matmul(features, features.T) / temperature
    (logits_max, _) = similarity.max(dim=1, keepdim=True)
    similarity = similarity - logits_max.detach()
    labels = labels.view(-1, 1)
    positive_mask = torch.eq(labels, labels.T).float().to(features.device)
    self_mask = torch.eye(batch_size, device=features.device)
    positive_mask = positive_mask * (1.0 - self_mask)
    exp_logits = torch.exp(similarity) * (1.0 - self_mask)
    log_prob = similarity - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
    positive_count = positive_mask.sum(dim=1)
    valid_mask = positive_count > 0
    mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / (positive_count + 1e-12)
    if valid_mask.any():
        loss = -mean_log_prob_pos[valid_mask].mean()
    else:
        loss = torch.tensor(0.0, device=features.device)
    return loss

class AggregationRebuild(torch.nn.Module):

    def __init__(self, device, temperature=0.3):
        super(AggregationRebuild, self).__init__()
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mse = torch.nn.MSELoss()

    def forward(self, similarity_matrix, batch_emb_om):
        cur_batch_shape = batch_emb_om.shape
        similarity_matrix /= self.temperature
        similarity_matrix = similarity_matrix - torch.eye(cur_batch_shape[0]).to(similarity_matrix.device).float() * 1000000000000.0
        rebuild_weight_matrix = self.softmax(similarity_matrix)
        batch_emb_om = batch_emb_om.reshape(cur_batch_shape[0], -1)
        rebuild_batch_emb = torch.matmul(rebuild_weight_matrix, batch_emb_om)
        rebuild_oral_batch_emb = rebuild_batch_emb.reshape(cur_batch_shape[0], cur_batch_shape[1], -1)
        return (rebuild_weight_matrix, rebuild_oral_batch_emb)

class NCEAverage(nn.Module):

    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.2, use_softmax=False):
        super(NCEAverage, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        self.K = K
        self.use_softmax = use_softmax
        self.register_buffer('params', torch.tensor([K, T, -1, -1, momentum]))
        stdv = 1.0 / math.sqrt(inputSize / 3)
        self.register_buffer('memory_l', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_ab', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, l, ab, y, idx=None):
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z_l = self.params[2].item()
        Z_ab = self.params[3].item()
        momentum = self.params[4].item()
        batchSize = l.size(0)
        outputSize = self.memory_l.size(0)
        inputSize = self.memory_l.size(1)
        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
            idx.select(1, 0).copy_(y.data)
        weight_l = torch.index_select(self.memory_l, 0, idx.view(-1)).detach()
        weight_l = weight_l.view(batchSize, K + 1, inputSize)
        out_ab = torch.bmm(weight_l, ab.view(batchSize, inputSize, 1))
        weight_ab = torch.index_select(self.memory_ab, 0, idx.view(-1)).detach()
        weight_ab = weight_ab.view(batchSize, K + 1, inputSize)
        out_l = torch.bmm(weight_ab, l.view(batchSize, inputSize, 1))
        if self.use_softmax:
            out_ab = torch.div(out_ab, T)
            out_l = torch.div(out_l, T)
            out_l = out_l.contiguous()
            out_ab = out_ab.contiguous()
        else:
            out_ab = torch.exp(torch.div(out_ab, T))
            out_l = torch.exp(torch.div(out_l, T))
            if Z_l < 0:
                self.params[2] = out_l.mean() * outputSize
                Z_l = self.params[2].clone().detach().item()
                print('normalization constant Z_l is set to {:.1f}'.format(Z_l))
            if Z_ab < 0:
                self.params[3] = out_ab.mean() * outputSize
                Z_ab = self.params[3].clone().detach().item()
                print('normalization constant Z_ab is set to {:.1f}'.format(Z_ab))
            out_l = torch.div(out_l, Z_l).contiguous()
            out_ab = torch.div(out_ab, Z_ab).contiguous()
        with torch.no_grad():
            l_pos = torch.index_select(self.memory_l, 0, y.view(-1))
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(l, 1 - momentum))
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_l = l_pos.div(l_norm)
            self.memory_l.index_copy_(0, y, updated_l)
            ab_pos = torch.index_select(self.memory_ab, 0, y.view(-1))
            ab_pos.mul_(momentum)
            ab_pos.add_(torch.mul(ab, 1 - momentum))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_ab = ab_pos.div(ab_norm)
            self.memory_ab.index_copy_(0, y, updated_ab)
        return (out_l, out_ab)

class AliasMethod(object):
    """
    From: https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    """

    def __init__(self, probs):
        if probs.sum() > 1:
            probs.div_(probs.sum())
        K = len(probs)
        self.prob = torch.zeros(K)
        self.alias = torch.LongTensor([0] * K)
        smaller = []
        larger = []
        for (kk, prob) in enumerate(probs):
            self.prob[kk] = K * prob
            if self.prob[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()
            self.alias[small] = large
            self.prob[large] = self.prob[large] - 1.0 + self.prob[small]
            if self.prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)
        for last_one in smaller + larger:
            self.prob[last_one] = 1

    def cuda(self):
        self.prob = self.prob.cuda()
        self.alias = self.alias.cuda()

    def draw(self, N):
        """
        Draw N samples from multinomial
        :param N: number of samples
        :return: samples
        """
        K = self.alias.size(0)
        kk = torch.zeros(N, dtype=torch.long, device=self.prob.device).random_(0, K)
        prob = self.prob.index_select(0, kk)
        alias = self.alias.index_select(0, kk)
        b = torch.bernoulli(prob)
        oq = kk.mul(b.long())
        oj = alias.mul((1 - b).long())
        return oq + oj
