from project_paths import configured_path
from self import data_transform_masked4cl, LSTMEncoder
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision
import math
import time
from tqdm import tqdm
from trans import Data_Transformer
import torch.optim as optim
from attention import FullAttention, ProbAttention, AttentionLayer, CrossAttention
from loss import AutomaticWeightedLoss, ContrastiveWeight, AggregationRebuild, CrossDomainContrastiveWeight, CenterLoss, supervised_contrastive_loss
from Transformer import EncoderLayer, DecoderLayer, Encoder, ConvLayer, Decoder
from load import load_data
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, f1_score, recall_score
import os
import matplotlib.pyplot as plt
import logging
import sys
from Embed import PositionalEmbedding, TokenEmbedding, FixedEmbedding, TemporalEmbedding, TimeFeatureEmbedding, DataEmbedding_wo_pos
from sklearn.metrics import confusion_matrix
import seaborn as sns
import warnings
from Data_Augmentation import Data_Augmentation, flip_row
from TCN import TemporalConvNet
warnings.filterwarnings('ignore')
import copy
from paint import plot_confusion_matrix_18db_loader
device = torch.device('cuda:0')
if __name__ == '__main__':
    ((mods, snrs, lbl), (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), (train_idx, val_idx, test_idx), Y_train_str) = load_data()
proatten = ProbAttention()
fullatten = FullAttention()
selfatten = AttentionLayer(proatten, 128, 8)
crossatten = CrossAttention(128, 8)
decoder = DecoderLayer(selfatten, crossatten, 128, 128)
encoder = EncoderLayer(selfatten, 128)

class projector(nn.Module):

    def __init__(self, d_model, proj_dim):
        super(projector, self).__init__()
        self.d_model = d_model
        self.proj_dim = proj_dim
        self.net = nn.Sequential(nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, 128))

    def forward(self, x):
        out = self.net(x)
        return out
model_s = projector(1024, 128)
model_s1 = projector(512, 128)
model_r = Decoder(layers=[decoder, decoder])

class pre_model(nn.Module):

    def __init__(self, encoder, encoder_masked, decoder, projector_h, projector_s):
        super(pre_model, self).__init__()
        self.encoder = encoder
        self.encoder_masked = encoder_masked
        self.decoder = decoder
        self.projector = projector_h
        self.projector_s = projector_s
        self.cross = ContrastiveWeight(device=device, positive_nums=4, temperature=0.1)
        self.cw = ContrastiveWeight(device=device, positive_nums=12, temperature=0.1)
        self.agg = AggregationRebuild(device=device)
        self.awl = AutomaticWeightedLoss(2)
        self.mse = torch.nn.MSELoss()
        self.lin = nn.Linear(512, 128)

    def forward(self, x_mask, x_inter):
        Z_orig = self.encoder(x_inter)
        Z_inter = self.encoder_masked(x_mask)
        avg_feat = F.adaptive_avg_pool1d(Z_orig, 1).squeeze(-1)
        max_feat = F.adaptive_max_pool1d(Z_orig, 1).squeeze(-1)
        h = torch.cat([avg_feat, max_feat], dim=1)
        avg_feat_m = F.adaptive_avg_pool1d(Z_inter, 1).squeeze(-1)
        max_feat_m = F.adaptive_max_pool1d(Z_inter, 1).squeeze(-1)
        s = torch.cat([avg_feat_m, max_feat_m], dim=1)
        S_orig = self.projector(h)
        S_inter = self.projector_s(s)
        (loss_cl1, similarity_matrix, logits, positives_mask) = self.cw(S_orig)
        (loss_cl2, similarity_matrix1, logits1, positives_mask1) = self.cross(S_inter)
        loss = self.awl(loss_cl1, loss_cl2)
        return (loss, loss_cl1, loss_cl2)

class Conv(nn.Module):

    def __init__(self):
        super(Conv, self).__init__()
        self.preprocess = nn.Sequential(nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1), nn.LayerNorm(64), nn.LeakyReLU(), nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1), nn.LayerNorm(64), nn.LeakyReLU(), nn.Conv1d(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0), nn.LayerNorm(64), nn.LeakyReLU(), nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1), nn.LayerNorm(32), nn.LeakyReLU(), nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1), nn.LayerNorm(32), nn.LeakyReLU(), nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0), nn.LayerNorm(32), nn.LeakyReLU())
        self.res1 = nn.Sequential(nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1), nn.LayerNorm(32), nn.LeakyReLU(), nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0), nn.LayerNorm(32), nn.LeakyReLU())
        self.ce1 = nn.Sequential(nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1), nn.LayerNorm(32))
        self.Max1 = nn.Sequential(nn.LayerNorm(32), nn.MaxPool1d(2, 2))
        self.res2 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1), nn.LayerNorm(16), nn.LeakyReLU(), nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0), nn.LayerNorm(16), nn.LeakyReLU())
        self.ce2 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1), nn.LayerNorm(16))
        self.Max2 = nn.Sequential(nn.LayerNorm(16), nn.MaxPool1d(2, 2))
        self.res3 = nn.Sequential(nn.Conv1d(in_channels=96, out_channels=128, kernel_size=3, stride=1, padding=1), nn.LayerNorm(128), nn.LeakyReLU(), nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), nn.LayerNorm(128), nn.LeakyReLU())
        self.ce3 = nn.Sequential(nn.Conv1d(in_channels=96, out_channels=128, kernel_size=1, stride=1, padding=0), nn.LayerNorm(128))
        self.Max3 = nn.Sequential(nn.LayerNorm(32), nn.MaxPool1d(2, 2))
        self.res4 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1), nn.LayerNorm(128), nn.LeakyReLU(), nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1), nn.LayerNorm(128), nn.LeakyReLU())
        self.ce4 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1, stride=1, padding=0), nn.LayerNorm(128))
        self.Max4 = nn.Sequential(nn.LayerNorm(16), nn.MaxPool1d(2, 2))
        self.conv = nn.Sequential(nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1), nn.LayerNorm(128), nn.LeakyReLU())
        self.drop = nn.Dropout1d(0.2)
        self.CNN = nn.Sequential(nn.Conv1d(in_channels=10, out_channels=16, kernel_size=3, stride=2, padding=1), nn.GroupNorm(8, 16), nn.LeakyReLU(), nn.Conv1d(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0), nn.GroupNorm(8, 16), nn.LeakyReLU(), nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1), nn.GroupNorm(8, 32), nn.LeakyReLU(), nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0), nn.GroupNorm(8, 32), nn.LeakyReLU(), nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.LeakyReLU(), nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0), nn.GroupNorm(8, 64), nn.LeakyReLU(), nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1), nn.GroupNorm(8, 128), nn.LeakyReLU(), nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0), nn.GroupNorm(8, 128), nn.LeakyReLU())
        self.LSTM = nn.LSTM(input_size=128, hidden_size=256, batch_first=True)

    def forward(self, x):
        out_fin = self.CNN(x)
        return out_fin

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.CNN = nn.Sequential(nn.Conv1d(in_channels=2, out_channels=16, kernel_size=3, stride=2, padding=1), nn.GroupNorm(8, 16), nn.LeakyReLU(), nn.Conv1d(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0), nn.GroupNorm(8, 16), nn.LeakyReLU(), nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1), nn.GroupNorm(8, 32), nn.LeakyReLU(), nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0), nn.GroupNorm(8, 32), nn.LeakyReLU(), nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.LeakyReLU(), nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0), nn.GroupNorm(8, 64), nn.LeakyReLU(), nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1), nn.GroupNorm(8, 128), nn.LeakyReLU(), nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0), nn.GroupNorm(8, 128), nn.LeakyReLU())
        self.LSTM = nn.LSTM(input_size=128, hidden_size=256, batch_first=True)
        self.drop = nn.Dropout1d(0.2)

    def forward(self, x):
        out = self.CNN(x)
        return out

class Encoder1(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = Conv()
        self.res = Encoder1D()

    def forward(self, x):
        Z = self.res(x)
        return Z

class ResBlock1D(nn.Module):

    def __init__(self, in_channels, out_channels, stride, groups=8):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.gn1 = nn.GroupNorm(groups, out_channels)
        self.act = nn.LeakyReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.gn2 = nn.GroupNorm(groups, out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.gn2(out)
        out = out + identity
        out = self.act(out)
        return out

class ResBlock1D_Dilated(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, dilation=2, groups=8):
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.gn1 = nn.GroupNorm(groups, out_channels)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.gn2 = nn.GroupNorm(groups, out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.gn2(out)
        out = out + identity
        out = self.act(out)
        return out

class Encoder1D(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(ResBlock1D(10, 32, stride=2), ResBlock1D(32, 64, stride=2), ResBlock1D(64, 128, stride=2), ResBlock1D(128, 128, stride=1), ResBlock1D_Dilated(128, 128, stride=1, dilation=2))

    def forward(self, x):
        return self.encoder(x)

class EncoderD(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(ResBlock1D(2, 32, stride=2), ResBlock1D(32, 64, stride=2), ResBlock1D(64, 128, stride=2), ResBlock1D(128, 128, stride=1), ResBlock1D_Dilated(128, 128, stride=1, dilation=2))

    def forward(self, x):
        return self.encoder(x)

class Channel_Attention_Module_Conv1D(nn.Module):

    def __init__(self, channels, gamma=2, b=1):
        super(Channel_Attention_Module_Conv1D, self).__init__()
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: [B, C, L]
        """
        avg_x = self.avg_pool(x)
        max_x = self.max_pool(x)
        avg_x = avg_x.transpose(1, 2)
        max_x = max_x.transpose(1, 2)
        avg_out = self.conv(avg_x)
        max_out = self.conv(max_x)
        attn = self.sigmoid(avg_out + max_out)
        attn = attn.transpose(1, 2)
        return x * attn

class CrossAttentionFusionConv(nn.Module):

    def __init__(self, in_dim=128, out_dim=256, reduction=16):
        super().__init__()
        hidden_dim = max(out_dim // reduction, 1)
        self.proj1 = nn.Conv1d(in_dim, out_dim, kernel_size=1)
        self.proj2 = nn.Conv1d(in_dim, out_dim, kernel_size=1)
        self.fc1_1 = nn.Linear(out_dim, hidden_dim)
        self.fc1_2 = nn.Linear(hidden_dim, out_dim)
        self.fc2_1 = nn.Linear(out_dim, hidden_dim)
        self.fc2_2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x1, x2):
        f1 = self.proj1(x1)
        f2 = self.proj2(x2)
        g1 = f1.mean(dim=2)
        g2 = f2.mean(dim=2)
        w1 = torch.sigmoid(self.fc1_2(F.relu(self.fc1_1(g1)))).unsqueeze(-1)
        w2 = torch.sigmoid(self.fc2_2(F.relu(self.fc2_1(g2)))).unsqueeze(-1)
        out1 = f1 + f1 * w2
        out2 = f2 + f2 * w1
        out = torch.stack([out1, out2], dim=1)
        return out

class FinetuneEncoder(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder
        self.encoder_masked = model.encoder_masked
        self.att = Channel_Attention_Module_Conv1D(channels=128)
        self.SE = SELayer_1d(128)
        self.Cross = CrossAttentionFusionConv()

    def forward(self, x_inter, x_mask):
        Z1 = self.encoder(x_inter)
        Z2 = self.encoder_masked(x_mask)
        Z = torch.cat((Z1, Z2), dim=2)
        return Z

def model_pretrain(model, model_optimizer, model_scheduler, train_loader, train_mask, device):
    total_loss = []
    total_cl_loss = []
    total_cl_loss2 = []
    model.train()
    with tqdm(total=len(train_loader)) as pb:
        for (batch_idx, ((data, labels), (data_mask, labels_mask))) in enumerate(zip(train_loader, train_mask)):
            model_optimizer.zero_grad()
            DA = []
            for idex in [4, 2, 3, 8, 9, 10, 5, 6, 7]:
                Data_DA = Data_Augmentation(data, idex, theta_list=None, std_list=None)
                DA.append(Data_DA)
            DA_stack = torch.cat(DA, dim=0)
            (data_masked_m, mask) = data_transform_masked4cl(data, 0.5, 5)
            data_masked_om = torch.cat([data, DA_stack, data_masked_m], 0)
            DT = []
            for idex in [1, 2, 3, 4]:
                DA_trans = Data_Transformer(data_mask, idex)
                DT.append(DA_trans)
            DT_stack = torch.cat(DT, dim=0)
            data_all = torch.cat([data_mask, DT_stack], 0)
            data_all = data_all.to(device)
            (data, labels, data_masked_om, data_masked_m, data_mask, labels_mask) = (data.float().to(device), labels.float().to(device), data_masked_om.float().to(device), data_masked_m.to(device), data_mask.to(device), labels_mask.to(device))
            (loss, loss_cl1, loss_cl2) = model(data_all, data_masked_om)
            loss.backward()
            model_optimizer.step()
            total_loss.append(loss.item())
            total_cl_loss.append(loss_cl1.item())
            total_cl_loss2.append(loss_cl2.item())
            pb.update(1)
    total_loss = torch.tensor(total_loss).mean()
    total_cl_loss = torch.tensor(total_cl_loss).mean()
    total_cl_loss2 = torch.tensor(total_cl_loss2).mean()
    model_scheduler.step()
    return (total_loss, total_cl_loss, total_cl_loss2)

class Data(Dataset):
    """Dataset wrapper for paired signal representations."""

    def __init__(self, img, tagert):
        self.img = img
        self.tagert = tagert

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        img = self.img[idx]
        tagert = self.tagert[idx]
        return (img, tagert)
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import torch
from tqdm import tqdm

def normalized_accuracy(y_true, y_pred):
    classes = np.unique(y_true)
    acc_list = []
    for c in classes:
        mask = y_true == c
        if mask.sum() == 0:
            continue
        acc_c = np.mean(y_pred[mask] == y_true[mask])
        acc_list.append(acc_c)
    return np.mean(acc_list)

@torch.no_grad()
def open_set_evaluate(model, val_dl, val_dl_mask, device, classifier=None, threshold=0.7, unknown_label=9, LSTM=None, verbose=True):
    model.eval()
    if classifier is not None:
        classifier.eval()
    if LSTM is not None:
        LSTM.eval()
    all_preds = []
    all_labels = []
    all_max_probs = []
    known_correct = 0
    known_total = 0
    unknown_correct = 0
    unknown_total = 0
    with tqdm(total=len(val_dl)) as pb:
        for ((data, labels), (data_mask, labels_mask)) in zip(val_dl, val_dl_mask):
            data = data.float().to(device)
            data_mask = data_mask.float().to(device)
            labels = labels.to(device)
            if labels.ndim > 1:
                labels = labels.argmax(dim=1)
            labels = labels.long().to(device)
            h = model(data_mask, data)
            if LSTM is not None:
                h = LSTM(h)
            logits = classifier(h)
            prob = torch.softmax(logits, dim=1)
            (max_prob, pred_known) = torch.max(prob, dim=1)
            pred_open = pred_known.clone()
            pred_open[max_prob < threshold] = unknown_label
            known_mask = labels != unknown_label
            unknown_mask = labels == unknown_label
            if known_mask.sum() > 0:
                known_correct += (pred_open[known_mask] == labels[known_mask]).sum().item()
                known_total += known_mask.sum().item()
            if unknown_mask.sum() > 0:
                unknown_correct += (pred_open[unknown_mask] == unknown_label).sum().item()
                unknown_total += unknown_mask.sum().item()
            all_preds.append(pred_open.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
            all_max_probs.append(max_prob.detach().cpu().numpy())
            pb.update(1)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_max_probs = np.concatenate(all_max_probs, axis=0)
    open_acc = np.mean(all_preds == all_labels)
    macro_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    NA = normalized_accuracy(all_labels, all_preds)
    known_acc = known_correct / known_total if known_total > 0 else 0.0
    unknown_recall = unknown_correct / unknown_total if unknown_total > 0 else 0.0
    os_score = 2 * known_acc * unknown_recall / (known_acc + unknown_recall + 1e-08)
    if verbose:
        print(f'Threshold: {threshold:.2f}')
        print(f'Known Acc: {known_acc:.4f}')
        print(f'Unknown Recall: {unknown_recall:.4f}')
        print(f'OS Score: {os_score:.4f}')
    return {'open_acc': open_acc, 'precision': macro_precision, 'recall': macro_recall, 'f1': macro_f1, 'NA': NA, 'known_acc': known_acc, 'unknown_recall': unknown_recall, 'os_score': os_score, 'threshold': threshold}

def search_best_threshold(model, val_dl, val_dl_mask, device, classifier, unknown_label=9, thresholds=None, LSTM=None, select_metric='os_score'):
    """Search the validation set for a threshold. select_metric: f1 selects macro F1; os_score selects the balanced score of known accuracy and unknown recall; open_acc selects overall open-set accuracy."""
    if thresholds is None:
        thresholds = np.arange(0.5, 0.96, 0.05)
    best_result = None
    best_score = -1.0
    for th in thresholds:
        result = open_set_evaluate(model=model, val_dl=val_dl, val_dl_mask=val_dl_mask, device=device, classifier=classifier, threshold=float(th), unknown_label=unknown_label, LSTM=LSTM, verbose=False)
        score = result[select_metric]
        if score > best_score:
            best_score = score
            best_result = result
    print(f"[Best Threshold] th={best_result['threshold']:.2f}, openAcc={best_result['open_acc']:.4f}, Precision={best_result['precision']:.4f}, Recall={best_result['recall']:.4f}, F1={best_result['f1']:.4f}, NA={best_result['NA']:.4f}, KnownAcc={best_result['known_acc']:.4f}, UnknownRecall={best_result['unknown_recall']:.4f}, OSScore={best_result['os_score']:.4f}")
    return best_result

@torch.no_grad()
def extract_feature(model, data, data_mask, device, LSTM=None):
    model.eval()
    data = data.float().to(device)
    data_mask = data_mask.float().to(device)
    h = model(data_mask, data)
    if LSTM is not None:
        h = LSTM(h)
    if h.ndim == 3:
        if h.shape[1] != 128:
            h = h.permute(0, 2, 1)
        h = F.adaptive_avg_pool1d(h, 1).squeeze(-1)
    h = F.normalize(h, dim=1)
    return h

@torch.no_grad()
def build_class_prototypes(model, train_dl, train_dl_mask, device, num_known_classes=9, LSTM=None):
    model.eval()
    feat_list = []
    label_list = []
    with tqdm(total=len(train_dl), desc='Build Prototypes') as pb:
        for ((data, labels), (data_mask, labels_mask)) in zip(train_dl, train_dl_mask):
            labels = labels.to(device)
            if labels.ndim > 1:
                labels = labels.argmax(dim=1)
            labels = labels.long().to(device)
            feat = extract_feature(model=model, data=data, data_mask=data_mask, device=device, LSTM=LSTM)
            feat_list.append(feat.detach().cpu())
            label_list.append(labels.detach().cpu())
            pb.update(1)
    feats = torch.cat(feat_list, dim=0)
    labels = torch.cat(label_list, dim=0)
    prototypes = []
    for c in range(num_known_classes):
        class_mask = labels == c
        if class_mask.sum() == 0:
            raise ValueError(f'Class {c} has no samples; its prototype cannot be computed.')
        proto_c = feats[class_mask].mean(dim=0)
        proto_c = F.normalize(proto_c, dim=0)
        prototypes.append(proto_c)
    prototypes = torch.stack(prototypes, dim=0).to(device)
    return prototypes

@torch.no_grad()
def prototype_open_set_evaluate(model, val_dl, val_dl_mask, device, prototypes, threshold=0.8, unknown_label=9, LSTM=None, verbose=True):
    model.eval()
    prototypes = prototypes.to(device)
    all_preds = []
    all_labels = []
    known_correct = 0
    known_total = 0
    unknown_correct = 0
    unknown_total = 0
    with tqdm(total=len(val_dl), desc='Prototype Open-set Eval') as pb:
        for ((data, labels), (data_mask, labels_mask)) in zip(val_dl, val_dl_mask):
            labels = labels.to(device)
            if labels.ndim > 1:
                labels = labels.argmax(dim=1)
            labels = labels.long().to(device)
            feat = extract_feature(model=model, data=data, data_mask=data_mask, device=device, LSTM=LSTM)
            dist = torch.cdist(feat, prototypes, p=2)
            (min_dist, pred_known) = torch.min(dist, dim=1)
            pred_open = pred_known.clone()
            pred_open[min_dist > threshold] = unknown_label
            known_mask = labels != unknown_label
            unknown_mask = labels == unknown_label
            if known_mask.sum() > 0:
                known_correct += (pred_open[known_mask] == labels[known_mask]).sum().item()
                known_total += known_mask.sum().item()
            if unknown_mask.sum() > 0:
                unknown_correct += (pred_open[unknown_mask] == unknown_label).sum().item()
                unknown_total += unknown_mask.sum().item()
            all_preds.append(pred_open.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
            pb.update(1)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    open_acc = np.mean(all_preds == all_labels)
    macro_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    NA = normalized_accuracy(all_labels, all_preds)
    known_acc = known_correct / known_total if known_total > 0 else 0.0
    unknown_recall = unknown_correct / unknown_total if unknown_total > 0 else 0.0
    os_score = 2 * known_acc * unknown_recall / (known_acc + unknown_recall + 1e-08)
    if verbose:
        print(f'Threshold: {threshold:.4f}')
        print(f'Known Acc: {known_acc:.4f}')
        print(f'Unknown Recall: {unknown_recall:.4f}')
        print(f'Open Acc: {open_acc:.4f}')
        print(f'ValF1: {macro_f1:.4f}')
        print(f'NA: {NA:.4f}')
        print(f'OSScore: {os_score:.4f}')
    return {'open_acc': open_acc, 'precision': macro_precision, 'recall': macro_recall, 'f1': macro_f1, 'NA': NA, 'known_acc': known_acc, 'unknown_recall': unknown_recall, 'os_score': os_score, 'threshold': threshold}

def search_best_prototype_threshold(model, val_dl, val_dl_mask, device, prototypes, unknown_label=9, thresholds=None, LSTM=None, select_metric='os_score'):
    if thresholds is None:
        thresholds = np.arange(0.3, 1.81, 0.05)
    best_result = None
    best_score = -1.0
    for th in thresholds:
        result = prototype_open_set_evaluate(model=model, val_dl=val_dl, val_dl_mask=val_dl_mask, device=device, prototypes=prototypes, threshold=float(th), unknown_label=unknown_label, LSTM=LSTM, verbose=False)
        score = result[select_metric]
        if score > best_score:
            best_score = score
            best_result = result
    print(f"[Best Prototype Threshold] th={best_result['threshold']:.4f}, openAcc={best_result['open_acc']:.4f}, Precision={best_result['precision']:.4f}, Recall={best_result['recall']:.4f}, F1={best_result['f1']:.4f}, NA={best_result['NA']:.4f}, KnownAcc={best_result['known_acc']:.4f}, UnknownRecall={best_result['unknown_recall']:.4f}, OSScore={best_result['os_score']:.4f}")
    return best_result

def channel_augment(data, data_mask, noise_std=0.02, scale_std=0.05, max_shift=3):
    """Construct channel-perturbed views. Input shapes: data [B, 2, 128], data_mask [B, 10, 128]. Return data_aug and data_mask_aug."""
    scale = 1.0 + scale_std * torch.randn(data.size(0), 1, 1).to(data.device)
    data_aug = data * scale
    data_mask_aug = data_mask * scale
    data_aug = data_aug + noise_std * torch.randn_like(data_aug)
    data_mask_aug = data_mask_aug + noise_std * torch.randn_like(data_mask_aug)
    shift = np.random.randint(-max_shift, max_shift + 1)
    data_aug = torch.roll(data_aug, shifts=shift, dims=-1)
    data_mask_aug = torch.roll(data_mask_aug, shifts=shift, dims=-1)
    return (data_aug, data_mask_aug)

def model_finetune_open_known(model, train_dl, train_dl_mask, device, model_optimizer, model_scheduler, class_scheduler, classifier, classifier_optimizer, LSTM=None, LSTM_optimizer=None, LSTM_scheduler=None):
    model.train()
    classifier.train()
    if LSTM is not None:
        LSTM.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = []
    total_acc = []
    with tqdm(total=len(train_dl)) as pb:
        for ((data, labels), (data_mask, labels_mask)) in zip(train_dl, train_dl_mask):
            model_optimizer.zero_grad()
            classifier_optimizer.zero_grad()
            if LSTM_optimizer is not None:
                LSTM_optimizer.zero_grad()
            data = data.float().to(device)
            data_mask = data_mask.float().to(device)
            labels = labels.to(device)
            (data, labels) = flip_row(data, labels)
            (data_mask, labels_mask) = flip_row(data_mask, labels_mask)
            (data_mask, labels_mask) = (data_mask.float().to(device), labels_mask.to(device))
            if labels.ndim > 1:
                labels = labels.argmax(dim=1)
            labels = labels.long().to(device)
            if labels.max().item() >= classifier.classifier[-1].out_features:
                raise ValueError(f'Training label exceeds the known-class range: max label={labels.max().item()}, but the classifier has {classifier.classifier[-1].out_features} outputs. Training labels must be known classes 0 through {classifier.classifier[-1].out_features - 1}。')
            h_ori = model(data_mask, data)
            if LSTM is not None:
                h_ori = LSTM(h_ori)
            logits_ori = classifier(h_ori)
            loss_ce = criterion(logits_ori, labels)
            (data_aug, data_mask_aug) = channel_augment(data=data, data_mask=data_mask, noise_std=0.01, scale_std=0.03, max_shift=2)
            h_aug = model(data_mask_aug, data_aug)
            if LSTM is not None:
                h_aug = LSTM(h_aug)
            logits_aug = classifier(h_aug)
            loss_ce_ori = criterion(logits_ori, labels)
            loss_ce_aug = criterion(logits_aug, labels)
            loss_ce = (loss_ce_ori + loss_ce_aug) / 2
            h_sup = h_ori.reshape(h_ori.size(0), -1)
            loss_supcon = supervised_contrastive_loss(features=h_sup, labels=labels, temperature=0.1)
            p_ori = torch.softmax(logits_ori.detach(), dim=1)
            log_p_aug = torch.log_softmax(logits_aug, dim=1)
            loss_kl = F.kl_div(log_p_aug, p_ori, reduction='batchmean')
            energy_ori = energy_score(logits_ori.detach(), temperature=1.0)
            energy_aug = energy_score(logits_aug, temperature=1.0)
            loss_energy_cons = F.mse_loss(energy_aug, energy_ori)
            lambda_kl = 0.02
            lambda_sup = 0.0
            lambda_energy = 0.0
            loss = loss_ce + lambda_kl * loss_kl + lambda_sup * loss_supcon + lambda_energy * loss_energy_cons
            predictions = logits_ori
            acc_bs = (labels == predictions.argmax(dim=1)).float().mean()
            total_loss.append(loss.item())
            total_acc.append(acc_bs.item())
            loss.backward()
            model_optimizer.step()
            classifier_optimizer.step()
            if LSTM_optimizer is not None:
                LSTM_optimizer.step()
            pb.update(1)
    total_loss = torch.tensor(total_loss).mean()
    total_acc = torch.tensor(total_acc).mean()
    model_scheduler.step()
    class_scheduler.step()
    if LSTM_scheduler is not None:
        LSTM_scheduler.step()
    return (total_loss, total_acc)
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

def model_finetune_open_known_fast(model, train_dl, train_dl_mask, device, model_optimizer, model_scheduler, class_scheduler, classifier, classifier_optimizer, LSTM=None, LSTM_optimizer=None, LSTM_scheduler=None, lambda_kl=0.02, use_amp=True, verify_labels=True, merge_views=True):
    """Train the known-class classifier. Return mean training loss and mean training accuracy."""
    device = torch.device(device)
    amp_enabled = bool(use_amp and device.type == 'cuda')
    model_trainable = any((param.requires_grad for param in model.parameters()))
    if model_trainable:
        model.train()
    else:
        model.eval()
    classifier.train()
    if LSTM is not None:
        if any((param.requires_grad for param in LSTM.parameters())):
            LSTM.train()
        else:
            LSTM.eval()
    if len(train_dl) != len(train_dl_mask):
        raise ValueError(f'The two DataLoaders have different numbers of batches: {len(train_dl)} != {len(train_dl_mask)}')
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=amp_enabled)
    loss_sum = 0.0
    correct_sum = 0
    sample_count = 0

    def extract_tensor(output):
        """Extract a tensor from a tensor, tuple, list or dictionary model output."""
        if torch.is_tensor(output):
            return output
        if isinstance(output, (tuple, list)):
            for item in output:
                if torch.is_tensor(item):
                    return item
        if isinstance(output, dict):
            for key in ('features', 'feature', 'embedding', 'output'):
                if key in output and torch.is_tensor(output[key]):
                    return output[key]
        raise TypeError('No usable tensor was found in the model output.')
    progress_bar = tqdm(zip(train_dl, train_dl_mask), total=len(train_dl), desc='Training')
    for (batch_2ch, batch_10ch) in progress_bar:
        (data, labels) = batch_2ch
        (data_mask, labels_mask) = batch_10ch
        if labels.ndim > 1:
            labels = labels.argmax(dim=1)
        if labels_mask.ndim > 1:
            labels_mask = labels_mask.argmax(dim=1)
        labels = labels.long()
        labels_mask = labels_mask.long()
        if verify_labels and (not torch.equal(labels, labels_mask)):
            raise ValueError('Labels in the two-channel and ten-channel DataLoaders are not aligned.')
        data = data.float().to(device, non_blocking=True)
        data_mask = data_mask.float().to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        labels_mask = labels_mask.to(device, non_blocking=True)
        (data_aug, labels_aug) = flip_row(data, labels)
        (data_mask_aug, labels_mask_aug) = flip_row(data_mask, labels_mask)
        if verify_labels:
            if not torch.equal(labels_aug, labels_mask_aug):
                raise ValueError('The augmented views have misaligned labels.')
            if data_aug.size(0) != labels_aug.size(0):
                raise ValueError(f'Augmented two-channel data and label counts differ: {data_aug.size(0)} != {labels_aug.size(0)}')
            if data_mask_aug.size(0) != labels_mask_aug.size(0):
                raise ValueError(f'Augmented ten-channel data and label counts differ: {data_mask_aug.size(0)} != {labels_mask_aug.size(0)}')
        if model_trainable and model_optimizer is not None:
            model_optimizer.zero_grad(set_to_none=True)
        classifier_optimizer.zero_grad(set_to_none=True)
        if LSTM is not None and LSTM_optimizer is not None:
            LSTM_optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp_enabled):
            if merge_views:
                data_all = torch.cat([data, data_aug], dim=0)
                data_mask_all = torch.cat([data_mask, data_mask_aug], dim=0)
                if model_trainable:
                    features_all = model(data_mask_all, data_all)
                else:
                    with torch.no_grad():
                        features_all = model(data_mask_all, data_all)
                features_all = extract_tensor(features_all)
                if LSTM is not None:
                    features_all = LSTM(features_all)
                    features_all = extract_tensor(features_all)
                logits_all = classifier(features_all)
                original_batch_size = data.size(0)
                logits_original = logits_all[:original_batch_size]
                logits_augmented = logits_all[original_batch_size:]
            else:
                if model_trainable:
                    features_original = model(data_mask, data)
                    features_augmented = model(data_mask_aug, data_aug)
                else:
                    with torch.no_grad():
                        features_original = model(data_mask, data)
                        features_augmented = model(data_mask_aug, data_aug)
                features_original = extract_tensor(features_original)
                features_augmented = extract_tensor(features_augmented)
                if LSTM is not None:
                    features_original = extract_tensor(LSTM(features_original))
                    features_augmented = extract_tensor(LSTM(features_augmented))
                logits_original = classifier(features_original)
                logits_augmented = classifier(features_augmented)
            if logits_original.size(0) != labels.size(0):
                raise ValueError(f'Classifier output and label counts differ: {logits_original.size(0)} != {labels.size(0)}')
            classification_loss = criterion(logits_original, labels)
            if lambda_kl > 0:
                log_prob_original = F.log_softmax(logits_original, dim=1)
                log_prob_augmented = F.log_softmax(logits_augmented, dim=1)
                prob_original = log_prob_original.exp()
                prob_augmented = log_prob_augmented.exp()
                kl_original_to_augmented = F.kl_div(log_prob_original, prob_augmented.detach(), reduction='batchmean')
                kl_augmented_to_original = F.kl_div(log_prob_augmented, prob_original.detach(), reduction='batchmean')
                consistency_loss = 0.5 * (kl_original_to_augmented + kl_augmented_to_original)
                loss = classification_loss + lambda_kl * consistency_loss
            else:
                loss = classification_loss
        scaler.scale(loss).backward()
        if model_trainable and model_optimizer is not None:
            scaler.step(model_optimizer)
        scaler.step(classifier_optimizer)
        if LSTM is not None and LSTM_optimizer is not None:
            scaler.step(LSTM_optimizer)
        scaler.update()
        with torch.no_grad():
            predictions = logits_original.argmax(dim=1)
            batch_size = labels.size(0)
            loss_sum += loss.detach().item() * batch_size
            correct_sum += (predictions == labels).sum().item()
            sample_count += batch_size
        progress_bar.set_postfix(loss=f'{loss_sum / sample_count:.4f}', acc=f'{correct_sum / sample_count:.4f}')
    if model_trainable and model_scheduler is not None:
        model_scheduler.step()
    if class_scheduler is not None:
        class_scheduler.step()
    if LSTM is not None and LSTM_scheduler is not None:
        LSTM_scheduler.step()
    mean_loss = loss_sum / max(sample_count, 1)
    mean_accuracy = correct_sum / max(sample_count, 1)
    return (mean_loss, mean_accuracy)

def energy_score(logits, temperature=1.0):
    """Energy = -T * logsumexp(logits / T). Higher energy indicates a greater likelihood of an unknown signal."""
    return -temperature * torch.logsumexp(logits / temperature, dim=1)

@torch.no_grad()
def collect_dual_view_energy_outputs(model, val_dl, val_dl_mask, device, classifier, unknown_label=9, temperature=1.0, beta=0.1, gamma=0.1, LSTM=None):
    """Collect dual-view energy outputs. The model expects model(data_mask, data), where data_mask has ten channels and data has two. Passing model(data, data) causes a channel mismatch."""
    model.eval()
    classifier.eval()
    if LSTM is not None:
        LSTM.eval()
    all_labels = []
    all_pred_known = []
    all_score = []
    with tqdm(total=len(val_dl), desc='Collect Dual-view Energy Outputs') as pb:
        for ((data, labels), (data_mask, labels_mask)) in zip(val_dl, val_dl_mask):
            data = data.float().to(device)
            data_mask = data_mask.float().to(device)
            labels = labels.to(device)
            if labels.ndim > 1:
                labels = labels.argmax(dim=1)
            labels = labels.long().to(device)
            h_view1 = model(data_mask, data)
            if LSTM is not None:
                h_view1 = LSTM(h_view1)
            logits_view1 = classifier(h_view1)
            pred_known = logits_view1.argmax(dim=1)
            energy_view1 = -temperature * torch.logsumexp(logits_view1 / temperature, dim=1)
            noise_std = 0.01
            data_mask_aug = data_mask + noise_std * torch.randn_like(data_mask)
            h_view2 = model(data_mask_aug, data)
            if LSTM is not None:
                h_view2 = LSTM(h_view2)
            logits_view2 = classifier(h_view2)
            energy_view2 = -temperature * torch.logsumexp(logits_view2 / temperature, dim=1)
            energy_gap = torch.abs(energy_view1 - energy_view2)
            p1 = torch.softmax(logits_view1, dim=1)
            p2 = torch.softmax(logits_view2, dim=1)
            log_p1 = torch.log(p1 + 1e-08)
            log_p2 = torch.log(p2 + 1e-08)
            kl_1_2 = F.kl_div(log_p1, p2, reduction='none').sum(dim=1)
            kl_2_1 = F.kl_div(log_p2, p1, reduction='none').sum(dim=1)
            kl_consistency = 0.5 * (kl_1_2 + kl_2_1)
            open_score = energy_view1 + beta * energy_gap + gamma * kl_consistency
            all_labels.append(labels.detach().cpu())
            all_pred_known.append(pred_known.detach().cpu())
            all_score.append(open_score.detach().cpu())
            pb.update(1)
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_pred_known = torch.cat(all_pred_known, dim=0).numpy()
    all_score = torch.cat(all_score, dim=0).numpy()
    return (all_labels, all_pred_known, all_score)

def augment_mask_view(data_mask, aug_type='noise', noise_std=0.01):
    """Augment data_mask while preserving its [B, 10, 128] shape."""
    if aug_type == 'noise':
        return data_mask + noise_std * torch.randn_like(data_mask)
    elif aug_type == 'scale':
        scale = 1.0 + 0.05 * torch.randn(data_mask.size(0), 1, 1).to(data_mask.device)
        return data_mask * scale
    elif aug_type == 'shift':
        shift = np.random.randint(-3, 4)
        return torch.roll(data_mask, shifts=shift, dims=-1)
    elif aug_type == 'drop_channel':
        out = data_mask.clone()
        (B, C, L) = out.shape
        drop_num = max(1, C // 5)
        for i in range(B):
            drop_idx = torch.randperm(C, device=out.device)[:drop_num]
            out[i, drop_idx, :] = 0.0
        return out
    elif aug_type == 'mask_time':
        out = data_mask.clone()
        (B, C, L) = out.shape
        mask_len = max(4, L // 10)
        start = np.random.randint(0, L - mask_len + 1)
        out[:, :, start:start + mask_len] = 0.0
        return out
    else:
        return data_mask

@torch.no_grad()
def collect_multiview_energy_outputs(model, val_dl, val_dl_mask, device, classifier, unknown_label=9, temperature=1.0, beta=0.03, gamma=0.03, LSTM=None, aug_types=('noise', 'scale', 'shift', 'drop_channel')):
    """Compute a multi-view energy open-set score. The base view uses classifier(model(data_mask, data)); augmented views use classifier(model(aug(data_mask), data)). open_score = E_base + beta * mean(abs(E_base - E_aug)) + gamma * mean(KL(p_base || p_aug)). Higher scores indicate greater unknownness."""
    model.eval()
    classifier.eval()
    if LSTM is not None:
        LSTM.eval()
    all_labels = []
    all_pred_known = []
    all_score = []
    with tqdm(total=len(val_dl), desc='Collect Multi-view Energy Outputs') as pb:
        for ((data, labels), (data_mask, labels_mask)) in zip(val_dl, val_dl_mask):
            data = data.float().to(device)
            data_mask = data_mask.float().to(device)
            labels = labels.to(device)
            if labels.ndim > 1:
                labels = labels.argmax(dim=1)
            labels = labels.long().to(device)
            h_base = model(data_mask, data)
            if LSTM is not None:
                h_base = LSTM(h_base)
            logits_base = classifier(h_base)
            pred_known = logits_base.argmax(dim=1)
            energy_base = -temperature * torch.logsumexp(logits_base / temperature, dim=1)
            p_base = torch.softmax(logits_base, dim=1)
            log_p_base = torch.log(p_base + 1e-08)
            energy_gap_list = []
            kl_list = []
            for aug_type in aug_types:
                data_mask_aug = augment_mask_view(data_mask, aug_type=aug_type, noise_std=0.01)
                h_aug = model(data_mask_aug, data)
                if LSTM is not None:
                    h_aug = LSTM(h_aug)
                logits_aug = classifier(h_aug)
                energy_aug = -temperature * torch.logsumexp(logits_aug / temperature, dim=1)
                p_aug = torch.softmax(logits_aug, dim=1)
                log_p_aug = torch.log(p_aug + 1e-08)
                energy_gap = torch.abs(energy_base - energy_aug)
                kl_base_aug = F.kl_div(log_p_base, p_aug, reduction='none').sum(dim=1)
                kl_aug_base = F.kl_div(log_p_aug, p_base, reduction='none').sum(dim=1)
                kl_consistency = 0.5 * (kl_base_aug + kl_aug_base)
                energy_gap_list.append(energy_gap)
                kl_list.append(kl_consistency)
            mean_energy_gap = torch.stack(energy_gap_list, dim=0).mean(dim=0)
            mean_kl = torch.stack(kl_list, dim=0).mean(dim=0)
            open_score = energy_base + beta * mean_energy_gap + gamma * mean_kl
            all_labels.append(labels.detach().cpu())
            all_pred_known.append(pred_known.detach().cpu())
            all_score.append(open_score.detach().cpu())
            pb.update(1)
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_pred_known = torch.cat(all_pred_known, dim=0).numpy()
    all_score = torch.cat(all_score, dim=0).numpy()
    all_score = (all_score - all_score.mean()) / (all_score.std() + 1e-08)
    return (all_labels, all_pred_known, all_score)

def build_classwise_thresholds(labels, pred_known, open_score, num_known_classes=9, unknown_label=9, q=0.95, use_correct_known=True):
    """Estimate one threshold per class from known validation samples. A larger q relaxes acceptance, potentially increasing known accuracy and reducing unknown accuracy; a smaller q has the opposite effect."""
    thresholds = np.zeros(num_known_classes)
    known_mask_all = labels != unknown_label
    global_scores = open_score[known_mask_all]
    if len(global_scores) == 0:
        raise ValueError('No known validation samples are available for threshold calibration.')
    global_threshold = np.quantile(global_scores, q)
    for c in range(num_known_classes):
        if use_correct_known:
            class_mask = (labels == c) & (pred_known == c)
        else:
            class_mask = labels == c
        class_scores = open_score[class_mask]
        if len(class_scores) > 0:
            thresholds[c] = np.quantile(class_scores, q)
        else:
            thresholds[c] = global_threshold
    return thresholds

def evaluate_classwise_open_set(labels, pred_known, open_score, thresholds, unknown_label=9):
    """Apply class-adaptive rejection: assign the unknown label if open_score > thresholds[pred_known]; otherwise keep pred_known."""
    pred_open = pred_known.copy()
    sample_thresholds = thresholds[pred_known]
    pred_open[open_score > sample_thresholds] = unknown_label
    known_mask = labels != unknown_label
    unknown_mask = labels == unknown_label
    known_acc = np.mean(pred_open[known_mask] == labels[known_mask]) if known_mask.sum() > 0 else 0.0
    unknown_acc = np.mean(pred_open[unknown_mask] == unknown_label) if unknown_mask.sum() > 0 else 0.0
    open_acc = np.mean(pred_open == labels)
    return {'known_acc': known_acc, 'unknown_acc': unknown_acc, 'open_acc': open_acc, 'pred_open': pred_open}

def search_best_classwise_threshold(model, val_dl, val_dl_mask, device, classifier, num_known_classes=9, unknown_label=9, q_values=None, temperature=1.0, beta=0.02, gamma=0.02, LSTM=None, min_known_acc=0.53):
    if q_values is None:
        q_values = np.arange(0.75, 0.931, 0.005)
    (labels, pred_known, open_score) = collect_dual_view_energy_outputs(model=model, val_dl=val_dl, val_dl_mask=val_dl_mask, device=device, classifier=classifier, temperature=temperature, beta=beta, gamma=gamma, LSTM=None, unknown_label=5)
    known_mask = labels != unknown_label
    closed_known_acc = np.mean(pred_known[known_mask] == labels[known_mask])
    best_result = None
    best_unknown_acc = -1.0
    fallback_result = None
    fallback_known_acc = -1.0
    for q in q_values:
        thresholds = build_classwise_thresholds(labels=labels, pred_known=pred_known, open_score=open_score, num_known_classes=num_known_classes, unknown_label=unknown_label, q=float(q), use_correct_known=True)
        result = evaluate_classwise_open_set(labels=labels, pred_known=pred_known, open_score=open_score, thresholds=thresholds, unknown_label=unknown_label)
        known_acc = result['known_acc']
        unknown_acc = result['unknown_acc']
        open_acc = result['open_acc']
        h_score = 2.0 * known_acc * unknown_acc / (known_acc + unknown_acc + 1e-12)
        if known_acc > fallback_known_acc:
            fallback_known_acc = known_acc
            fallback_result = {'known_acc': known_acc, 'unknown_acc': unknown_acc, 'open_acc': open_acc, 'h_score': h_score, 'thresholds': thresholds.copy(), 'best_q': float(q)}
        if known_acc < min_known_acc:
            continue
        if unknown_acc > best_unknown_acc:
            best_unknown_acc = unknown_acc
            best_result = {'known_acc': known_acc, 'unknown_acc': unknown_acc, 'open_acc': open_acc, 'h_score': h_score, 'thresholds': thresholds.copy(), 'best_q': float(q)}
    if best_result is None:
        print(f'[Warning] No q satisfies KnownAcc >= {min_known_acc:.4f}')
        print('Selecting the operating point with the highest known-class accuracy.')
        best_result = fallback_result
    print(f"[Best Class-wise Threshold] q={best_result['best_q']:.3f}, KnownAcc={best_result['known_acc']:.4f}, UnknownAcc={best_result['unknown_acc']:.4f}, OpenAcc={best_result['open_acc']:.4f}, HScore={best_result['h_score']:.4f}, CAcc={closed_known_acc:.4f}")
    print('Class-wise thresholds:')
    for (c, th) in enumerate(best_result['thresholds']):
        print(f'Class {c}: threshold={th:.4f}')
    return {'known_acc': best_result['known_acc'], 'unknown_acc': best_result['unknown_acc'], 'open_acc': best_result['open_acc'], 'h_score': best_result['h_score'], 'closed_known_acc': closed_known_acc, 'thresholds': best_result['thresholds'], 'best_q': best_result['best_q']}

@torch.no_grad()
def extract_feature(model, data, data_mask, device, LSTM=None):
    """Extract encoder features and reshape them to [B, D]."""
    model.eval()
    data = data.float().to(device)
    data_mask = data_mask.float().to(device)
    h = model(data_mask, data)
    if LSTM is not None:
        h = LSTM(h)
    if h.ndim == 3:
        if h.shape[1] != 128:
            h = h.permute(0, 2, 1)
        h = F.adaptive_avg_pool1d(h, 1).squeeze(-1)
    h = F.normalize(h, dim=1)
    return h

@torch.no_grad()
def build_class_prototypes(model, train_dl, train_dl_mask, device, num_known_classes=9, LSTM=None):
    """Compute one prototype per class from known training samples."""
    model.eval()
    feat_list = []
    label_list = []
    with tqdm(total=len(train_dl), desc='Build Prototypes') as pb:
        for ((data, labels), (data_mask, labels_mask)) in zip(train_dl, train_dl_mask):
            labels = labels.to(device)
            if labels.ndim > 1:
                labels = labels.argmax(dim=1)
            labels = labels.long().to(device)
            feat = extract_feature(model=model, data=data, data_mask=data_mask, device=device, LSTM=LSTM)
            feat_list.append(feat.detach().cpu())
            label_list.append(labels.detach().cpu())
            pb.update(1)
    feats = torch.cat(feat_list, dim=0)
    labels = torch.cat(label_list, dim=0)
    prototypes = []
    for c in range(num_known_classes):
        class_mask = labels == c
        if class_mask.sum() == 0:
            raise ValueError(f'Class {c} has no samples; its prototype cannot be computed.')
        proto_c = feats[class_mask].mean(dim=0)
        proto_c = F.normalize(proto_c, dim=0)
        prototypes.append(proto_c)
    prototypes = torch.stack(prototypes, dim=0).to(device)
    return prototypes

class SELayer_1d(nn.Module):

    def __init__(self, channel, reduction=16):
        super(SELayer_1d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.linear1 = nn.Sequential(nn.Linear(channel, channel // reduction, bias=False), nn.ReLU(inplace=True))
        self.linear2 = nn.Sequential(nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid())

    def forward(self, X_input):
        """
        X_input: shape = [B, C, L]
        """
        (b, c, _) = X_input.size()
        y = self.avg_pool(X_input)
        y = y.view(b, c)
        y = self.linear1(y)
        y = self.linear2(y)
        y = y.view(b, c, 1)
        return X_input * y.expand_as(X_input)

class target_classifier(nn.Module):

    def __init__(self, num_classes=33):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, num_classes))

    def forward(self, emb):
        if emb.shape[1] != 128:
            emb = emb.permute(0, 2, 1)
        feat = self.pool(emb).squeeze(-1)
        feat = F.normalize(feat, dim=1)
        return self.classifier(feat)

class LS_TM(nn.Module):

    def __init__(self):
        super(LS_TM, self).__init__()
        self.LSTM = nn.LSTM(input_size=16, hidden_size=16, num_layers=2, batch_first=True, bidirectional=True)

    def forward(self, x):
        out = self.LSTM(x)
        return out

def build_model(lr, device, chkpoint=None):
    premodel = pre_model(model_z, model_LSTM, model_r, model_s, model_s1).to(device)
    if chkpoint:
        pretrained_dict = chkpoint['model_state_dict']
        model_dict = premodel.state_dict()
        model_dict.update(pretrained_dict)
        premodel.load_state_dict(model_dict)
    ft_model = FinetuneEncoder(premodel).to(device)
    ft_model_dict = ft_model.state_dict()
    LSTM = SELayer_1d(128).to(device)
    classifier = target_classifier(num_classes=int(os.environ.get('RADIO_SSL_NUM_KNOWN_CLASSES', '33'))).to(device)
    model_optimizer = torch.optim.AdamW(ft_model.parameters(), lr=0.003, weight_decay=0.05)
    classifier_optimizer = torch.optim.AdamW(classifier.parameters(), lr=0.0001, weight_decay=0.05)
    LSTM_optimizer = torch.optim.AdamW(LSTM.parameters(), lr=0.0001, weight_decay=0.01)
    model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=model_optimizer, T_max=100)
    LSTM_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=LSTM_optimizer, T_max=200)
    class_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=classifier_optimizer, T_max=100)
    return (ft_model, classifier, classifier_optimizer, class_scheduler, model_optimizer, model_scheduler, LSTM, LSTM_optimizer, LSTM_scheduler)

def model_test(model, test_dl, device, classifier=None):
    model.eval()
    classifier.eval()
    total_loss = []
    total_acc = []
    total_auc = []
    total_prc = []
    (total_precision, total_recall, total_f1) = ([], [], [])
    criterion = nn.CrossEntropyLoss()
    outs = np.array([])
    trgs = np.array([])
    emb_test_all = []
    with torch.no_grad():
        (labels_numpy_all, pred_numpy_all) = (np.zeros(1), np.zeros(1))
        for (data, labels) in test_dl:
            (data, labels) = (data.float().to(device), labels.long().to(device))
            (h, _) = model(data)
            fea_concat = h
            predictions_test = classifier(fea_concat)
            labels = labels.argmax(dim=1)
            labels = labels.long().to(device)
            fea_concat_flat = fea_concat.reshape(fea_concat.shape[0], -1)
            emb_test_all.append(fea_concat_flat)
            loss = criterion(predictions_test, labels)
            acc_bs = labels.eq(predictions_test.detach().argmax(dim=1)).float().mean()
            onehot_label = F.one_hot(labels)
            pred_numpy = predictions_test.detach().cpu().numpy()
            labels_numpy = labels.detach().cpu().numpy()
            try:
                auc_bs = roc_auc_score(onehot_label.detach().cpu().numpy(), pred_numpy, average='macro', multi_class='ovr')
            except:
                auc_bs = 0.0
            try:
                prc_bs = average_precision_score(onehot_label.detach().cpu().numpy(), pred_numpy, average='macro')
            except:
                prc_bs = 0.0
            pred_numpy = np.argmax(pred_numpy, axis=1)
            precision = precision_score(labels_numpy, pred_numpy, average='macro')
            recall = recall_score(labels_numpy, pred_numpy, average='macro')
            F1 = f1_score(labels_numpy, pred_numpy, average='macro')
            total_acc.append(acc_bs)
            if auc_bs != 0:
                total_auc.append(auc_bs)
            if prc_bs != 0:
                total_prc.append(prc_bs)
            total_precision.append(precision)
            total_recall.append(recall)
            total_f1.append(F1)
            total_loss.append(loss.item())
            pred = predictions_test.max(1, keepdim=True)[1]
            outs = np.append(outs, pred.cpu().numpy())
            trgs = np.append(trgs, labels.data.cpu().numpy())
            labels_numpy_all = np.concatenate((labels_numpy_all, labels_numpy))
            pred_numpy_all = np.concatenate((pred_numpy_all, pred_numpy))
    labels_numpy_all = labels_numpy_all[1:]
    pred_numpy_all = pred_numpy_all[1:]
    precision = precision_score(labels_numpy_all, pred_numpy_all, average='macro', zero_division=0)
    recall = recall_score(labels_numpy_all, pred_numpy_all, average='macro', zero_division=0)
    F1 = f1_score(labels_numpy_all, pred_numpy_all, average='macro', zero_division=0)
    acc = accuracy_score(labels_numpy_all, pred_numpy_all)
    total_loss = torch.tensor(total_loss).mean()
    total_acc = torch.tensor(total_acc).mean()
    total_auc = torch.tensor(total_auc).mean()
    total_prc = torch.tensor(total_prc).mean()
    performance = [acc * 100, precision * 100, recall * 100, F1 * 100, total_auc * 100, total_prc * 100]
    emb_test_all = torch.cat(tuple(emb_test_all))
    return (total_loss, total_acc, total_auc, total_prc, emb_test_all, trgs, performance)

@torch.no_grad()
def energy_open_set_evaluate(model, val_dl, val_dl_mask, device, classifier, threshold=-2.0, unknown_label=9, temperature=1.0, LSTM=None, verbose=True):
    """Evaluate energy-based open-set recognition. Report known-class accuracy, unknown-class accuracy and overall open-set accuracy."""
    model.eval()
    classifier.eval()
    if LSTM is not None:
        LSTM.eval()
    known_correct = 0
    known_total = 0
    unknown_correct = 0
    unknown_total = 0
    with tqdm(total=len(val_dl), desc='Energy Open-set Eval') as pb:
        for ((data, labels), (data_mask, labels_mask)) in zip(val_dl, val_dl_mask):
            data = data.float().to(device)
            data_mask = data_mask.float().to(device)
            labels = labels.to(device)
            if labels.ndim > 1:
                labels = labels.argmax(dim=1)
            labels = labels.long().to(device)
            h = model(data_mask, data)
            if LSTM is not None:
                h = LSTM(h)
            logits = classifier(h)
            pred_known = logits.argmax(dim=1)
            energy = -temperature * torch.logsumexp(logits / temperature, dim=1)
            pred_open = pred_known.clone()
            pred_open[energy > threshold] = unknown_label
            known_mask = labels != unknown_label
            unknown_mask = labels == unknown_label
            if known_mask.sum() > 0:
                known_correct += (pred_open[known_mask] == labels[known_mask]).sum().item()
                known_total += known_mask.sum().item()
            if unknown_mask.sum() > 0:
                unknown_correct += (pred_open[unknown_mask] == unknown_label).sum().item()
                unknown_total += unknown_mask.sum().item()
            pb.update(1)
    known_acc = known_correct / known_total if known_total > 0 else 0.0
    unknown_acc = unknown_correct / unknown_total if unknown_total > 0 else 0.0
    total_correct = known_correct + unknown_correct
    total_samples = known_total + unknown_total
    open_acc = total_correct / total_samples if total_samples > 0 else 0.0
    if verbose:
        print(f'Threshold: {threshold:.4f}')
        print(f'KnownAcc: {known_acc:.4f}')
        print(f'UnknownAcc: {unknown_acc:.4f}')
        print(f'OpenAcc: {open_acc:.4f}')
    return {'known_acc': known_acc, 'unknown_acc': unknown_acc, 'open_acc': open_acc, 'threshold': threshold}

def search_best_energy_threshold(model, val_dl, val_dl_mask, device, classifier, unknown_label=9, thresholds=None, temperature=1.0, LSTM=None):
    """Select the energy threshold with the highest validation open-set accuracy."""
    if thresholds is None:
        thresholds = np.arange(-10.0, 2.01, 0.5)
    best_result = None
    best_score = -1.0
    for th in thresholds:
        result = energy_open_set_evaluate(model=model, val_dl=val_dl, val_dl_mask=val_dl_mask, device=device, classifier=classifier, threshold=float(th), unknown_label=unknown_label, temperature=temperature, LSTM=LSTM, verbose=False)
        score = result['open_acc']
        if score > best_score:
            best_score = score
            best_result = result
    print(f"[Best Energy Threshold] th={best_result['threshold']:.4f}, KnownAcc={best_result['known_acc']:.4f}, UnknownAcc={best_result['unknown_acc']:.4f}, OpenAcc={best_result['open_acc']:.4f}")
    return best_result

def classwise_dual_view_open_set_test(model, test_dl, test_dl_mask, device, classifier, thresholds, unknown_label=9, temperature=1.0, beta=0.05, gamma=0.05, LSTM=None, test_seed=2026):
    """Evaluate the test set using class-wise thresholds fixed during validation. Do not search q or recompute thresholds. Also report closed-known accuracy to describe known-class classification difficulty."""
    model.eval()
    classifier.eval()
    if LSTM is not None:
        LSTM.eval()
    torch.manual_seed(test_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(test_seed)
        torch.cuda.manual_seed_all(test_seed)
    (labels, pred_known, open_score) = collect_dual_view_energy_outputs(model=model, val_dl=test_dl, val_dl_mask=test_dl_mask, device=device, classifier=classifier, unknown_label=unknown_label, temperature=temperature, beta=beta, gamma=gamma, LSTM=LSTM)
    thresholds = np.asarray(thresholds, dtype=np.float32)
    num_known_classes = len(thresholds)
    if pred_known.min() < 0 or pred_known.max() >= num_known_classes:
        raise ValueError(f'Invalid pred_known range: min={pred_known.min()}, max={pred_known.max()}, but thresholds cover only {num_known_classes} classes')
    known_mask = labels != unknown_label
    if known_mask.sum() > 0:
        closed_known_acc = np.mean(pred_known[known_mask] == labels[known_mask])
    else:
        closed_known_acc = 0.0
    result = evaluate_classwise_open_set(labels=labels, pred_known=pred_known, open_score=open_score, thresholds=thresholds, unknown_label=unknown_label)
    sample_thresholds = thresholds[pred_known]
    pred_open = pred_known.copy()
    pred_open[open_score > sample_thresholds] = unknown_label
    if known_mask.sum() > 0:
        known_reject_rate = np.mean(pred_open[known_mask] == unknown_label)
    else:
        known_reject_rate = 0.0
    correct_known_mask = known_mask & (pred_known == labels)
    if correct_known_mask.sum() > 0:
        correct_known_false_reject_rate = np.mean(pred_open[correct_known_mask] == unknown_label)
    else:
        correct_known_false_reject_rate = 0.0
    print('\n' + '=' * 60)
    print('Fixed-threshold Open-set Test')
    print('=' * 60)
    print(f'ClosedKnownAcc: {closed_known_acc:.4f}')
    print(f"KnownAcc:       {result['known_acc']:.4f}")
    print(f"UnknownAcc:     {result['unknown_acc']:.4f}")
    print(f"OpenAcc:        {result['open_acc']:.4f}")
    print(f'KnownRejectRate: {known_reject_rate:.4f}')
    print(f'CorrectKnownFalseRejectRate: {correct_known_false_reject_rate:.4f}')
    print('=' * 60)
    return {'closed_known_acc': closed_known_acc, 'known_acc': result['known_acc'], 'unknown_acc': result['unknown_acc'], 'open_acc': result['open_acc'], 'known_reject_rate': known_reject_rate, 'correct_known_false_reject_rate': correct_known_false_reject_rate, 'pred_known': pred_known, 'pred_open': pred_open, 'labels': labels, 'open_score': open_score, 'thresholds': thresholds}

@torch.no_grad()
def compute_class_prototypes(model, train_dl, train_dl_mask, device, num_known_classes=9, LSTM=None):
    """Compute prototypes from known training samples only. Average normalized features within each class, then L2-normalize each prototype."""
    model.eval()
    if LSTM is not None:
        LSTM.eval()
    feature_sums = None
    class_counts = torch.zeros(num_known_classes, device=device)
    for ((data, labels), (data_mask, labels_mask)) in zip(train_dl, train_dl_mask):
        data = data.float().to(device)
        data_mask = data_mask.float().to(device)
        labels = labels.to(device)
        if labels.ndim > 1:
            labels = labels.argmax(dim=1)
        labels = labels.long()
        h = model(data_mask, data)
        if LSTM is not None:
            h = LSTM(h)
        h_vec = h.reshape(h.size(0), -1)
        h_vec = F.normalize(h_vec, p=2, dim=1)
        if feature_sums is None:
            feature_sums = torch.zeros(num_known_classes, h_vec.size(1), device=device)
        for c in range(num_known_classes):
            mask = labels == c
            if mask.any():
                feature_sums[c] += h_vec[mask].sum(dim=0)
                class_counts[c] += mask.sum()
    prototypes = torch.zeros_like(feature_sums)
    for c in range(num_known_classes):
        if class_counts[c] > 0:
            prototypes[c] = feature_sums[c] / class_counts[c]
    prototypes = F.normalize(prototypes, p=2, dim=1)
    return prototypes

@torch.no_grad()
def collect_multiview_energy_prototype_outputs(model, val_dl, val_dl_mask, device, classifier, prototypes=None, unknown_label=9, temperature=1.0, beta=0.05, gamma=0.05, num_aug_views=3, noise_std=0.2, LSTM=None, eval_seed=2026):
    """Combine multi-view energy with prototype distance. Return labels, pred_known, base_score and proto_distance. Compute open_score = base_score + eta * proto_distance afterwards so that eta can be searched without repeating inference."""
    model.eval()
    classifier.eval()
    if LSTM is not None:
        LSTM.eval()
    generator = torch.Generator(device=device)
    generator.manual_seed(eval_seed)
    all_labels = []
    all_pred_known = []
    all_base_score = []
    all_proto_distance = []
    with tqdm(total=len(val_dl), desc='Collect Multi-view Energy + Prototype') as pb:
        for ((data, labels), (data_mask, labels_mask)) in zip(val_dl, val_dl_mask):
            data = data.float().to(device)
            data_mask = data_mask.float().to(device)
            labels = labels.to(device)
            if labels.ndim > 1:
                labels = labels.argmax(dim=1)
            labels = labels.long()
            h0 = model(data_mask, data)
            if LSTM is not None:
                h0 = LSTM(h0)
            logits0 = classifier(h0)
            pred_known = logits0.argmax(dim=1)
            energy0 = -temperature * torch.logsumexp(logits0 / temperature, dim=1)
            p0 = torch.softmax(logits0, dim=1)
            log_p0 = torch.log(p0 + 1e-08)
            energy_gap_sum = torch.zeros_like(energy0)
            kl_sum = torch.zeros_like(energy0)
            for _ in range(num_aug_views):
                noise = torch.randn(data_mask.shape, generator=generator, device=data_mask.device, dtype=data_mask.dtype)
                data_mask_aug = data_mask + noise_std * noise
                h_aug = model(data_mask_aug, data)
                if LSTM is not None:
                    h_aug = LSTM(h_aug)
                logits_aug = classifier(h_aug)
                energy_aug = -temperature * torch.logsumexp(logits_aug / temperature, dim=1)
                energy_gap_sum += torch.abs(energy0 - energy_aug)
                p_aug = torch.softmax(logits_aug, dim=1)
                log_p_aug = torch.log(p_aug + 1e-08)
                kl_0_aug = F.kl_div(log_p0, p_aug, reduction='none').sum(dim=1)
                kl_aug_0 = F.kl_div(log_p_aug, p0, reduction='none').sum(dim=1)
                kl_sym = 0.5 * (kl_0_aug + kl_aug_0)
                kl_sum += kl_sym
            mean_energy_gap = energy_gap_sum / num_aug_views
            mean_kl = kl_sum / num_aug_views
            base_score = energy0 + beta * mean_energy_gap + gamma * mean_kl
            if prototypes is not None:
                h_vec = h0.reshape(h0.size(0), -1)
                h_vec = F.normalize(h_vec, p=2, dim=1)
                pred_proto = prototypes[pred_known]
                cosine_sim = torch.sum(h_vec * pred_proto, dim=1)
                proto_distance = 1.0 - cosine_sim
            else:
                proto_distance = torch.zeros_like(energy0)
            all_labels.append(labels.detach().cpu())
            all_pred_known.append(pred_known.detach().cpu())
            all_base_score.append(base_score.detach().cpu())
            all_proto_distance.append(proto_distance.detach().cpu())
            pb.update(1)
    labels = torch.cat(all_labels).numpy()
    pred_known = torch.cat(all_pred_known).numpy()
    base_score = torch.cat(all_base_score).numpy()
    proto_distance = torch.cat(all_proto_distance).numpy()
    return (labels, pred_known, base_score, proto_distance)

def calculate_h_score(known_acc, unknown_acc):
    """H-score: the harmonic mean of known-class and unknown-class accuracies."""
    return 2.0 * known_acc * unknown_acc / (known_acc + unknown_acc + 1e-12)

def search_best_multiview_prototype_threshold(model, val_dl, val_dl_mask, device, classifier, prototypes, num_known_classes=9, unknown_label=9, q_values=None, eta_values=None, temperature=1.0, beta=0.05, gamma=0.05, num_aug_views=3, noise_std=0.01, LSTM=None, min_known_acc=0.5, eval_seed=2026):
    if q_values is None:
        q_values = np.arange(0.8, 0.941, 0.005)
    if eta_values is None:
        eta_values = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
    (labels, pred_known, base_score, proto_distance) = collect_multiview_energy_proto_outputs(model=model, val_dl=val_dl, val_dl_mask=val_dl_mask, device=device, classifier=classifier, prototypes=prototypes, temperature=temperature, beta=beta, gamma=gamma, num_aug_views=num_aug_views, noise_std=noise_std, LSTM=LSTM, eval_seed=eval_seed)
    known_mask = labels != unknown_label
    if known_mask.sum() > 0:
        closed_known_acc = np.mean(pred_known[known_mask] == labels[known_mask])
    else:
        closed_known_acc = 0.0
    best_valid = None
    best_any = None
    all_results = []
    for eta in eta_values:
        open_score = base_score + float(eta) * proto_distance
        for q in q_values:
            thresholds = build_classwise_thresholds(labels=labels, pred_known=pred_known, open_score=open_score, num_known_classes=num_known_classes, unknown_label=unknown_label, q=float(q), use_correct_known=True)
            result = evaluate_classwise_open_set(labels=labels, pred_known=pred_known, open_score=open_score, thresholds=thresholds, unknown_label=unknown_label)
            known_acc = float(result['known_acc'])
            unknown_acc = float(result['unknown_acc'])
            open_acc = float(result['open_acc'])
            h_score = calculate_h_score(known_acc, unknown_acc)
            current = {'known_acc': known_acc, 'unknown_acc': unknown_acc, 'open_acc': open_acc, 'h_score': h_score, 'thresholds': thresholds.copy(), 'best_q': float(q), 'best_eta': float(eta)}
            all_results.append(current)
            if best_any is None or h_score > best_any['h_score']:
                best_any = current
            if min_known_acc is not None and known_acc < min_known_acc:
                continue
            if best_valid is None or h_score > best_valid['h_score']:
                best_valid = current
            elif abs(h_score - best_valid['h_score']) < 1e-12 and open_acc > best_valid['open_acc']:
                best_valid = current
    if best_valid is None:
        if min_known_acc is not None:
            print(f'[Warning] No combination satisfies KnownAcc >= {min_known_acc:.4f}')
            print('Selecting the combination with the highest global H-score.')
        best_result = best_any
    else:
        best_result = best_valid
    print('\n==================================================')
    print('[Best Multi-view Prototype Threshold]')
    print(f"q={best_result['best_q']:.3f}, eta={best_result['best_eta']:.3f}, CAcc={closed_known_acc:.4f}, KnownAcc={best_result['known_acc']:.4f}, UnknownAcc={best_result['unknown_acc']:.4f}, OpenAcc={best_result['open_acc']:.4f}, HScore={best_result['h_score']:.4f}")
    print('==================================================')
    print('Class-wise thresholds:')
    for (c, th) in enumerate(best_result['thresholds']):
        print(f'Class {c}: threshold={th:.4f}')
    if min_known_acc is not None:
        valid_results = [r for r in all_results if r['known_acc'] >= min_known_acc]
    else:
        valid_results = all_results
    if len(valid_results) == 0:
        valid_results = all_results
    top5 = sorted(valid_results, key=lambda x: x['h_score'], reverse=True)[:5]
    print('\nTop-5 q / eta combinations:')
    for (i, r) in enumerate(top5, start=1):
        print(f"{i}. q={r['best_q']:.3f}, eta={r['best_eta']:.3f}, Known={r['known_acc']:.4f}, Unknown={r['unknown_acc']:.4f}, Open={r['open_acc']:.4f}, H={r['h_score']:.4f}")
    return {'known_acc': best_result['known_acc'], 'unknown_acc': best_result['unknown_acc'], 'open_acc': best_result['open_acc'], 'h_score': best_result['h_score'], 'closed_known_acc': float(closed_known_acc), 'thresholds': best_result['thresholds'], 'best_q': best_result['best_q'], 'best_eta': best_result['best_eta'], 'beta': beta, 'gamma': gamma, 'temperature': temperature, 'num_aug_views': num_aug_views, 'noise_std': noise_std}
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def test_open_set_by_snr(model, classifier, device, thresholds, dataset_root='./open_set_train_test/WBFM/test_by_snr', snr_list=None, batch_size=128, unknown_label=5, temperature=1.0, beta=0.05, gamma=0.05, LSTM=None, test_seed=2026, num_workers=0):
    """Run open-set inference once on pooled SNR samples, then report metrics by SNR. This shares the same randomized dual-view inference across SNR groups, avoids resetting the seed per SNR and keeps pooled and per-SNR metrics aligned."""
    dataset_root = Path(dataset_root)
    device = torch.device(device)
    if snr_list is None:
        snr_list = list(range(-20, 20, 2))
    thresholds = np.asarray(thresholds, dtype=np.float32)
    if thresholds.ndim != 1:
        raise ValueError(f'thresholds must be a one-dimensional array; received shape={thresholds.shape}')
    all_raw = []
    all_sequence = []
    all_labels = []
    all_snrs = []
    for snr in snr_list:
        snr = int(snr)
        snr_dir = dataset_root / f'snr_{snr}dB'
        raw_path = snr_dir / f'raw_test_{snr}dB_data.npy'
        sequence_path = snr_dir / f'sequence_test_{snr}dB_data.npy'
        label_path = snr_dir / f'test_{snr}dB_label.npy'
        for path in [raw_path, sequence_path, label_path]:
            if not path.exists():
                raise FileNotFoundError(f'Test file not found: {path}')
        raw_data = np.load(raw_path, allow_pickle=False).astype(np.float32)
        sequence_data = np.load(sequence_path, allow_pickle=False).astype(np.float32)
        labels = np.load(label_path, allow_pickle=False).astype(np.int64)
        if not len(raw_data) == len(sequence_data) == len(labels):
            raise ValueError(f'SNR={snr}Sample counts differ.')
        print(f'SNR={snr:>3} dB | N={len(labels)} | labels={np.unique(labels, return_counts=True)}')
        all_raw.append(raw_data)
        all_sequence.append(sequence_data)
        all_labels.append(labels)
        all_snrs.append(np.full(len(labels), snr, dtype=np.int32))
    raw_data = np.concatenate(all_raw, axis=0)
    sequence_data = np.concatenate(all_sequence, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    snrs_np = np.concatenate(all_snrs, axis=0)
    print('\nFull test set:')
    print('raw:', raw_data.shape)
    print('sequence:', sequence_data.shape)
    print('labels:', labels_np.shape)
    print('label counts:', np.unique(labels_np, return_counts=True))
    raw_tensor = torch.from_numpy(np.ascontiguousarray(raw_data)).float()
    sequence_tensor = torch.from_numpy(np.ascontiguousarray(sequence_data)).float()
    label_tensor = torch.from_numpy(np.ascontiguousarray(labels_np)).long()
    raw_dataset = TensorDataset(raw_tensor, label_tensor)
    sequence_dataset = TensorDataset(sequence_tensor, label_tensor)
    raw_loader = DataLoader(raw_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=device.type == 'cuda')
    sequence_loader = DataLoader(sequence_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=device.type == 'cuda')
    torch.manual_seed(test_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(test_seed)
        torch.cuda.manual_seed_all(test_seed)
    model.eval()
    classifier.eval()
    (labels_out, pred_known, open_score) = collect_dual_view_energy_outputs(model=model, val_dl=raw_loader, val_dl_mask=sequence_loader, device=device, classifier=classifier, unknown_label=unknown_label, temperature=temperature, beta=beta, gamma=gamma, LSTM=LSTM)
    if not np.array_equal(labels_out, labels_np):
        raise RuntimeError('Model output labels do not match the original label order.')
    full_result = evaluate_classwise_open_set(labels=labels_out, pred_known=pred_known, open_score=open_score, thresholds=thresholds, unknown_label=unknown_label)
    known_mask = labels_out < unknown_label
    closed_known_acc = np.mean(pred_known[known_mask] == labels_out[known_mask])
    sample_thresholds = thresholds[pred_known]
    pred_open = pred_known.copy()
    pred_open[open_score > sample_thresholds] = unknown_label
    known_reject_rate = np.mean(pred_open[known_mask] == unknown_label)
    correct_known_mask = known_mask & (pred_known == labels_out)
    if correct_known_mask.sum() > 0:
        correct_known_frr = np.mean(pred_open[correct_known_mask] == unknown_label)
    else:
        correct_known_frr = 0.0
    print('\n' + '=' * 60)
    print('Full test performance')
    print('=' * 60)
    print(f'ClosedKnownAcc: {closed_known_acc:.4f}')
    print(f"KnownAcc:       {full_result['known_acc']:.4f}")
    print(f"UnknownAcc:     {full_result['unknown_acc']:.4f}")
    print(f"OpenAcc:        {full_result['open_acc']:.4f}")
    print(f'KnownReject:    {known_reject_rate:.4f}')
    print(f'CorrectKnownFRR:{correct_known_frr:.4f}')
    results_by_snr = {}
    for snr in snr_list:
        snr = int(snr)
        snr_mask = snrs_np == snr
        labels_snr = labels_out[snr_mask]
        pred_snr = pred_known[snr_mask]
        score_snr = open_score[snr_mask]
        result = evaluate_classwise_open_set(labels=labels_snr, pred_known=pred_snr, open_score=score_snr, thresholds=thresholds, unknown_label=unknown_label)
        known_mask_snr = labels_snr < unknown_label
        closed_snr = np.mean(pred_snr[known_mask_snr] == labels_snr[known_mask_snr])
        sample_thresholds_snr = thresholds[pred_snr]
        pred_open_snr = pred_snr.copy()
        pred_open_snr[score_snr > sample_thresholds_snr] = unknown_label
        known_reject_snr = np.mean(pred_open_snr[known_mask_snr] == unknown_label)
        correct_known_snr = known_mask_snr & (pred_snr == labels_snr)
        if correct_known_snr.sum() > 0:
            correct_frr_snr = np.mean(pred_open_snr[correct_known_snr] == unknown_label)
        else:
            correct_frr_snr = 0.0
        results_by_snr[snr] = {'closed_known_acc': float(closed_snr), 'known_acc': float(result['known_acc']), 'unknown_acc': float(result['unknown_acc']), 'open_acc': float(result['open_acc']), 'known_reject_rate': float(known_reject_snr), 'correct_known_false_reject_rate': float(correct_frr_snr)}
    print('\n')
    print('=' * 100)
    print('Summary')
    print('=' * 100)
    print(f"{'SNR':>6} | {'ClosedKnown':>11} | {'KnownAcc':>9} | {'UnknownAcc':>10} | {'OpenAcc':>8} | {'KnownReject':>11} | {'CorrectKnownFRR':>15}")
    print('-' * 100)
    for snr in snr_list:
        result = results_by_snr[int(snr)]
        print(f"{int(snr):>4}dB | {result['closed_known_acc']:>11.4f} | {result['known_acc']:>9.4f} | {result['unknown_acc']:>10.4f} | {result['open_acc']:>8.4f} | {result['known_reject_rate']:>11.4f} | {result['correct_known_false_reject_rate']:>15.4f}")
    mean_closed = np.mean([x['closed_known_acc'] for x in results_by_snr.values()])
    mean_known = np.mean([x['known_acc'] for x in results_by_snr.values()])
    mean_unknown = np.mean([x['unknown_acc'] for x in results_by_snr.values()])
    mean_open = np.mean([x['open_acc'] for x in results_by_snr.values()])
    mean_reject = np.mean([x['known_reject_rate'] for x in results_by_snr.values()])
    mean_frr = np.mean([x['correct_known_false_reject_rate'] for x in results_by_snr.values()])
    print('\n' + '=' * 60)
    print('Mean performance across all SNRs')
    print('=' * 60)
    print(f'MeanClosedKnownAcc: {mean_closed:.4f}')
    print(f'MeanKnownAcc: {mean_known:.4f}')
    print(f'MeanUnknownAcc: {mean_unknown:.4f}')
    print(f'MeanOpenAcc: {mean_open:.4f}')
    print(f'MeanKnownRejectRate: {mean_reject:.4f}')
    print(f'MeanCorrectKnownFalseRejectRate: {mean_frr:.4f}')
    return results_by_snr
import numpy as np
import torch

def fit_snr_classwise_thresholds(model, classifier, device, val_loaders_by_snr, num_known_classes=9, unknown_label=9, known_frr=0.05, min_samples=20, temperature=1.0, beta=0.05, gamma=0.05, LSTM=None):
    """Calibrate SNR- and class-specific thresholds on validation data. val_loaders_by_snr maps each SNR to (two-channel loader, ten-channel loader). Higher open_score indicates greater unknownness. known_frr=0.05 targets rejection of approximately 5% of known validation samples."""
    model.eval()
    classifier.eval()
    if LSTM is not None:
        LSTM.eval()
    quantile = 1.0 - known_frr
    cached_outputs = {}
    all_known_scores = []
    for (snr, (loader_2ch, loader_10ch)) in sorted(val_loaders_by_snr.items()):
        (labels, pred_known, open_score) = collect_dual_view_energy_outputs(model=model, val_dl=loader_2ch, val_dl_mask=loader_10ch, device=device, classifier=classifier, unknown_label=unknown_label, temperature=temperature, beta=beta, gamma=gamma, LSTM=LSTM)
        labels = np.asarray(labels)
        pred_known = np.asarray(pred_known)
        open_score = np.asarray(open_score)
        known_mask = labels != unknown_label
        cached_outputs[int(snr)] = {'labels': labels, 'pred_known': pred_known, 'open_score': open_score, 'known_mask': known_mask}
        all_known_scores.append(open_score[known_mask])
    global_known_scores = np.concatenate(all_known_scores)
    global_threshold = float(np.quantile(global_known_scores, quantile))
    thresholds_by_snr = {}
    for (snr, output) in cached_outputs.items():
        pred_known = output['pred_known']
        open_score = output['open_score']
        known_mask = output['known_mask']
        snr_known_scores = open_score[known_mask]
        if len(snr_known_scores) >= min_samples:
            snr_fallback = float(np.quantile(snr_known_scores, quantile))
        else:
            snr_fallback = global_threshold
        thresholds = np.empty(num_known_classes, dtype=np.float32)
        for class_id in range(num_known_classes):
            class_mask = known_mask & (pred_known == class_id)
            class_scores = open_score[class_mask]
            if len(class_scores) >= min_samples:
                thresholds[class_id] = np.quantile(class_scores, quantile)
            else:
                thresholds[class_id] = snr_fallback
        thresholds_by_snr[int(snr)] = thresholds
        print(f'SNR={snr:>3} dB | thresholds={np.round(thresholds, 4)}')
    return thresholds_by_snr

def get_thresholds_for_snr(thresholds_by_snr, snr):
    """Use the nearest available SNR when the requested SNR has no threshold."""
    snr = int(snr)
    if snr in thresholds_by_snr:
        return np.asarray(thresholds_by_snr[snr], dtype=np.float32)
    available_snrs = np.asarray(sorted(thresholds_by_snr.keys()))
    nearest_snr = int(available_snrs[np.argmin(np.abs(available_snrs - snr))])
    return np.asarray(thresholds_by_snr[nearest_snr], dtype=np.float32)

def Trainer(pre_model, pre_optimizer, pre_scheduler, train_dl, train_dl_mask, valid_dl, test_dl, device, experiment_log_dir):
    os.makedirs(os.path.join(experiment_log_dir, 'saved_models'), exist_ok=True)
    best_pre_ck = torch.load('./saved_models/best9.pt', map_location=device)
    print('Start fine-tuning with Class-wise Dual-view Energy Open-set Recognition ...')
    pre_model.load_state_dict(best_pre_ck['model_state_dict'])
    (ft_model, classifier, cls_opt, cls_sched, ft_opt, ft_sched, LSTM, LSTM_optimizer, LSTM_scheduler) = build_model(0.0005, device, chkpoint=best_pre_ck)
    best_open_acc = -1.0
    best_ck = None
    best_epoch = 0
    save_path = os.path.join(experiment_log_dir, 'saved_models', 'best_open_classwise_dual.pt')
    num_known_classes = 5
    unknown_label = 5
    beta = 0.02
    gamma = 0.02
    q_values = np.arange(0.83, 0.941, 0.005)
    for ep in range(1, 100 + 1):
        (train_loss, train_acc) = model_finetune_open_known(model=ft_model, train_dl=Open_Dataloader_train_2, train_dl_mask=Open_Dataloader_train_10, device=device, model_optimizer=ft_opt, model_scheduler=ft_sched, class_scheduler=cls_sched, classifier=classifier, classifier_optimizer=cls_opt, LSTM=None, LSTM_optimizer=None, LSTM_scheduler=None)
        val_result = search_best_classwise_threshold(model=ft_model, val_dl=Open_Dataloader_val_2, val_dl_mask=Open_Dataloader_val_10, device=device, classifier=classifier, num_known_classes=num_known_classes, unknown_label=unknown_label, q_values=q_values, temperature=1.0, beta=beta, gamma=gamma, LSTM=None, min_known_acc=0.55)
        known_acc = val_result['known_acc']
        unknown_acc = val_result['unknown_acc']
        open_acc = val_result['open_acc']
        best_thresholds = val_result['thresholds']
        best_q = val_result['best_q']
        closed_known_acc = val_result['closed_known_acc']
        print(f'[Finetune-ClasswiseDualEnergy] Ep {ep} TrainLoss {train_loss:.4f} train_acc {train_acc:.4f} KnownAcc {known_acc:.4f} UnknownAcc {unknown_acc:.4f} OpenAcc {open_acc:.4f} closed_known_acc {closed_known_acc:.4f} ')
        if open_acc > best_open_acc:
            best_open_acc = open_acc
            best_epoch = ep
            best_ck = {'encoder_state': copy.deepcopy(ft_model.state_dict()), 'classifier_state': copy.deepcopy(classifier.state_dict()), 'epoch': ep, 'thresholds': best_thresholds, 'best_q': best_q, 'beta': beta, 'gamma': gamma, 'known_acc': known_acc, 'unknown_acc': unknown_acc, 'open_acc': open_acc, 'method': 'classwise_dual_view_energy'}
            best_ck['LSTM_state'] = copy.deepcopy(LSTM.state_dict())
            torch.save(best_ck, save_path)
            print(f'Save best classwise-dual-energy model at epoch {ep}, KnownAcc={known_acc:.4f}, UnknownAcc={unknown_acc:.4f}, OpenAcc={open_acc:.4f}, BestQ={best_q:.2f}')
    print(f'[Best OpenAcc] {best_open_acc:.4f}')
    print(f'[Best Epoch] {best_epoch}')
    best_ck = torch.load('./saved_models/best_open_classwise_dual_energy3.pt', map_location=device)
    ft_model.load_state_dict(best_ck['encoder_state'])
    classifier.load_state_dict(best_ck['classifier_state'])
    if 'LSTM_state' in best_ck:
        if LSTM is None:
            print('Warning: checkpoint contains LSTM_state, but LSTM=None. LSTM parameters were not loaded.')
        else:
            LSTM.load_state_dict(best_ck['LSTM_state'])
    thresholds = best_ck['thresholds']
    if torch.is_tensor(thresholds):
        thresholds = thresholds.detach().cpu().numpy()
    thresholds = np.asarray(thresholds, dtype=np.float32)
    beta = float(best_ck.get('beta', 0.05))
    gamma = float(best_ck.get('gamma', 0.05))
    best_q = float(best_ck.get('best_q', 0.0))
    best_epoch = int(best_ck.get('epoch', -1))
    print(f'Load best model from epoch {best_epoch}')
    print(f'Use best q = {best_q:.2f}')
    print(f'Use beta = {beta:.4f}')
    print(f'Use gamma = {gamma:.4f}')
    print('Use class-wise thresholds:')
    for (class_index, threshold) in enumerate(thresholds):
        print(f'Class {class_index}: threshold={threshold:.4f}')
    thresholds_by_snr = {int(snr): np.asarray(values, dtype=np.float32) for (snr, values) in best_ck['thresholds_by_snr'].items()}
    snr_list = list(range(-20, 20, 2))
    snr_results = {}
    for snr in snr_list:
        snr_thresholds = get_thresholds_for_snr(thresholds_by_snr=thresholds_by_snr, snr=snr)
        snr_results = test_open_set_by_snr(model=ft_model, classifier=classifier, device=device, thresholds=snr_thresholds, dataset_root=configured_path('RADIO_SSL_TEST_ROOT', 'test'), snr_list=[snr], batch_size=128, unknown_label=9, temperature=1.0, beta=beta, gamma=gamma, LSTM=None, test_seed=2026, num_workers=0)
    metric_names = ['closed_known_acc', 'known_acc', 'unknown_acc', 'open_acc', 'known_reject_rate', 'correct_known_false_reject_rate']
    mean_results = {}
    for metric_name in metric_names:
        metric_values = [snr_result[metric_name] for snr_result in snr_results.values()]
        mean_results[metric_name] = float(np.mean(metric_values))
    print('\n' + '=' * 60)
    print('Mean performance across all SNRs')
    print('=' * 60)
    print(f"MeanClosedKnownAcc: {mean_results['closed_known_acc']:.4f}")
    print(f"MeanKnownAcc: {mean_results['known_acc']:.4f}")
    print(f"MeanUnknownAcc: {mean_results['unknown_acc']:.4f}")
    print(f"MeanOpenAcc: {mean_results['open_acc']:.4f}")
    print(f"MeanKnownRejectRate: {mean_results['known_reject_rate']:.4f}")
    print(f"MeanCorrectKnownFalseRejectRate: {mean_results['correct_known_false_reject_rate']:.4f}")
    print('=' * 60)
    return {'best_q': best_q, 'best_epoch': best_epoch, 'beta': beta, 'gamma': gamma, 'thresholds': thresholds, 'snr_results': snr_results, 'mean_closed_known_acc': mean_results['closed_known_acc'], 'mean_known_acc': mean_results['known_acc'], 'mean_unknown_acc': mean_results['unknown_acc'], 'mean_open_acc': mean_results['open_acc'], 'mean_known_reject_rate': mean_results['known_reject_rate'], 'mean_correct_known_false_reject_rate': mean_results['correct_known_false_reject_rate']}
import os
import copy
import numpy as np
import torch
import torch.nn as nn

def get_classifier_num_classes(classifier):
    """Read the output class count from the last Linear layer of the classifier."""
    linear_layers = [module for module in classifier.modules() if isinstance(module, nn.Linear)]
    if not linear_layers:
        raise ValueError('No nn.Linear layer was found in the classifier.')
    return linear_layers[-1].out_features

def get_thresholds_for_snr(thresholds_by_snr, snr):
    """Return class thresholds for an SNR, falling back to the nearest available SNR."""
    snr = int(snr)
    if snr in thresholds_by_snr:
        return np.asarray(thresholds_by_snr[snr], dtype=np.float32)
    available_snrs = np.asarray(list(thresholds_by_snr.keys()), dtype=np.int32)
    nearest_snr = int(available_snrs[np.argmin(np.abs(available_snrs - snr))])
    print(f'Warning: no threshold for SNR={snr} dB; using SNR={nearest_snr} dB thresholds.')
    return np.asarray(thresholds_by_snr[nearest_snr], dtype=np.float32)

def calibrate_thresholds_by_snr(model, classifier, device, val_loaders_by_snr, global_thresholds, q, num_known_classes, unknown_label, temperature=1.0, beta=0.02, gamma=0.02, LSTM=None, min_samples=20):
    """Calibrate thresholds for each SNR and predicted class using an independent validation set. val_loaders_by_snr maps SNR values to (two-channel loader, ten-channel loader). Higher open_score indicates greater unknownness."""
    model.eval()
    classifier.eval()
    if LSTM is not None:
        LSTM.eval()
    global_thresholds = np.asarray(global_thresholds, dtype=np.float32)
    if len(global_thresholds) != num_known_classes:
        raise ValueError(f'Global threshold count is {len(global_thresholds)}, but the number of known classes is {num_known_classes}。')
    thresholds_by_snr = {}
    for snr in sorted(val_loaders_by_snr.keys()):
        (val_dl, val_dl_mask) = val_loaders_by_snr[snr]
        (labels, pred_known, open_score) = collect_dual_view_energy_outputs(model=model, val_dl=val_dl, val_dl_mask=val_dl_mask, device=device, classifier=classifier, unknown_label=unknown_label, temperature=temperature, beta=beta, gamma=gamma, LSTM=LSTM)
        labels = np.asarray(labels, dtype=np.int64)
        pred_known = np.asarray(pred_known, dtype=np.int64)
        open_score = np.asarray(open_score, dtype=np.float32)
        known_mask = labels != unknown_label
        snr_thresholds = global_thresholds.copy()
        for class_index in range(num_known_classes):
            class_mask = known_mask & (pred_known == class_index)
            class_scores = open_score[class_mask]
            if len(class_scores) >= min_samples:
                snr_thresholds[class_index] = np.quantile(class_scores, q)
            else:
                print(f'Warning: SNR={snr:>3} dB, class={class_index}, only {len(class_scores)} validation samples; using the global class threshold.')
        thresholds_by_snr[int(snr)] = snr_thresholds.astype(np.float32)
        print(f'SNR={snr:>3} dB thresholds: {np.round(snr_thresholds, 4)}')
    return thresholds_by_snr
import numpy as np
from sklearn.metrics import roc_auc_score

def compute_auroc_oscr(labels, pred_known, open_score, unknown_label):
    """Compute AUROC for known/unknown separation and OSCR for joint known-class classification and unknown rejection. Higher open_score indicates greater unknownness."""
    labels = np.asarray(labels).reshape(-1)
    pred_known = np.asarray(pred_known).reshape(-1)
    open_score = np.asarray(open_score, dtype=np.float64).reshape(-1)
    if not len(labels) == len(pred_known) == len(open_score):
        raise ValueError('labels, pred_known and open_score must have equal lengths.')
    known_mask = labels != unknown_label
    unknown_mask = labels == unknown_label
    if known_mask.sum() == 0 or unknown_mask.sum() == 0:
        return (np.nan, np.nan)
    binary_unknown = unknown_mask.astype(np.int64)
    auroc = roc_auc_score(binary_unknown, open_score)
    correct_known = (known_mask & (pred_known == labels)).astype(np.float64)
    order = np.argsort(open_score, kind='mergesort')
    sorted_scores = open_score[order]
    cumulative_correct = np.cumsum(correct_known[order])
    cumulative_unknown = np.cumsum(unknown_mask[order].astype(np.float64))
    group_ends = np.r_[np.flatnonzero(np.diff(sorted_scores) != 0), len(sorted_scores) - 1]
    ccr = cumulative_correct[group_ends] / known_mask.sum()
    fpr = cumulative_unknown[group_ends] / unknown_mask.sum()
    ccr = np.r_[0.0, ccr]
    fpr = np.r_[0.0, fpr]
    if hasattr(np, 'trapezoid'):
        oscr = np.trapezoid(ccr, fpr)
    else:
        oscr = np.trapz(ccr, fpr)
    return (float(auroc), float(oscr))

def Trainer1(train_dl, train_dl_mask, valid_dl, valid_dl_mask, val_loaders_by_snr, device, experiment_log_dir, pretrained_path='./saved_models/best9.pt', test_dataset_root=None, epochs=100, lr=0.0005, beta=0.02, gamma=0.02, q_start=0.83, q_end=0.94, q_step=0.005, min_known_acc=0.9, batch_size=64, test_seed=2026, excel_path=None):
    """
    Train with the original validation/selection policy, evaluate the best
    checkpoint once, and export thresholded open-set metrics plus AUROC/OSCR
    to an Excel workbook. The original call remains valid; ``excel_path`` is
    optional.

    Required existing functions in the same module:
      build_model, model_finetune_open_known,
      collect_dual_view_energy_outputs, build_classwise_thresholds,
      evaluate_classwise_open_set, get_classifier_num_classes,
      calibrate_thresholds_by_snr, build_loaders_by_snr.
    """
    import copy
    import os
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score
    history = []

    def compute_auroc_oscr(labels, pred_known, open_score, unknown_label):
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        pred_known = np.asarray(pred_known, dtype=np.int64).reshape(-1)
        open_score = np.asarray(open_score, dtype=np.float64).reshape(-1)
        if not len(labels) == len(pred_known) == len(open_score):
            raise ValueError('labels, pred_known and open_score must have equal lengths.')
        known_mask = labels != unknown_label
        unknown_mask = labels == unknown_label
        if known_mask.sum() == 0 or unknown_mask.sum() == 0:
            return (float('nan'), float('nan'))
        auroc = roc_auc_score(unknown_mask.astype(np.int64), open_score)
        correct_known = (known_mask & (pred_known == labels)).astype(np.float64)
        order = np.argsort(open_score, kind='mergesort')
        sorted_scores = open_score[order]
        cumulative_correct = np.cumsum(correct_known[order])
        cumulative_unknown = np.cumsum(unknown_mask[order].astype(np.float64))
        group_ends = np.r_[np.flatnonzero(np.diff(sorted_scores) != 0), len(sorted_scores) - 1]
        ccr = cumulative_correct[group_ends] / known_mask.sum()
        fpr = cumulative_unknown[group_ends] / unknown_mask.sum()
        ccr = np.r_[0.0, ccr]
        fpr = np.r_[0.0, fpr]
        if hasattr(np, 'trapezoid'):
            oscr = np.trapezoid(ccr, fpr)
        else:
            oscr = np.trapz(ccr, fpr)
        return (float(auroc), float(oscr))

    def search_thresholds_from_outputs(labels, pred_known, open_score, num_known_classes, unknown_label, q_values, min_known_acc):
        """Run the original q search without performing inference again."""
        labels = np.asarray(labels, dtype=np.int64)
        pred_known = np.asarray(pred_known, dtype=np.int64)
        open_score = np.asarray(open_score, dtype=np.float64)
        known_mask = labels != unknown_label
        if known_mask.sum() == 0:
            raise ValueError('The validation set contains no known samples.')
        closed_known_acc = float(np.mean(pred_known[known_mask] == labels[known_mask]))
        best_result = None
        best_unknown_acc = -np.inf
        fallback_result = None
        fallback_known_acc = -np.inf
        for q in q_values:
            thresholds = build_classwise_thresholds(labels=labels, pred_known=pred_known, open_score=open_score, num_known_classes=num_known_classes, unknown_label=unknown_label, q=float(q), use_correct_known=True)
            result = evaluate_classwise_open_set(labels=labels, pred_known=pred_known, open_score=open_score, thresholds=thresholds, unknown_label=unknown_label)
            known_acc = float(result['known_acc'])
            unknown_acc = float(result['unknown_acc'])
            open_acc = float(result['open_acc'])
            h_score = 2.0 * known_acc * unknown_acc / (known_acc + unknown_acc + 1e-12)
            candidate = {'closed_known_acc': closed_known_acc, 'known_acc': known_acc, 'unknown_acc': unknown_acc, 'open_acc': open_acc, 'h_score': float(h_score), 'thresholds': np.asarray(thresholds, dtype=np.float32), 'best_q': float(q)}
            if known_acc > fallback_known_acc:
                fallback_known_acc = known_acc
                fallback_result = candidate
            if known_acc >= min_known_acc and unknown_acc > best_unknown_acc:
                best_unknown_acc = unknown_acc
                best_result = candidate
        if best_result is None:
            if fallback_result is None:
                raise RuntimeError('No valid threshold candidate was produced.')
            print(f'[Warning] No q satisfies KnownAcc >= {min_known_acc:.4f}; using the candidate with the highest KnownAcc.')
            best_result = fallback_result
        print(f"[Best Class-wise Threshold] q={best_result['best_q']:.3f}, KnownAcc={best_result['known_acc']:.4f}, UnknownAcc={best_result['unknown_acc']:.4f}, OpenAcc={best_result['open_acc']:.4f}, HScore={best_result['h_score']:.4f}, CAcc={best_result['closed_known_acc']:.4f}")
        return best_result

    def evaluate_from_predictions(labels, pred_known, pred_open, unknown_label):
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        pred_known = np.asarray(pred_known, dtype=np.int64).reshape(-1)
        pred_open = np.asarray(pred_open, dtype=np.int64).reshape(-1)
        known_mask = labels != unknown_label
        unknown_mask = ~known_mask
        correct_known_mask = known_mask & (pred_known == labels)

        def safe_mean(values):
            return float(np.mean(values)) if len(values) else float('nan')
        return {'closed_known_acc': safe_mean(pred_known[known_mask] == labels[known_mask]), 'known_acc': safe_mean(pred_open[known_mask] == labels[known_mask]), 'unknown_acc': safe_mean(pred_open[unknown_mask] == unknown_label), 'open_acc': safe_mean(pred_open == labels), 'known_reject_rate': safe_mean(pred_open[known_mask] == unknown_label), 'correct_known_false_reject_rate': safe_mean(pred_open[correct_known_mask] == unknown_label)}

    def get_snr_thresholds(thresholds_by_snr, snr, global_thresholds):
        for key in (snr, int(snr), str(int(snr)), f'{int(snr)}dB'):
            if key in thresholds_by_snr:
                return np.asarray(thresholds_by_snr[key], dtype=np.float32)
        return np.asarray(global_thresholds, dtype=np.float32)

    def collect_metrics_by_snr(loaders_by_snr, model, classifier, thresholds_by_snr, global_thresholds):
        """Run one inference pass per SNR and calculate every test metric."""
        if not loaders_by_snr:
            raise ValueError('loaders_by_snr is empty.')
        torch.manual_seed(test_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(test_seed)
            torch.cuda.manual_seed_all(test_seed)
        labels_all = []
        pred_known_all = []
        pred_open_all = []
        open_score_all = []
        by_snr = {}
        for snr in sorted(loaders_by_snr.keys()):
            (loader_2ch, loader_10ch) = loaders_by_snr[snr]
            (labels, pred_known, open_score) = collect_dual_view_energy_outputs(model=model, val_dl=loader_2ch, val_dl_mask=loader_10ch, device=device, classifier=classifier, unknown_label=unknown_label, temperature=1.0, beta=beta, gamma=gamma, LSTM=active_lstm)
            (auroc, oscr) = compute_auroc_oscr(labels=labels, pred_known=pred_known, open_score=open_score, unknown_label=unknown_label)
            snr_thresholds = get_snr_thresholds(thresholds_by_snr=thresholds_by_snr, snr=snr, global_thresholds=global_thresholds)
            if pred_known.min() < 0 or pred_known.max() >= len(snr_thresholds):
                raise ValueError(f'SNR={snr}: predicted class is outside threshold range.')
            pred_open = np.asarray(pred_known, dtype=np.int64).copy()
            sample_thresholds = snr_thresholds[pred_known]
            pred_open[np.asarray(open_score) > sample_thresholds] = unknown_label
            snr_metrics = evaluate_from_predictions(labels=labels, pred_known=pred_known, pred_open=pred_open, unknown_label=unknown_label)
            snr_metrics.update({'auroc': auroc, 'oscr': oscr, 'n_samples': int(len(labels))})
            by_snr[int(snr)] = snr_metrics
            labels_all.append(np.asarray(labels))
            pred_known_all.append(np.asarray(pred_known))
            pred_open_all.append(pred_open)
            open_score_all.append(np.asarray(open_score))
            print(f"SNR={int(snr):>3} dB | ClosedKnown={snr_metrics['closed_known_acc']:.4f} | Known={snr_metrics['known_acc']:.4f} | Unknown={snr_metrics['unknown_acc']:.4f} | Open={snr_metrics['open_acc']:.4f} | AUROC={auroc:.4f} | OSCR={oscr:.4f}")
        labels_all = np.concatenate(labels_all)
        pred_known_all = np.concatenate(pred_known_all)
        pred_open_all = np.concatenate(pred_open_all)
        open_score_all = np.concatenate(open_score_all)
        (overall_auroc, overall_oscr) = compute_auroc_oscr(labels=labels_all, pred_known=pred_known_all, open_score=open_score_all, unknown_label=unknown_label)
        overall = evaluate_from_predictions(labels=labels_all, pred_known=pred_known_all, pred_open=pred_open_all, unknown_label=unknown_label)
        overall.update({'auroc': overall_auroc, 'oscr': overall_oscr, 'n_samples': int(len(labels_all))})
        return {'overall': overall, 'by_snr': by_snr}

    def export_metrics_to_excel(path, metrics, history, checkpoint):
        try:
            from openpyxl import Workbook
            from openpyxl.styles import Alignment, Font, PatternFill
            from openpyxl.utils import get_column_letter
        except ImportError as exc:
            raise ImportError('Excel export requires openpyxl: pip install openpyxl') from exc
        path = os.path.abspath(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        workbook = Workbook()
        workbook.remove(workbook.active)
        header_fill = PatternFill('solid', fgColor='1F4E78')
        header_font = Font(color='FFFFFF', bold=True)
        percent_keys = {'closed_known_acc', 'known_acc', 'unknown_acc', 'open_acc', 'known_reject_rate', 'correct_known_false_reject_rate', 'auroc', 'oscr'}

        def style_sheet(sheet, widths=None):
            sheet.freeze_panes = 'A2'
            sheet.auto_filter.ref = sheet.dimensions
            for cell in sheet[1]:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal='center')
            for column_cells in sheet.columns:
                index = column_cells[0].column
                width = max((len(str(cell.value or '')) for cell in column_cells))
                sheet.column_dimensions[get_column_letter(index)].width = min(max(width + 2, 12), 30)
            if widths:
                for (column, width) in widths.items():
                    sheet.column_dimensions[column].width = width
        metric_order = ['closed_known_acc', 'known_acc', 'unknown_acc', 'open_acc', 'known_reject_rate', 'correct_known_false_reject_rate', 'auroc', 'oscr', 'n_samples']
        display_names = {'closed_known_acc': 'ClosedKnownAcc', 'known_acc': 'KnownAcc', 'unknown_acc': 'UnknownAcc', 'open_acc': 'OpenAcc', 'known_reject_rate': 'KnownRejectRate', 'correct_known_false_reject_rate': 'CorrectKnownFRR', 'auroc': 'AUROC', 'oscr': 'OSCR', 'n_samples': 'N'}
        overall_sheet = workbook.create_sheet('Test_Overall')
        overall_sheet.append(['Metric', 'Value'])
        for key in metric_order:
            overall_sheet.append([display_names[key], metrics['overall'][key]])
            if key in percent_keys:
                overall_sheet.cell(overall_sheet.max_row, 2).number_format = '0.00%'
        style_sheet(overall_sheet, {'A': 28, 'B': 16})
        snr_sheet = workbook.create_sheet('Test_by_SNR')
        snr_sheet.append(['SNR (dB)'] + [display_names[key] for key in metric_order])
        for (snr, values) in sorted(metrics['by_snr'].items()):
            snr_sheet.append([snr] + [values[key] for key in metric_order])
            for (column, key) in enumerate(metric_order, start=2):
                if key in percent_keys:
                    snr_sheet.cell(snr_sheet.max_row, column).number_format = '0.00%'
        style_sheet(snr_sheet)
        history_sheet = workbook.create_sheet('Training_History')
        history_headers = ['Epoch', 'TrainLoss', 'TrainAcc', 'ClosedKnownAcc', 'KnownAcc', 'UnknownAcc', 'OpenAcc', 'AUROC', 'OSCR', 'q']
        history_sheet.append(history_headers)
        for item in history:
            history_sheet.append([item[key] for key in ('epoch', 'train_loss', 'train_acc', 'closed_known_acc', 'known_acc', 'unknown_acc', 'open_acc', 'auroc', 'oscr', 'q')])
            for column in range(3, 10):
                history_sheet.cell(history_sheet.max_row, column).number_format = '0.00%'
        style_sheet(history_sheet)
        threshold_sheet = workbook.create_sheet('Thresholds')
        threshold_sheet.append(['SNR (dB)'] + [f'Class_{i}' for i in range(num_known_classes)])
        for (snr, values) in sorted(thresholds_by_snr.items(), key=lambda x: int(x[0])):
            threshold_sheet.append([int(snr)] + list(np.asarray(values, dtype=float)))
        style_sheet(threshold_sheet)
        metadata_sheet = workbook.create_sheet('Metadata')
        metadata_sheet.append(['Field', 'Value'])
        for (key, value) in (('Best epoch', checkpoint['epoch']), ('Best q', checkpoint['best_q']), ('Known classes', num_known_classes), ('Unknown label', unknown_label), ('beta', checkpoint['beta']), ('gamma', checkpoint['gamma']), ('Checkpoint', save_path)):
            metadata_sheet.append([key, value])
        style_sheet(metadata_sheet, {'A': 22, 'B': 70})
        workbook.save(path)
        return path
    save_dir = os.path.join(experiment_log_dir, 'saved_models')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'best_open_classwise_dual_ML100.pt')
    pretrained_ck = torch.load(pretrained_path, map_location=device)
    (ft_model, classifier, cls_opt, cls_sched, ft_opt, ft_sched, _unused_lstm, _unused_lstm_optimizer, _unused_lstm_scheduler) = build_model(lr, device, chkpoint=pretrained_ck)
    active_lstm = None
    num_known_classes = get_classifier_num_classes(classifier)
    unknown_label = num_known_classes
    print(f'Known classes: {num_known_classes}')
    print(f'Unknown label: {unknown_label}')
    q_values = np.arange(q_start, q_end + q_step / 2, q_step)
    best_open_acc = -np.inf
    best_epoch = -1
    for epoch in range(1, epochs + 1):
        (train_loss, train_acc) = model_finetune_open_known(model=ft_model, train_dl=train_dl, train_dl_mask=train_dl_mask, device=device, model_optimizer=ft_opt, model_scheduler=ft_sched, class_scheduler=cls_sched, classifier=classifier, classifier_optimizer=cls_opt, LSTM=active_lstm, LSTM_optimizer=None, LSTM_scheduler=None)
        (val_labels, val_pred_known, val_open_score) = collect_dual_view_energy_outputs(model=ft_model, val_dl=valid_dl, val_dl_mask=valid_dl_mask, device=device, classifier=classifier, unknown_label=unknown_label, temperature=1.0, beta=beta, gamma=gamma, LSTM=active_lstm)
        val_result = search_thresholds_from_outputs(labels=val_labels, pred_known=val_pred_known, open_score=val_open_score, num_known_classes=num_known_classes, unknown_label=unknown_label, q_values=q_values, min_known_acc=min_known_acc)
        (val_auroc, val_oscr) = compute_auroc_oscr(labels=val_labels, pred_known=val_pred_known, open_score=val_open_score, unknown_label=unknown_label)
        print(f"[Epoch {epoch:03d}] Loss={float(train_loss):.4f} TrainAcc={float(train_acc):.4f} ClosedKnown={val_result['closed_known_acc']:.4f} Known={val_result['known_acc']:.4f} Unknown={val_result['unknown_acc']:.4f} Open={val_result['open_acc']:.4f} AUROC={val_auroc:.4f} OSCR={val_oscr:.4f} q={val_result['best_q']:.3f}")
        history.append({'epoch': int(epoch), 'train_loss': float(train_loss), 'train_acc': float(train_acc), 'closed_known_acc': float(val_result['closed_known_acc']), 'known_acc': float(val_result['known_acc']), 'unknown_acc': float(val_result['unknown_acc']), 'open_acc': float(val_result['open_acc']), 'auroc': float(val_auroc), 'oscr': float(val_oscr), 'q': float(val_result['best_q'])})
        if val_result['open_acc'] > best_open_acc:
            best_open_acc = val_result['open_acc']
            best_epoch = epoch
            checkpoint = {'encoder_state': copy.deepcopy(ft_model.state_dict()), 'classifier_state': copy.deepcopy(classifier.state_dict()), 'epoch': epoch, 'thresholds': np.asarray(val_result['thresholds'], dtype=np.float32), 'best_q': float(val_result['best_q']), 'beta': float(beta), 'gamma': float(gamma), 'num_known_classes': num_known_classes, 'unknown_label': unknown_label, 'closed_known_acc': float(val_result['closed_known_acc']), 'known_acc': float(val_result['known_acc']), 'unknown_acc': float(val_result['unknown_acc']), 'open_acc': float(val_result['open_acc']), 'auroc': float(val_auroc), 'oscr': float(val_oscr), 'method': 'classwise_dual_view_energy_snr_conditioned'}
            torch.save(checkpoint, save_path)
            print(f'Saved best model: epoch={epoch}, OpenAcc={best_open_acc:.4f}, AUROC={val_auroc:.4f}, OSCR={val_oscr:.4f}')
    if best_epoch < 0:
        raise RuntimeError('No valid checkpoint was saved; check min_known_acc.')
    best_ck = torch.load(save_path, map_location=device)
    ft_model.load_state_dict(best_ck['encoder_state'])
    classifier.load_state_dict(best_ck['classifier_state'])
    thresholds_by_snr = calibrate_thresholds_by_snr(model=ft_model, classifier=classifier, device=device, val_loaders_by_snr=val_loaders_by_snr, global_thresholds=best_ck['thresholds'], q=best_ck['best_q'], num_known_classes=num_known_classes, unknown_label=unknown_label, temperature=1.0, beta=best_ck['beta'], gamma=best_ck['gamma'], LSTM=active_lstm, min_samples=20)
    best_ck['thresholds_by_snr'] = {int(snr): np.asarray(values).tolist() for (snr, values) in thresholds_by_snr.items()}
    torch.save(best_ck, save_path)
    print(f'Best epoch: {best_epoch}')
    print(f'Best validation OpenAcc: {best_open_acc:.4f}')
    print(f'Checkpoint: {save_path}')
    if test_dataset_root is None:
        metric_loaders = val_loaders_by_snr
        metric_split = 'validation'
    else:
        metric_loaders = build_loaders_by_snr(dataset_root=test_dataset_root, split='test', batch_size=batch_size)
        metric_split = 'test'
    metrics = collect_metrics_by_snr(loaders_by_snr=metric_loaders, model=ft_model, classifier=classifier, thresholds_by_snr=thresholds_by_snr, global_thresholds=best_ck['thresholds'])
    print('\n' + '=' * 60)
    print(f'{metric_split.capitalize()} threshold-free metrics')
    print('=' * 60)
    overall = metrics['overall']
    print(f"ClosedKnownAcc: {overall['closed_known_acc']:.4f}")
    print(f"KnownAcc:       {overall['known_acc']:.4f}")
    print(f"UnknownAcc:     {overall['unknown_acc']:.4f}")
    print(f"OpenAcc:        {overall['open_acc']:.4f}")
    print(f"KnownReject:    {overall['known_reject_rate']:.4f}")
    print(f"CorrectKnownFRR:{overall['correct_known_false_reject_rate']:.4f}")
    print(f"Overall AUROC:  {overall['auroc']:.4f}")
    print(f"Overall OSCR:   {overall['oscr']:.4f}")
    if excel_path is None:
        excel_path = os.path.join(experiment_log_dir, 'metrics', 'Trainer1_open_set_ML100.xlsx')
    excel_path = export_metrics_to_excel(path=excel_path, metrics=metrics, history=history, checkpoint=best_ck)
    print(f'Excel metrics saved to: {excel_path}')
    return {'overall': metrics['overall'], 'by_snr': metrics['by_snr'], 'history': history, 'thresholds': np.asarray(best_ck['thresholds']), 'thresholds_by_snr': thresholds_by_snr, 'best_epoch': int(best_ck['epoch']), 'best_q': float(best_ck['best_q']), 'checkpoint_path': save_path, 'excel_path': excel_path}
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch

def build_loaders_by_snr(dataset_root, split='test', snr_list=None, batch_size=128, num_workers=0):
    if snr_list is None:
        snr_list = [int(value) for value in os.environ.get('RADIO_SSL_SNRS', '-4,0,4').split(',')]
    val_loaders_by_snr = {}
    for snr in snr_list:
        snr_dir = os.path.join(dataset_root, f'snr_{snr}dB')
        raw_path = os.path.join(snr_dir, f'raw_{split}_{snr}dB_data.npy')
        sequence_path = os.path.join(snr_dir, f'sequence_{split}_{snr}dB_data.npy')
        label_path = os.path.join(snr_dir, f'{split}_{snr}dB_label.npy')
        for path in (raw_path, sequence_path, label_path):
            if not os.path.exists(path):
                raise FileNotFoundError(path)
        raw_data = torch.from_numpy(np.load(raw_path)).float()
        sequence_data = torch.from_numpy(np.load(sequence_path)).float()
        labels = torch.from_numpy(np.load(label_path))
        if labels.ndim > 1:
            labels = labels.argmax(dim=1)
        labels = labels.long()
        if len(raw_data) != len(sequence_data):
            raise ValueError(f'SNR={snr}: two-channel and ten-channel sample counts differ')
        if len(raw_data) != len(labels):
            raise ValueError(f'SNR={snr}: data and label counts differ')
        raw_loader = DataLoader(TensorDataset(raw_data, labels), batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
        sequence_loader = DataLoader(TensorDataset(sequence_data, labels), batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
        val_loaders_by_snr[int(snr)] = (raw_loader, sequence_loader)
    return val_loaders_by_snr
if __name__ == '__main__':
    loss_model = CenterLoss(11, 128).to(device)
    model_z = Encoder1()
    model_LSTM = EncoderD()
    encoder.to(device)
    fullatten.to(device)
    proatten.to(device)
    selfatten.to(device)
    crossatten.to(device)
    model_z.to(device)
    model_s.to(device)
    model_s1.to(device)
    decoder.to(device)
    data_mask = np.load('./data/all/X_all.npy')
    data_mask_target = np.load('./data/all/Y_all.npy')
    data_train = np.load('./data/X_all.npy')
    data_train = torch.tensor(data_train)
    data_train = data_train.to(torch.float32)
    target_train = np.load('./data/all/Y_all.npy')
    data_val_mask = X_val
    data_val = np.load('./data/val/100.npy')
    data_val = torch.tensor(data_val)
    data_val = data_val.to(torch.float32)
    target_val = np.load('./data/val/target_val.npy')
    data_test = np.load('./data/val.npy')
    data_test = torch.tensor(data_test)
    data_test = data_test.to(torch.float32)
    target_test = Y_test
    data_fintune = np.load('./data/X_100.npy')
    target_fintune = np.load('./data/fintune/Y_samples100.npy')
    data_f = np.load('./data/fintune/X_samples100.npy')
    data = np.load('./data/val/val.npy')
    target = np.load('./data/val/target_val.npy')
    Data_mask = Data(data_mask, data_mask_target)
    Data_train = Data(data_train, target_train)
    Data_val = Data(data_val, target_val)
    Data_test = Data(data_test, target_test)
    dataset_val_mask = Data(data_val_mask, target_val)
    data_fintune = Data(data_fintune, target_fintune)
    dataset_f = Data(data_f, target_fintune)
    dataset = Data(data, target)
    Data_loader_mask = DataLoader(Data_mask, 64, shuffle=True)
    Data_loader_train = DataLoader(Data_train, 64, shuffle=True)
    Data_loader_val = DataLoader(Data_val, 64)
    Data_loader_test = DataLoader(Data_test, 64)
    Data_loader_val_mask = DataLoader(dataset_val_mask, 64)
    Data_loader_fintune = DataLoader(data_fintune, 64)
    Data_loader_f = DataLoader(dataset_f, 64)
    dataloader = DataLoader(dataset, 64)
    Open_train_2 = np.load('./open_set_train_test/QPSK/train_known_raw_2ch_data.npy')
    Open_train_10 = np.load('./open_set_train_test/QPSK/train_known_sequence_10ch_data.npy')
    Open_train_label = np.load('./open_set_train_test/QPSK/train_known_label.npy')
    Open_val_2 = np.load('./open_set_train_test/QPSK/validation/val_open_raw_2ch_data.npy')
    Open_val_10 = np.load('./open_set_train_test/QPSK/validation/val_open_sequence_10ch_data.npy')
    Open_val_label = np.load('./open_set_train_test/QPSK/validation/val_open_label.npy')
    Open_Data_train_2 = Data(Open_train_2, Open_train_label)
    Open_Data_train_10 = Data(Open_train_10, Open_train_label)
    Open_Data_val_2 = Data(Open_val_2, Open_val_label)
    Open_Data_val_10 = Data(Open_val_10, Open_val_label)
    Open_Dataloader_train_2 = DataLoader(Open_Data_train_2, 16)
    Open_Dataloader_train_10 = DataLoader(Open_Data_train_10, 16)
    Open_Dataloader_val_2 = DataLoader(Open_Data_val_2, 16)
    Open_Dataloader_val_10 = DataLoader(Open_Data_val_10, 16)
    a = pre_model(model_z, model_LSTM, model_r, model_s, model_s1)
    a.to(device)
    optimizer = optim.AdamW(a.parameters(), lr=0.05, weight_decay=0.001)
    loss_optimizer = optim.SGD(loss_model.parameters(), lr=0.1)
    model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=150)
    experiment_log_dir = './'
    val_loaders_by_snr = build_loaders_by_snr(dataset_root=configured_path('RADIO_SSL_OPEN_VAL_ROOT', 'open_set_data/validation/by_snr'), split='val', batch_size=16)
    result = Trainer1(train_dl=Open_Dataloader_train_2, train_dl_mask=Open_Dataloader_train_10, valid_dl=Open_Dataloader_val_2, valid_dl_mask=Open_Dataloader_val_10, val_loaders_by_snr=val_loaders_by_snr, device=device, experiment_log_dir=experiment_log_dir, pretrained_path='./saved_models/ML_best.pt', test_dataset_root=configured_path('RADIO_SSL_OPEN_TEST_ROOT', 'open_set_data/test/by_snr'), epochs=int(os.environ.get('RADIO_SSL_EPOCHS', '100')))
