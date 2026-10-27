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
from loss import AutomaticWeightedLoss, ContrastiveWeight, AggregationRebuild, CrossDomainContrastiveWeight, CenterLoss
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
from paint import plot_confusion_matrix_18db_loader
from radar import load_data_radcom_npy
warnings.filterwarnings('ignore')
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

def model_validate(model, val_dl, val_dl_mask, device, classifier=None, LSTM=None):
    model.eval()
    classifier.eval()
    if LSTM is not None:
        LSTM.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        with tqdm(total=len(val_dl)) as pb:
            for ((data, labels), (data_mask, labels_mask)) in zip(val_dl, val_dl_mask):
                data = data.float().to(device)
                if labels.ndim > 1:
                    labels = labels.argmax(dim=1)
                labels = labels.long().to(device)
                data_mask = data_mask.float().to(device)
                if labels_mask.ndim > 1:
                    labels_mask = labels_mask.argmax(dim=1)
                labels_mask = labels_mask.long().to(device)
                h = model(data, data_mask)
                preds = classifier(h)
                preds = preds.argmax(dim=1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                pb.update(1)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    F1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return (acc, precision, recall, F1)

def model_finetune(model, loss_model, val_dl, val_dl_mask, device, class_scheduler, model_optimizer, model_scheduler, LSTM_scheduler, LSTM_optimizer, classifier=None, classifier_optimizer=None, LSTM=None, center_optimizer=None):
    model.eval()
    classifier.train()
    loss_model.train()
    loss_center = []
    total_loss = []
    total_acc = []
    total_auc = []
    total_prc = []
    criterion = nn.CrossEntropyLoss()
    outs = np.array([])
    trgs = np.array([])
    with tqdm(total=len(val_dl)) as pb:
        for ((data, labels), (data_mask, labels_mask)) in zip(val_dl, val_dl_mask):
            model_optimizer.zero_grad()
            classifier_optimizer.zero_grad()
            (data, labels, data_mask, labels_mask) = (data.float().to(device), labels.to(device), data_mask.float().to(device), labels_mask.to(device))
            (data, labels) = flip_row(data, labels)
            (data_mask, labels_mask) = flip_row(data_mask, labels_mask)
            (data_mask, labels_mask) = (data_mask.float().to(device), labels_mask.to(device))
            h = model(data_mask, data)
            fea_concat = h
            predictions = classifier(fea_concat)
            fea_concat_flat = fea_concat.reshape(fea_concat.shape[0], -1)
            if labels.ndim > 1:
                labels = labels.argmax(dim=1)
            labels = labels.long().to(device)
            loss = criterion(predictions, labels)
            acc_bs = (labels == predictions.argmax(dim=1)).float().mean()
            labels = labels.long()
            onehot_label = F.one_hot(labels)
            pred_numpy = predictions.detach().cpu().numpy()
            try:
                auc_bs = roc_auc_score(onehot_label.detach().cpu().numpy(), pred_numpy, average='macro', multi_class='ovr')
            except:
                auc_bs = 0.0
            try:
                prc_bs = average_precision_score(onehot_label.detach().cpu().numpy(), pred_numpy)
            except:
                prc_bs = 0.0
            total_acc.append(acc_bs)
            if auc_bs != 0:
                total_auc.append(auc_bs)
            if prc_bs != 0:
                total_prc.append(prc_bs)
            total_loss.append(loss.item())
            loss.backward()
            classifier_optimizer.step()
            pred = predictions.max(1, keepdim=True)[1]
            outs = np.append(outs, pred.cpu().numpy())
            trgs = np.append(trgs, labels.data.cpu().numpy())
            pb.update(1)
    labels_numpy = labels.detach().cpu().numpy()
    pred_numpy = np.argmax(pred_numpy, axis=1)
    total_loss = torch.tensor(total_loss).mean()
    total_acc = torch.tensor(total_acc).mean()
    total_auc = torch.tensor(total_auc).mean()
    total_prc = torch.tensor(total_prc).mean()
    class_scheduler.step()
    return (total_loss, total_acc, total_auc, total_prc, fea_concat_flat, trgs)

def model_finetune_fast(model, val_dl, val_dl_mask, device, class_scheduler, classifier, classifier_optimizer):
    model.eval()
    classifier.train()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    sample_count = 0
    epoch_logits = []
    epoch_targets = []
    last_features = None
    device_type = torch.device(device).type
    use_amp = device_type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    with tqdm(total=min(len(val_dl), len(val_dl_mask))) as progress:
        for ((data, labels), (data_mask, labels_mask)) in zip(val_dl, val_dl_mask):
            classifier_optimizer.zero_grad(set_to_none=True)
            data = torch.as_tensor(data, dtype=torch.float32).to(device, non_blocking=True)
            data_mask = torch.as_tensor(data_mask, dtype=torch.float32).to(device, non_blocking=True)
            labels = torch.as_tensor(labels).to(device, non_blocking=True)
            if labels.ndim > 1:
                labels = labels.argmax(dim=1)
            labels = labels.long()
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.float16 if use_amp else torch.bfloat16, enabled=use_amp):
                    features = model(data_mask, data)
            with torch.autocast(device_type=device_type, dtype=torch.float16 if use_amp else torch.bfloat16, enabled=use_amp):
                predictions = classifier(features)
                loss = criterion(predictions, labels)
            scaler.scale(loss).backward()
            scaler.step(classifier_optimizer)
            scaler.update()
            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            correct += (predictions.argmax(dim=1) == labels).sum().item()
            sample_count += batch_size
            epoch_logits.append(predictions.detach().float().cpu())
            epoch_targets.append(labels.detach().cpu())
            last_features = features.detach()
            progress.update(1)
    epoch_logits = torch.cat(epoch_logits, dim=0)
    epoch_targets = torch.cat(epoch_targets, dim=0)
    probabilities = torch.softmax(epoch_logits, dim=1).numpy()
    target_numpy = epoch_targets.numpy()
    num_classes = probabilities.shape[1]
    target_onehot = np.eye(num_classes)[target_numpy]
    try:
        epoch_auc = roc_auc_score(target_onehot, probabilities, average='macro', multi_class='ovr')
    except ValueError:
        epoch_auc = 0.0
    try:
        epoch_auprc = average_precision_score(target_onehot, probabilities, average='macro')
    except ValueError:
        epoch_auprc = 0.0
    epoch_loss = running_loss / sample_count
    epoch_accuracy = correct / sample_count
    class_scheduler.step()
    return (epoch_loss, epoch_accuracy, epoch_auc, epoch_auprc, last_features.reshape(last_features.size(0), -1), target_numpy)

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

    def __init__(self, num_classes=11):
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
        self.LSTM = nn.LSTM(input_size=32, hidden_size=32, num_layers=2, batch_first=True, bidirectional=True)

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
    for param in ft_model.parameters():
        param.requires_grad = False
    ft_model_dict = ft_model.state_dict()
    LSTM = TemporalConvNet(num_inputs=128, num_channels=[128, 128], kernel_size=3, dropout=0.1).to(device)
    classifier = target_classifier(num_classes=int(os.environ.get('RADIO_SSL_NUM_KNOWN_CLASSES', '11'))).to(device)
    model_optimizer = torch.optim.AdamW(ft_model.parameters(), lr=0.003, weight_decay=0.05)
    classifier_optimizer = torch.optim.AdamW(classifier.parameters(), lr=0.001, weight_decay=0.0001)
    LSTM_optimizer = torch.optim.AdamW(LSTM.parameters(), lr=0.0001, weight_decay=0.05)
    model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=model_optimizer, T_max=50)
    LSTM_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=LSTM_optimizer, T_max=50)
    class_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=classifier_optimizer, T_max=50)
    return (ft_model, classifier, classifier_optimizer, class_scheduler, model_optimizer, model_scheduler, LSTM, LSTM_optimizer, LSTM_scheduler)
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from pathlib import Path
import numpy as np
import torch

def get_snr_name(snr):
    """Convert an SNR value to a data-directory name: -20 -> m20dB, 0 -> p0dB, 18 -> p18dB."""
    snr = int(snr)
    if snr < 0:
        return f'm{abs(snr)}dB'
    return f'p{snr}dB'

def extract_model_output(output):
    """Extract model output from a tensor, tuple, list or dictionary."""
    if isinstance(output, (tuple, list)):
        return output[0]
    if isinstance(output, dict):
        for key in ('logits', 'features', 'output', 'out'):
            if key in output:
                return output[key]
        raise KeyError('The model returned a dictionary without logits, features, output or out.')
    return output

def model_test_single_snr_acc(model, classifier, device, snr, dataset_root='./dataset/RML2016.10b/test_by_snr', batch_size=64, LSTM=None, num_workers=0):
    """Evaluate classification accuracy at one SNR. Model input order: model(sequence_data, raw_data)."""
    dataset_root = Path(dataset_root)
    snr = int(snr)
    snr_name = get_snr_name(snr)
    snr_dir = dataset_root / f'snr_{snr_name}'
    raw_data_path = snr_dir / f'raw_test_{snr_name}_data.npy'
    raw_label_path = snr_dir / f'raw_test_{snr_name}_label.npy'
    sequence_data_path = snr_dir / f'sequence_test_{snr_name}_data.npy'
    sequence_label_path = snr_dir / f'sequence_test_{snr_name}_label.npy'
    required_paths = [raw_data_path, raw_label_path, sequence_data_path]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f'Test file not found: {path}')
    raw_data = np.load(raw_data_path)
    sequence_data = np.load(sequence_data_path)
    labels = np.load(raw_label_path)
    if labels.ndim > 1:
        labels = np.argmax(labels, axis=1)
    labels = labels.astype(np.int64, copy=False)
    if len(raw_data) != len(sequence_data):
        raise ValueError(f'{snr} dB: raw and 10-channel sample counts differ: {len(raw_data)} != {len(sequence_data)}')
    if len(raw_data) != len(labels):
        raise ValueError(f'{snr} dB: data and label counts differ: {len(raw_data)} != {len(labels)}')
    if sequence_label_path.exists():
        sequence_labels = np.load(sequence_label_path)
        if sequence_labels.ndim > 1:
            sequence_labels = np.argmax(sequence_labels, axis=1)
        sequence_labels = sequence_labels.astype(np.int64, copy=False)
        if not np.array_equal(labels, sequence_labels):
            raise ValueError(f'{snr} dB: raw and 10-channel labels are not aligned.')
    raw_tensor = torch.from_numpy(raw_data.astype(np.float32, copy=False))
    sequence_tensor = torch.from_numpy(sequence_data.astype(np.float32, copy=False))
    label_tensor = torch.from_numpy(labels)
    test_dataset = TensorDataset(sequence_tensor, raw_tensor, label_tensor)
    pin_memory = isinstance(device, torch.device) and device.type == 'cuda'
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
    model.eval()
    classifier.eval()
    if LSTM is not None:
        LSTM.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc=f'Testing {snr} dB', total=len(test_loader))
        for (sequence_batch, raw_batch, label_batch) in progress_bar:
            sequence_batch = sequence_batch.to(device, dtype=torch.float32, non_blocking=True)
            raw_batch = raw_batch.to(device, dtype=torch.float32, non_blocking=True)
            label_batch = label_batch.to(device, dtype=torch.long, non_blocking=True)
            features = model(sequence_batch, raw_batch)
            features = extract_model_output(features)
            if LSTM is not None:
                features = LSTM(features)
                features = extract_model_output(features)
            logits = classifier(features)
            logits = extract_model_output(logits)
            if logits.ndim != 2:
                raise RuntimeError(f'Classifier output must have shape [B, num_classes]; received {tuple(logits.shape)}。')
            predictions = logits.argmax(dim=1)
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(label_batch.cpu().numpy())
    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    accuracy = accuracy_score(all_labels, all_predictions)
    return accuracy

def model_test_all_snr_acc(model, classifier, device, dataset_root='./dataset/RML2016.10b/test_by_snr', batch_size=64, LSTM=None, num_workers=0):
    """Read snrs.npy and evaluate all SNRs. Return per-SNR results and the arithmetic mean of their accuracies."""
    dataset_root = Path(dataset_root)
    snr_path = dataset_root / 'snrs.npy'
    if not snr_path.exists():
        raise FileNotFoundError(f'SNR configuration file not found: {snr_path}')
    snr_list = np.load(snr_path, allow_pickle=True)
    snr_list = sorted((int(snr) for snr in snr_list))
    results = {}
    for snr in snr_list:
        accuracy = model_test_single_snr_acc(model=model, classifier=classifier, device=device, snr=snr, dataset_root=dataset_root, batch_size=batch_size, LSTM=LSTM, num_workers=num_workers)
        results[snr] = accuracy
        print(f'SNR={snr:>3} dB | Acc={accuracy:.4f} | Acc={accuracy * 100:.2f}%')
    mean_accuracy = float(np.mean(list(results.values())))
    print('=' * 70)
    print(f'Mean accuracy over {len(results)} SNRs: {mean_accuracy:.4f} ({mean_accuracy * 100:.2f}%)')
    return (results, mean_accuracy)

def Trainer(pre_model, pre_optimizer, pre_scheduler, train_dl, train_dl_mask, valid_dl, test_dl, device, experiment_log_dir):
    os.makedirs(os.path.join(experiment_log_dir, 'saved_models'), exist_ok=True)
    best_pre_loss = float('inf')
    best_pre_ck = None
    for epoch in range(1, 100 + 1):
        (total_loss, loss_c1, loss_c2) = model_pretrain(pre_model, pre_optimizer, pre_scheduler, train_dl, train_dl_mask, device)
        print(f'[Pretrain] Ep {epoch} loss {total_loss:.4f} loss_c1 {loss_c1:.4f}  loss_c2 {loss_c2:.4f}')
        if total_loss < best_pre_loss:
            best_pre_loss = total_loss
            best_pre_ck = {'model_state_dict': pre_model.state_dict()}
            torch.save(best_pre_ck, os.path.join(experiment_log_dir, 'saved_models', 'ML_best.pt'))
    best_pre_ck = torch.load('./saved_models/best9.pt')
    (ft_model, classifier, cls_opt, cls_sched, ft_opt, ft_sched, LSTM, LSTM_optimizer, LSTM_scheduler) = build_model(0.0005, device, chkpoint=best_pre_ck)
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import to_rgba, to_rgb

def plot_tsne_18db_loader1(checkpoint_path, device, feature_dl, feature_dl_mask, build_model_func, pre_ck=None, save_path='tsne_18db.png', max_samples_per_class=500, use_lstm=False):
    """Visualize features using t-SNE and an existing loader containing 18 dB samples."""
    (ft_model, classifier, cls_opt, cls_sched, ft_opt, ft_sched, LSTM, LSTM_optimizer, LSTM_scheduler) = build_model_func(0.0005, device, chkpoint=pre_ck)
    best_ck = torch.load(checkpoint_path, map_location=device)
    ft_model.load_state_dict(best_ck['encoder_state'])
    classifier.load_state_dict(best_ck['classifier_state'])
    ft_model.eval()
    classifier.eval()
    if use_lstm and LSTM is not None:
        LSTM.eval()
    features = []
    labels_all = []
    with torch.no_grad():
        for ((data, labels), (data_mask, labels_mask)) in zip(feature_dl, feature_dl_mask):
            data = data.float().to(device)
            data_mask = data_mask.float().to(device)
            labels = labels.to(device)
            if labels.ndim > 1:
                labels = labels.argmax(dim=1)
            labels = labels.long()
            h = ft_model(data_mask, data)
            if use_lstm and LSTM is not None:
                h = LSTM(h)
            if h.ndim > 2:
                h = h.view(h.size(0), -1)
            features.append(h.cpu().numpy())
            labels_all.append(labels.cpu().numpy())
    features = np.concatenate(features, axis=0)
    labels_all = np.concatenate(labels_all, axis=0)
    print('Raw features shape:', features.shape)
    unique_labels = np.unique(labels_all)
    (unique_labels, counts) = np.unique(labels_all, return_counts=True)
    print('Unique labels:', unique_labels)
    print('Number of classes:', len(unique_labels))
    print('Samples per class:', dict(zip(unique_labels, counts)))
    idx_keep = []
    np.random.seed(42)
    for lab in unique_labels:
        idx_lab = np.where(labels_all == lab)[0]
        if len(idx_lab) > max_samples_per_class:
            idx_lab = np.random.choice(idx_lab, max_samples_per_class, replace=False)
        idx_keep.extend(idx_lab)
    features = features[idx_keep]
    labels_all = labels_all[idx_keep]
    print('After sampling features shape:', features.shape)
    features = StandardScaler().fit_transform(features)
    pca = PCA(n_components=min(50, features.shape[1]))
    features_pca = pca.fit_transform(features)
    tsne = TSNE(n_components=2, perplexity=50, learning_rate=300, n_iter=4000, random_state=42, init='pca')
    tsne_result = tsne.fit_transform(features_pca)
    tsne_result = tsne_result - tsne_result.mean(axis=0, keepdims=True)
    rml2016_mods = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
    plt.figure(figsize=(3.46, 2.99))
    ax = plt.gca()
    ax.set_facecolor('white')
    colors = ['#5AA2D1', '#F28E2B', '#55A868', '#E78AC3', '#8C6BB1', '#8FB9D9', '#F4E77D', '#FF9896', '#A6D96A', '#C7B6D9', '#76C5B5']

    def darken_color(color, factor=0.78):
        """Darken a color. Smaller factors produce darker colors; suggested range: 0.75 to 0.85."""
        rgb = to_rgb(color)
        return tuple((channel * factor for channel in rgb))
    for (i, lab) in enumerate(unique_labels):
        idx = labels_all == lab
        pts = tsne_result[idx]
        face_color = colors[i % len(colors)]
        edge_color = darken_color(face_color, factor=0.78)
        plt.scatter(pts[:, 0], pts[:, 1], s=34, facecolors='none', edgecolors=(1, 1, 1, 0.55), linewidths=0.65, rasterized=True)
        plt.scatter(pts[:, 0], pts[:, 1], s=20, facecolors=[to_rgba(face_color, 0.35)], edgecolors=[to_rgba(edge_color, 0.88)], linewidths=0.2, label=str(lab), rasterized=True)
    ax = plt.gca()
    ax.set_xlim(-110, 110)
    ax.set_ylim(-110, 110)
    ticks = [-100, -50, 0, 50, 100]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(axis='both', labelsize=7)
    from matplotlib.lines import Line2D
    from matplotlib.ticker import AutoMinorLocator
    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    plt.grid(True, which='major', color='#D0D0D0', linewidth=0.8, alpha=0.65)
    plt.grid(True, which='minor', color='#E5E5E5', linewidth=0.6, alpha=0.55)
    plt.tight_layout()
    plt.savefig(save_path, dpi=900, bbox_inches='tight')
    plt.show()
    print(f't-SNE saved to: {save_path}')
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
RML2016_MODS = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
COLORS = ['#4C9AD4', '#B5693C', '#2CA02C', '#F28E2B', '#9467BD', '#8C564B', '#E377C2', '#C5B0D5', '#D8D56A', '#9DBFE5', '#86D47E']

def plot_tsne_18db_loader(checkpoint_path, device, feature_dl, feature_dl_mask, build_model_func, pre_ck=None, save_path='tsne_18db.png', max_samples_per_class=200, use_lstm=False, show=False):
    """Extract features and create a compact, publication-style t-SNE panel."""
    (ft_model, classifier, cls_opt, cls_sched, ft_opt, ft_sched, LSTM, LSTM_optimizer, LSTM_scheduler) = build_model_func(0.0005, device, chkpoint=pre_ck)
    best_ck = torch.load(checkpoint_path, map_location=device)
    ft_model.load_state_dict(best_ck['encoder_state'])
    classifier.load_state_dict(best_ck['classifier_state'])
    ft_model.eval()
    classifier.eval()
    if use_lstm and LSTM is not None:
        LSTM.eval()
    features = []
    labels_all = []
    with torch.no_grad():
        for ((data, labels), (data_mask, labels_mask)) in zip(feature_dl, feature_dl_mask):
            data = data.float().to(device)
            data_mask = data_mask.float().to(device)
            labels = labels.to(device)
            if labels.ndim > 1:
                labels = labels.argmax(dim=1)
            if labels_mask.ndim > 1:
                labels_mask = labels_mask.argmax(dim=1)
            labels = labels.long()
            labels_mask = labels_mask.to(device).long()
            if not torch.equal(labels, labels_mask):
                raise ValueError('feature_dl and feature_dl_mask are not sample-aligned.')
            h = ft_model(data_mask, data)
            if use_lstm and LSTM is not None:
                h = LSTM(h)
            if h.ndim > 2:
                h = h.reshape(h.size(0), -1)
            features.append(h.cpu().numpy())
            labels_all.append(labels.cpu().numpy())
    features = np.concatenate(features, axis=0)
    labels_all = np.concatenate(labels_all, axis=0)
    unique_labels = np.unique(labels_all)
    rng = np.random.default_rng(42)
    idx_keep = []
    for label in unique_labels:
        idx_label = np.flatnonzero(labels_all == label)
        if idx_label.size > max_samples_per_class:
            idx_label = rng.choice(idx_label, max_samples_per_class, replace=False)
        idx_keep.extend(idx_label.tolist())
    idx_keep = np.asarray(idx_keep, dtype=int)
    features = features[idx_keep]
    labels_all = labels_all[idx_keep]
    features_scaled = StandardScaler().fit_transform(features)
    n_components = min(50, features_scaled.shape[1], features_scaled.shape[0] - 1)
    features_pca = PCA(n_components=n_components, random_state=42).fit_transform(features_scaled)
    tsne_result = TSNE(n_components=2, perplexity=50, learning_rate=300, n_iter=4000, random_state=42, init='pca').fit_transform(features_pca)
    tsne_result = tsne_result - tsne_result.mean(axis=0, keepdims=True)
    mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 'font.size': 7, 'axes.linewidth': 0.75, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    (fig, ax) = plt.subplots(figsize=(89 / 25.4, 76 / 25.4))
    fig.subplots_adjust(left=0.12, right=0.98, top=0.97, bottom=0.25)
    ax.set_facecolor('white')
    legend_handles = []
    for (index, label) in enumerate(unique_labels):
        selected = labels_all == label
        points = tsne_result[selected]
        color = COLORS[index % len(COLORS)]
        ax.scatter(points[:, 0], points[:, 1], s=7, color=color, alpha=0.62, edgecolors='none', rasterized=True, zorder=2)
        center = np.median(points, axis=0)
        ax.scatter(center[0], center[1], s=30, color=color, marker='X', edgecolors='black', linewidths=0.65, zorder=4)
        class_index = int(label)
        label_name = RML2016_MODS[class_index] if 0 <= class_index < len(RML2016_MODS) else f'Class {class_index}'
        legend_handles.append(Line2D([0], [0], marker='o', linestyle='none', markerfacecolor=color, markeredgecolor='none', markersize=4, label=label_name))
    (x_min, x_max) = (tsne_result[:, 0].min(), tsne_result[:, 0].max())
    (y_min, y_max) = (tsne_result[:, 1].min(), tsne_result[:, 1].max())
    x_margin = max(0.03 * (x_max - x_min), 1e-06)
    y_margin = max(0.03 * (y_max - y_min), 1e-06)
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(direction='out', length=2.5, width=0.7)
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
        spine.set_color('#333333')
    ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=6, frameon=False, handletextpad=0.35, columnspacing=1.0, borderaxespad=0.0)
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
    fig.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    fig.savefig(output_path.with_suffix('.svg'), bbox_inches='tight', facecolor='white')
    if show:
        plt.show()
    plt.close(fig)
    print(f't-SNE saved to: {output_path}')

def sequence_transform(data):
    """Convert raw IQ data of shape (N, 2, 128) to features of shape (N, 10, 128)."""
    data = np.asarray(data, dtype=np.float32)
    i_data = data[:, 0, :]
    q_data = data[:, 1, :]
    complex_data = i_data + 1j * q_data
    amplitude = np.abs(complex_data)
    phase = np.angle(complex_data)
    power2_spectrum = np.abs(np.fft.fft(complex_data ** 2, axis=-1))
    power4_spectrum = np.abs(np.fft.fft(complex_data ** 4, axis=-1))
    fft_data = np.fft.fft(complex_data, axis=-1)
    fft_real = fft_data.real
    fft_imag = fft_data.imag
    fft_magnitude = np.log1p(np.abs(fft_data))
    fft_phase = np.angle(fft_data)
    features = np.stack([i_data, q_data, amplitude, phase, power2_spectrum, power4_spectrum, fft_real, fft_imag, fft_magnitude, fft_phase], axis=1)
    return features.astype(np.float32)
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
    data_2 = np.load('./val_test_18dB/X_val_test_0db_2ch.npy')
    data_2 = torch.tensor(data_2)
    data_2 = data_2.to(torch.float32)
    target_2 = np.load('./val_test_18dB/Y_val_test_0db.npy')
    data_10 = np.load('./val_test_18dB/X_val_test_0db_10ch.npy')
    data_10 = torch.tensor(data_10)
    data_10 = data_10.to(torch.float32)
    target_10 = np.load('./val_test_18dB/Y_val_test_0db.npy')
    dataset2 = Data(data_2, target_2)
    dataset10 = Data(data_10, target_10)
    dataloader2 = DataLoader(dataset2, 64)
    dataloader10 = DataLoader(dataset10, 64)
    data_fintune = np.load('./data/X_100.npy')
    target_fintune = np.load('./data/fintune/Y_samples100.npy')
    data_f = np.load('./data/fintune/X_samples100.npy')
    data = np.load('./data/val/val.npy')
    target = np.load('./data/val/target_val.npy')
    data_few1_2 = np.load('./dataset/train_fewshot/5shot/raw_train_5shot_data.npy')
    data_few1_10 = np.load('./dataset/train_fewshot/5shot/sequence_train_5shot_data.npy')
    target_few1_2 = np.load('./dataset/train_fewshot/5shot/raw_train_5shot_label.npy')
    dataset_few1_2 = Data(data_few1_2, target_few1_2)
    dataset_few1_10 = Data(data_few1_10, target_few1_2)
    dataloader_few1_2 = DataLoader(dataset_few1_2, 64)
    dataloader_few1_10 = DataLoader(dataset_few1_10, 64)
    data_b_100_2 = np.load('./dataset/RML2016.10b/val_fewshot/5shot/raw_val_5shot_data.npy')
    target_b_100_2 = np.load('./dataset/RML2016.10b/val_fewshot/5shot/raw_val_5shot_label.npy')
    dataset_b_100_2 = Data(data_b_100_2, target_b_100_2)
    data_b_100_10 = np.load('./dataset/RML2016.10b/val_fewshot/5shot/sequence_val_5shot_data.npy')
    target_b_100_10 = np.load('./dataset/RML2016.10b/val_fewshot/5shot/sequence_val_5shot_label.npy')
    dataset_b_100_10 = Data(data_b_100_10, target_b_100_10)
    dataloader_b_100_2 = DataLoader(dataset_b_100_2, 64)
    dataloader_b_100_10 = DataLoader(dataset_b_100_10, 64)
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
    a = pre_model(model_z, model_LSTM, model_r, model_s, model_s1)
    a.to(device)
    optimizer = optim.AdamW(a.parameters(), lr=0.01, weight_decay=0.001)
    loss_optimizer = optim.AdamW(loss_model.parameters(), lr=0.1)
    model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=100)
    experiment_log_dir = './'
    from plot_feature_similarity import plot_feature_similarity_from_checkpoint
    import torch
    from plot_feature_similarity import plot_branch_similarity_comparison
    pre_ck = torch.load('./saved_models/best9.pt', map_location=device)
    comparison_results = plot_branch_similarity_comparison(checkpoint_path='./saved_models/best_finetune9.pt', device=device, feature_dl=dataloader2, feature_dl_mask=dataloader10, build_model_func=build_model, pre_ck=pre_ck, save_path='./figures/branch_similarity_comparison.png', unknown_label=9, pairs_per_class=5000, seed=2026, y_max=7.0, bw_method='scott', show=True)
