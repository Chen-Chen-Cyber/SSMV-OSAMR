import torch
import torch.nn as nn
import pickle
from torch.utils import data
from random import *
import math
import random
import numpy as np
from self import data_transform_masked4cl
device = torch.device('cuda:0')

def Data_Augmentation(inputs, random_seed, theta_list=None, std_list=None):
    """Apply one of the original augmentation modes 2 through 10.

    random_seed is a mode identifier. Mode 1 had no implementation in the
    supplied source and is rejected explicitly. Rotation angles, including
    the original 3*pi/4 angle for mode 6, are preserved. theta_list and
    std_list remain compatibility arguments; the original fixed values apply.
    """
    if inputs.ndim != 3 or inputs.shape[1] % 2:
        raise ValueError('Expected [batch, even_number_of_channels, length].')
    if random_seed == None:
        random_seed = randint(2, 10)
    if random_seed not in range(2, 11):
        raise ValueError('Augmentation mode must be an integer from 2 through 10.')
    if theta_list == None:
        theta_list = [0, math.pi / 2, math.pi, math.pi / 2 * 3]
    if std_list == None:
        std_list = [0, 0.05, 0.1, 0.15]
    if random_seed == 4:
        flip_v = torch.zeros_like(inputs)
        flip_v[:, :inputs.shape[1] // 2, :] = inputs[:, :inputs.shape[1] // 2, :]
        flip_v[:, inputs.shape[1] // 2:, :] = -inputs[:, inputs.shape[1] // 2:, :]
        Data_Aug = flip_v
    elif random_seed == 2:
        flip_h = torch.zeros_like(inputs)
        flip_h[:, :inputs.shape[1] // 2, :] = -inputs[:, :inputs.shape[1] // 2, :]
        flip_h[:, inputs.shape[1] // 2:, :] = inputs[:, inputs.shape[1] // 2:, :]
        Data_Aug = flip_h
    elif random_seed == 3:
        flip_v_h = torch.zeros_like(inputs)
        flip_v_h[:, :inputs.shape[1] // 2, :] = -inputs[:, :inputs.shape[1] // 2, :]
        flip_v_h[:, inputs.shape[1] // 2:, :] = -inputs[:, inputs.shape[1] // 2:, :]
        Data_Aug = flip_v_h
    elif random_seed == 8:
        std_temp = 0.05
        noise = torch.normal(mean=0, std=std_temp, size=inputs.shape).to(inputs.device)
        Data_Aug = inputs + noise
    elif random_seed == 9:
        std_temp = 0.1
        noise = torch.normal(mean=0, std=std_temp, size=inputs.shape).to(inputs.device)
        Data_Aug = inputs + noise
    elif random_seed == 10:
        std_temp = 0.15
        noise = torch.normal(mean=0, std=std_temp, size=inputs.shape).to(inputs.device)
        Data_Aug = inputs + noise
    elif 4 < random_seed <= 7:
        if random_seed == 5:
            theta = math.pi / 2
            i_data = inputs[:, :inputs.shape[1] // 2, :]
            q_data = inputs[:, inputs.shape[1] // 2:, :]
            i_data_gen = math.cos(theta) * i_data - math.sin(theta) * q_data
            q_data_gen = math.sin(theta) * i_data + math.cos(theta) * q_data
            Data_Aug = torch.cat([i_data_gen, q_data_gen], dim=1)
        if random_seed == 6:
            theta = math.pi / 4 * 3
            i_data = inputs[:, :inputs.shape[1] // 2, :]
            q_data = inputs[:, inputs.shape[1] // 2:, :]
            i_data_gen = math.cos(theta) * i_data - math.sin(theta) * q_data
            q_data_gen = math.sin(theta) * i_data + math.cos(theta) * q_data
            Data_Aug = torch.cat([i_data_gen, q_data_gen], dim=1)
        if random_seed == 7:
            theta = math.pi / 2 * 3
            i_data = inputs[:, :inputs.shape[1] // 2, :]
            q_data = inputs[:, inputs.shape[1] // 2:, :]
            i_data_gen = math.cos(theta) * i_data - math.sin(theta) * q_data
            q_data_gen = math.sin(theta) * i_data + math.cos(theta) * q_data
            Data_Aug = torch.cat([i_data_gen, q_data_gen], dim=1)
    return Data_Aug

def flip_row(inputs, label):
    """Stack the original, three flips and three masked views with labels.

    The masking helper repeats the batch three times at masking_ratio=0.5,
    so seven label copies match the seven output view blocks.
    """
    flip_v = torch.zeros_like(inputs)
    flip_h = torch.zeros_like(inputs)
    flip_v_h = torch.zeros_like(inputs)
    flip_v[:, :inputs.shape[1] // 2, :] = inputs[:, :inputs.shape[1] // 2, :]
    flip_v[:, inputs.shape[1] // 2:, :] = -inputs[:, inputs.shape[1] // 2:, :]
    flip_h[:, :inputs.shape[1] // 2, :] = -inputs[:, :inputs.shape[1] // 2, :]
    flip_h[:, inputs.shape[1] // 2:, :] = inputs[:, inputs.shape[1] // 2:, :]
    flip_v_h[:, :inputs.shape[1] // 2, :] = -inputs[:, :inputs.shape[1] // 2, :]
    flip_v_h[:, inputs.shape[1] // 2:, :] = -inputs[:, inputs.shape[1] // 2:, :]
    theta = math.pi / 2
    i_data = inputs[:, :inputs.shape[1] // 2, :]
    q_data = inputs[:, inputs.shape[1] // 2:, :]
    i_data_gen = math.cos(theta) * i_data - math.sin(theta) * q_data
    q_data_gen = math.sin(theta) * i_data + math.cos(theta) * q_data
    Data_Aug_2 = torch.cat([i_data_gen, q_data_gen], dim=1)
    (data_masked_m, mask) = data_transform_masked4cl(inputs, masking_ratio=0.5, lm=7)
    return (torch.cat([inputs, flip_v, flip_h, flip_v_h, data_masked_m], dim=0), torch.cat([label, label, label, label, label, label, label], dim=0))

def flip(inputs):
    random_seed = randint(1, 3)
    if random_seed == 1:
        flip_v = torch.zeros_like(inputs)
        flip_v[:, 0, :] = inputs[:, 0, :]
        flip_v[:, 1, :] = -inputs[:, 1, :]
        flip_data = flip_v
    elif random_seed == 2:
        flip_h = torch.zeros_like(inputs)
        flip_h[:, 0, :] = -inputs[:, 0, :]
        flip_h[:, 1, :] = inputs[:, 1, :]
        flip_data = flip_h
    else:
        flip_v_h = torch.zeros_like(inputs)
        flip_v_h[:, 0, :] = -inputs[:, 0, :]
        flip_v_h[:, 1, :] = -inputs[:, 1, :]
        flip_data = flip_v_h
    return flip_data

def rotation(inputs, label, theta: list):
    out = []
    label_list = []
    for r in theta:
        i_data = inputs[:, 0, :]
        q_data = inputs[:, 1, :]
        i_data_gen = math.cos(r) * i_data - math.sin(r) * q_data
        q_data_gen = math.sin(r) * i_data + math.cos(r) * q_data
        out.append(torch.cat([i_data_gen.unsqueeze(1), q_data_gen.unsqueeze(1)], dim=1))
        label_list.append(label)
    return (torch.cat(out, dim=0), torch.cat(label_list, dim=0))

def addguassiannoise(inputs, label, theta_2_all: list):
    data_gen = []
    label_gen = []
    for theta_2 in theta_2_all:
        noise = torch.normal(mean=0, std=theta_2, size=inputs.shape).to(inputs.device)
        data_gen.append(inputs + noise)
        label_gen.append(label)
    return (torch.cat(data_gen, dim=0), torch.cat(label_gen, dim=0))

def rotation_flip(inputs, label):
    (flip_data, flip_label) = (flip(inputs), label)
    (r_data, r_label) = rotation(inputs, label, [math.pi / 2, 3 / 2 * math.pi])
    return (torch.cat([flip_data, r_data], dim=0), torch.cat([flip_label, r_label], dim=0))

def load_pickle(file_name):
    f = open(file_name, 'rb')
    data = pickle.load(f)
    f.close()
    return data
if __name__ == '__main__':
    data = torch.randn((32, 8, 128))
    DA = []
    for idex in [4, 2, 3, 8, 5]:
        Data_DA = Data_Augmentation(data, idex, theta_list=None, std_list=None)
        DA.append(Data_DA)
    DA_stack = torch.cat(DA, dim=0)
    print(DA_stack.shape)
