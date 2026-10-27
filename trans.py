import numpy as np
from scipy.signal import get_window
import pickle
import matplotlib.pyplot as plt
import torch
import math
from random import *

def Data_Transformer(inputs, random_seed, num=None):
    """Transform [batch, 2, length] IQ signals without changing their length.

    Modes: 1 amplitude/phase; 2 magnitude spectra of IQ squared/fourth power;
    3 FFT real/imaginary; 4 log1p(FFT magnitude)/FFT phase.
    random_seed is the historical mode identifier, not an RNG seed.
    num is retained as an unused compatibility argument.
    """
    if inputs.ndim != 3 or inputs.shape[1] != 2:
        raise ValueError('Expected raw IQ data with shape [batch, 2, length].')
    if random_seed == None:
        random_seed = randint(1, 4)
    if random_seed not in (1, 2, 3, 4):
        raise ValueError('Representation mode must be 1, 2, 3 or 4.')
    AP_train = []
    sc_train = []
    fftiq = []
    i = inputs[:, 0, :]
    q = inputs[:, 1, :]
    iq = i + 1j * q
    iq = iq.to(torch.complex64)
    if random_seed == 1:
        A = torch.abs(iq)
        A = torch.reshape(A, (iq.shape[0], 1, inputs.shape[-1]))
        P = torch.angle(iq)
        P = torch.reshape(P, (iq.shape[0], 1, inputs.shape[-1]))
        AP = torch.cat((A, P), dim=1)
        return AP
    if random_seed == 2:
        power2_spectrum = torch.abs(torch.fft.fft(iq ** 2, dim=-1))
        power4_spectrum = torch.abs(torch.fft.fft(iq ** 4, dim=-1))
        return torch.stack((power2_spectrum, power4_spectrum), dim=1)
    if random_seed == 3:
        fiq = torch.fft.fft(iq)
        fi = fiq.real
        fi = torch.reshape(fi, (iq.shape[0], 1, inputs.shape[-1]))
        fq = fiq.imag
        fq = torch.reshape(fq, (iq.shape[0], 1, inputs.shape[-1]))
        fiq = torch.cat((fi, fq), dim=1)
        return fiq
    if random_seed == 4:
        fiq = torch.fft.fft(iq)
        mag = torch.log1p(torch.abs(fiq)).reshape(iq.shape[0], 1, inputs.shape[-1])
        pha = torch.angle(fiq).reshape(iq.shape[0], 1, inputs.shape[-1])
        spec = torch.cat((mag, pha), dim=1)
        return spec
