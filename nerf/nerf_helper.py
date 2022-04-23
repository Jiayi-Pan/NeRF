
from tracemalloc import start
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F


def hello_model():
    print("hello from model.py")



def PosEncode(x, L):
    '''
    Mapping inputs to a higher dimensional space using
    high frequency functions enables better fitting of data
    x -> (sin(2^0 x), cos(2^0 x), ... , sin(2^(L-1) x), cos(2^(L-1) x))

    args:
        x: (N, 3)
        L: half the expanded dimension
    
    out:
        enc_x: (N, 3*2*L)
    '''
    N = x.shape[0]
    enc_x = torch.zeros((N, 3, 2*L), dtype=x.dtype, device=x.device)

    freq_band = 2**torch.floor(torch.arange(0, L, 0.5, device=x.device))
    freq_t_x = freq_band.reshape(1, -1) * x.reshape(N, -1, 1)  # freq_band*x (N, 3, 2L)

    enc_x[:,:,0::2] = torch.sin(freq_t_x[:, :,0::2])
    enc_x[:,:,1::2] = torch.cos(freq_t_x[:, :,1::2])
    enc_x = torch.flatten(enc_x, start_dim=1)
    return enc_x