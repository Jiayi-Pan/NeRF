
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

    freq_band = torch.linspace(0, L-1, L, dtype=x.dtype, device=x.device)