import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F

def hello_model():
    print("hello from model.py")



class NeRF(nn.Module):
    def __init__(self, ch_in_pos, ch_in_dir, skips=[4], fc_depth=8, fc_width=256):
        super(NeRF, self).__init__()
        '''

        args:
            ch_in_pos: number of input channels for encoded position (3*2*L)
            ch_in_dir: number of input channels for encoded direction (3*2*L)
            skip: where to include skip connection
            fc_depth: depth of fully connected layer
            fc_width: width of fully connected layer
        '''

        torch.set_default_dtype(torch.double)
        self.ch_in_pos = ch_in_pos
        self.ch_in_dir = ch_in_dir
        self.skips = skips
        self.fc_depth = fc_depth
        self.fc_width = fc_width

        for i in range(fc_depth):
            if i==0:
                layer = nn.Linear(ch_in_pos, fc_width)
            elif i in skips:
                layer = nn.Linear(ch_in_pos + fc_width, fc_width)
            else:
                layer = nn.Linear(fc_width, fc_width)
            
            torch.nn.init.kaiming_normal_(layer.weight)
            layer = nn.Sequential(
                layer,
                nn.ReLU()
            )
            setattr(self, f"fc_{i+1}", layer)
        
        self.fc_noact = nn.Linear(fc_width, fc_width)

        self.fc_dir = nn.Sequential(
            nn.Linear(fc_width + ch_in_dir, fc_width//2),
            nn.ReLU()
        )

        # Output Layers
        self.sigma = nn.Linear(fc_width, 1)
        self.rgb = nn.Sequential(
            nn.Linear(fc_width//2, 3),
            nn.Sigmoid()
        )
    
    def forward(self, pos, dirt):
        '''
        (pos + dir) ---NN--> (rgb, sigma)

        args:
            pos: (N, ch_in_pos)
            dirt: (N, ch_in_dir)
        
        out:
            out: (N, 4)
                rgb(3) + sigma(1)
        '''
        out = pos
        for i in range(self.fc_depth):
            if i in self.skips:
                out = torch.cat((pos, out), -1)
            out = getattr(self, f"fc_{i+1}")(out)
        
        sigma = self.sigma(out)

        out = self.fc_noact(out)
        out = torch.cat((out, dirt), -1)
        out = self.fc_dir(out)
        rgb = self.rgb(out)

        out = torch.cat((rgb, sigma), -1)
        return out



class TinyNeRF(nn.Module):
    def __init__(self, ch_in_pos, fc_width=128):
        super(TinyNeRF, self).__init__()
        '''

        args:
            ch_in_pos: number of input channels for encoded position (3*2*L)
            ch_in_dir: number of input channels for encoded direction (3*2*L)
        '''

        torch.set_default_dtype(torch.double)
        self.ch_in_pos = ch_in_pos
        self.fc_width = fc_width

        self.layer1 = torch.nn.Linear(ch_in_pos, fc_width)
        # Layer 2 (default: 128 -> 128)
        self.layer2 = torch.nn.Linear(fc_width, fc_width)
        # Layer 3 (default: 128 -> 4)
        self.layer3 = torch.nn.Linear(fc_width, 4)
        # Short hand for torch.nn.functional.relu
        self.relu = torch.nn.functional.relu
    
    def forward(self, x):
        '''
        (pos + dir) ---NN--> (rgb, sigma)

        '''
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x