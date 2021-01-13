import torch
import torch.nn as nn
import torch.nn.functional as F

def conv1x1(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    
    def forward(self, x):
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(in_channel, out_channel, stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        if downsample is None:
            self.downsample = Identity()
        else:
            self.downsample = downsample
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.downsample(x) + self.bn2(self.conv2(out)))

        return out

class MLP(nn.Module):
    def __init__(self, in_dim=256, out_dim=8, dropout=0.5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x    
    
    
class DCNet(nn.Module):
    def __init__(self, dim=64):
        super(DCNet, self).__init__()        
        
        self.conv1 = nn.Conv2d(3, dim, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        ndims = [1*dim, 2*dim, 4*dim, 8*dim] 

        self.res1 = ResBlock(ndims[0], ndims[1], stride=2, downsample=nn.Sequential(
            conv1x1(ndims[0], ndims[1], stride=2), nn.BatchNorm2d(ndims[1])))

        self.choice_contrast1 = nn.Sequential(conv3x3(ndims[1], ndims[1]), nn.BatchNorm2d(ndims[1]))
        self.res2 = ResBlock(ndims[1], ndims[2], stride=2, downsample=nn.Sequential(
            conv1x1(ndims[1], ndims[2], stride=2), nn.BatchNorm2d(ndims[2])))

        self.choice_contrast2 = nn.Sequential(conv3x3(ndims[2], ndims[2]), nn.BatchNorm2d(ndims[2]))
        self.res3 = ResBlock(ndims[2], ndims[3], stride=2, downsample=nn.Sequential(
            conv1x1(ndims[2], ndims[3], stride=2), nn.BatchNorm2d(ndims[3])))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = MLP(ndims[3], 1, dropout=0.5)
                
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.InstanceNorm2d, nn.BatchNorm2d)):
                if m.affine:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)                                 
    
    # x: [b, 10, 3, h, w]   
    def dual_contrast(self, x):
        b = x.size(0)

        # encode row/column feature
        x = x.view((b*10, -1) + x.shape[3:])                               # [b*10, 3, h, w]
        x = self.bn1(self.conv1(x))                                        # [b*10, 64, h/2, w/2]   
        x = self.maxpool(self.relu(x))                                     # [b*10, 64, h/4, w/4]      
        x = self.res1(x)                                                   # [b*10, 2*64, h/8, w/8] 

        # rule contrast on row/column
        x = x.view((b, 10, -1) + x.shape[2:])                              # [b, 10, 2*64, h/8, w/8]    
        x = x[:, 2:] - x[:, 0:2].mean(dim=1, keepdim=True)                 # [b, 8, 2*64, h/8, w/8]

        # choice contrast
        cc = self.choice_contrast1(x.mean(dim=1))                          # [b, 2*64, h/8, w/8]     
        x = x - cc.unsqueeze(1)                                            # [b, 8, 2*64, h/8, w/8]   
        x = self.res2(x.view((b*8, -1) + x.shape[3:]))                     # [b*8, 4*64, h/16, w/16]  

        x = x.view((b, 8, -1) + x.shape[2:])                               # [b, 8, 4*64, h/16, w/16]    
        cc = self.choice_contrast2(x.mean(dim=1))                          # [b, 4*64, h/16, w/16]     
        x = x - cc.unsqueeze(1)                                            # [b, 8, 4*64, h/16, w/16]   
        x = self.res3(x.view((b*8, -1) + x.shape[3:]))                     # [b*8, 8*64, h/32, w/32]  

        return x

    # x: [b, 16, h, w]
    def forward(self, x):
        b = x.shape[0]

        # images of the choices                              
        choices = x[:, 8:].unsqueeze(dim=2)                                # [b, 8, 1, h, w]   
        
        # images of the rows
        row1 = x[:, 0:3].unsqueeze(1)                                      # [b, 1, 3, h, w]  
        row2 = x[:, 3:6].unsqueeze(1)                                      # [b, 1, 3, h, w]     
    
        row3_p = x[:, 6:8].unsqueeze(dim=1).repeat(1, 8, 1, 1, 1)          # [b, 8, 2, h, w]
        row3 = torch.cat((row3_p, choices), dim=2)                         # [b, 8, 3, h, w]

        rows = torch.cat((row1, row2, row3), dim=1)                        # [b, 10, 3, h, w] 

        # images of the columns
        col1 = x[:, 0:8:3].unsqueeze(1)                                    # [b, 1, 3, h, w]  
        col2 = x[:, 1:8:3].unsqueeze(1)                                    # [b, 1, 3, h, w] 
    
        col3_p = x[:, 2:8:3].unsqueeze(dim=1).repeat(1, 8, 1, 1, 1)        # [b, 8, 2, h, w]
        col3 = torch.cat((col3_p, choices), dim=2)                         # [b, 8, 3, h, w]    

        cols = torch.cat((col1, col2, col3), dim=1)                        # [b, 10, 3, h, w]     

        # inference               
        xr = self.dual_contrast(rows)                                      # [b*8, 8*64, h/32, w/32]  
        xc = self.dual_contrast(cols)                                      # [b*8, 8*64, h/32, w/32]  
        x = xr + xc

        x = self.avgpool(x).view(b*8, -1)                                  # [b*8, 8*64]  
        x = self.mlp(x)                                                    # [b*8, 1]   
        
        return x.view(b, 8)

