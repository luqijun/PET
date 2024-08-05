from torchvision import models
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import torch
import torch.nn as nn
from torch.nn import Conv2d
import torch.nn.functional as F
from .ODConv2d import ODConv2d
from util.misc import NestedTensor
from models.position_encoding import build_position_encoding


__all__ = ['FFNet']

class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
 
    def forward(self, x):
        return torch.Tensor.permute(x, self.dims)

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)      
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps) 
        x = x.permute(0, 3, 1, 2)      
        return x
    
def conv(in_ch, out_ch, ks, stride):
    
    pad = (ks - 1) // 2
    stage = nn.Sequential(nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ks, stride=stride,
                                       padding=pad, bias=False),
                          LayerNorm2d((out_ch,), eps=1e-06, elementwise_affine=True),
                          nn.GELU(approximate='none'))
    return stage

class ChannelAttention(nn.Module):  
    def __init__(self, channel):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel, 1, bias=False),
            nn.ReLU()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x=self.avg_pool(x)
        avgout = self.shared_MLP(x)
        return self.sigmoid(avgout)
    
class SpatialAttention(nn.Module):  
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
 
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
 
        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        x = torch.mean(x, dim=1, keepdim=True)
        x = self.conv1(x)
        return self.sigmoid(x)

class Backbone(nn.Module):
    def __init__(self):
        super(Backbone,self).__init__()
        
        feats =list(convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1).features.children())
        
        self.stem = nn.Sequential(*feats[0:2])
        self.stage1 = nn.Sequential(*feats[2:4])
        self.stage2 = nn.Sequential(*feats[4:6])
        # self.stage3 = nn.Sequential(*feats[6:12])
        
    def forward(self, x):
        x = x.float()
        x = self.stem(x)
        feature0 = x
        x = self.stage1(x)
        feature1 = x
        x = self.stage2(x)
        feature2 = x
        # x = self.stage3(x)
        
        return feature0, feature1, feature2

class ccsm(nn.Module):
    def __init__(self, channel, channel2, num_filters):
        super(ccsm, self).__init__()
        self.ch_att_s = ChannelAttention(channel)
        self.sa_s = SpatialAttention(7)
        self.conv1 = nn.Sequential(
            ODConv2d(channel, channel, kernel_size=1, stride=1, padding=0), 
            nn.ReLU(),
            nn.BatchNorm2d(num_features = channel))
        self.conv2 = nn.Sequential(
            ODConv2d(channel, channel2, kernel_size=1, stride=1, padding=0), 
            nn.ReLU(),
            nn.BatchNorm2d(num_features = channel2))
        
        self.conv3 = nn.Sequential(
            ODConv2d(channel2, channel2, kernel_size=1, stride=1, padding=0), 
            nn.ReLU(),
            nn.BatchNorm2d(num_features = channel2))
        self.conv4 = nn.Sequential(
            ODConv2d(channel2, num_filters, kernel_size=1, stride=1, padding=0), 
            nn.ReLU(),
            nn.BatchNorm2d(num_features = num_filters))
           
    def forward(self, x):
        x = self.ch_att_s(x)*x
        pool1 = x
        x = self.conv1(x)
        x = x + pool1
        x = self.conv2(x)
        pool2 = x
        x = self.conv3(x)
        x = x + pool2
        x = self.conv4(x)
        
        x = self.sa_s(x)*x

        return x


class Fusion(nn.Module):
    def __init__(self, num_filters1, num_filters2, num_filters3, out_channels=1):
        super(Fusion, self).__init__()
        self.downx1_x2 = nn.Conv2d(num_filters1, num_filters1, kernel_size=2, stride=2)
        self.downx1_x3 = nn.Conv2d(num_filters1, num_filters1, kernel_size=4, stride=4)
        self.downx2_x3 = nn.Conv2d(num_filters2, num_filters2, kernel_size=2, stride=2)
        self.upsample_x2_x1 = nn.ConvTranspose2d(in_channels=num_filters2, out_channels=num_filters2, kernel_size=4,
                                                 padding=1, stride=2)
        self.upsample_x3_x1 = nn.ConvTranspose2d(in_channels=num_filters3, out_channels=num_filters3, kernel_size=4,
                                                 padding=0, stride=4)
        self.upsample_x3_x2 = nn.ConvTranspose2d(in_channels=num_filters3, out_channels=num_filters3, kernel_size=4,
                                                 padding=1, stride=2)

        self.finalx1 = nn.Sequential(
            nn.Conv2d(num_filters1 + num_filters2 + num_filters3, out_channels, kernel_size=1, padding=0),
            nn.ReLU(),
        )

        self.finalx2 = nn.Sequential(
            nn.Conv2d(num_filters1 + num_filters2 + num_filters3, out_channels, kernel_size=1, padding=0),
            nn.ReLU(),
        )

        self.finalx3 = nn.Sequential(
            nn.Conv2d(num_filters1 + num_filters2 + num_filters3, out_channels, kernel_size=1, padding=0),
            nn.ReLU(),
        )

    def forward(self, x1, x2, x3):
        x2_up_x1 = self.upsample_x2_x1(x2)
        x3_up_x1 = self.upsample_x3_x1(x3)
        final_x1 = torch.cat([x1, x2_up_x1, x3_up_x1], dim=1)
        # final_x1 = self.finalx1(torch.cat([x1, x2_up_x1, x3_up_x1], dim=1))

        x1_down_x2 = self.downx1_x2(x1)
        x3_up_x2 = self.upsample_x3_x2(x3)
        final_x2 = torch.cat([x1_down_x2, x2, x3_up_x2], dim=1)
        # final_x2 = self.finalx2(torch.cat([x1_down_x2, x2, x3_up_x2], dim=1))

        x1_down_x3 = self.downx1_x3(x1)
        x2_down_x3 = self.downx2_x3(x2)
        final_x3 = torch.cat([x1_down_x3, x2_down_x3, x3], dim=1)
        # final_x3 = self.finalx3(torch.cat([x1_down_x3, x2_down_x3, x3], dim=1))

        return final_x1, final_x2, final_x3

from ..backbone_vgg import FeatsFusion
class FFNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        out_channels = 16 + 32 + 64
        num_filters = [16, 32, 64]
        self.backbone = Backbone()
        self.num_channels = 256 # out_channels
        self.position_embedding = build_position_encoding(args)

        # self.seg_head = Segmentation_Head(in_channels=[384, 192, 96])

        # self.ccsm1 = ccsm(96, 38, num_filters[0])
        # self.ccsm2 = ccsm(192, 96, num_filters[1])
        # self.ccsm3 = ccsm(384, 192, num_filters[2])
        # self.fusion = FeatsFusion(num_filters[0], num_filters[1], num_filters[2]) # Fusion(num_filters[0], num_filters[1], num_filters[2], out_channels=out_channels)
        self.fusion = FeatsFusion(96, 192, 384) # Fusion(num_filters[0], num_filters[1], num_filters[2], out_channels=out_channels)


    def forward(self, samples: NestedTensor):
        x = samples.tensors
        pool1, pool2, pool3 = self.backbone(x)

        # pred_seg_map = self.seg_head(pool1, pool2, pool3)
        # seg_attention = pred_seg_map.sigmoid()
        # pool1 = pool1 * F.interpolate(seg_attention, size=pool1.shape[-2:])
        # pool2 = pool2 * F.interpolate(seg_attention, size=pool2.shape[-2:])
        # pool3 = pool3 * F.interpolate(seg_attention, size=pool3.shape[-2:])

        # pool1 = self.ccsm1(pool1)
        # pool2 = self.ccsm2(pool2)
        # pool3 = self.ccsm3(pool3)
        pool1, pool2, pool3 = self.fusion((pool1, pool2, pool3))

        m = samples.mask
        # pos_x = self.position_embedding(samples).to(samples.tensors.dtype)
        mask_4x = F.interpolate(m[None].float(), size=pool1.shape[-2:]).to(torch.bool)[0]
        mask_8x = F.interpolate(m[None].float(), size=pool2.shape[-2:]).to(torch.bool)[0]

        out: Dict[str, NestedTensor] = {}
        out['4x'] = NestedTensor(pool1, mask_4x)
        out['8x'] = NestedTensor(pool2, mask_8x)

        pos = {}
        pos_4x = self.position_embedding(out['4x']).to(out['4x'].tensors.dtype)
        pos_8x = self.position_embedding(out['8x']).to(out['8x'].tensors.dtype)
        pos['4x'] = pos_4x
        pos['8x'] = pos_8x
        return out, pos



def build_ffnet(args):

    # treats persons as a single class
    num_classes = 1

    model = FFNet(args)
    return model



if __name__ == '__main__':
    x = torch.rand(size=(16, 3, 512, 512), dtype=torch.float32)
    model = FFNet()
    
    mu, mu_norm = model(x)
    print(mu.size(), mu_norm.size())
    