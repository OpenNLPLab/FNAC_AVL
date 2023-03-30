import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18, vgg16
from utils import truncated_normal_  # a replacement for nn.init.truncated_normal in pytorch 1.12
from typing import Union, List, Dict, Any, cast
import shutil
import os
import copy


class Asy(nn.Module):
    def __init__(self, tau, dim,  dropout_img, dropout_aud):
        super(Asy, self).__init__()
        self.tau = tau

        # Vision model
        self.imgnet = resnet18(pretrained=True)
        self.imgnet.avgpool = nn.Identity()
        self.imgnet.fc = nn.Identity()
        self.img_proj = nn.Conv2d(512, dim, kernel_size=(1, 1))
        self.img_dropout = nn.Dropout(p= dropout_img)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Audio model
        self.audnet = resnet18()
        self.audnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.audnet.avgpool = nn.AdaptiveMaxPool2d((1, 1))
        self.audnet.fc = nn.Identity()
        self.aud_proj = nn.Linear(512, dim)
        self.aud_dropout = nn.Dropout(p= dropout_aud)

        self.high_conf_thresh = 0.6
        # self.low_conf_thresh = 0.4

        # Initialize weights (except pretrained visual model)
        for net in [self.audnet, self.img_proj, self.aud_proj]:
            for m in net.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(
                        m.weight, mean=0.0, std=0.01)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.normal_(m.weight, mean=1, std=0.02)
                    nn.init.constant_(m.bias, 0)

    def calculate_loss(self, img, aud, name=None):
        aud_attn = (aud@aud.transpose(0,1)) / self.tau

        img_avg =  self.avgpool(img)[:,:,0,0]
        img_attn = (img_avg@img_avg.transpose(0,1)) / self.tau
        
        B = img.shape[0]
        h,w = img.shape[2], img.shape[3]
        
        Slogits = torch.einsum('nchw,mc->nmhw', img, aud) / self.tau

        loc_map = Slogits[torch.arange(B), torch.arange(B)]
        loc_map = (loc_map - torch.amin(loc_map, (1,2), keepdim=True))/ \
        (torch.amax(loc_map, (1,2), keepdim=True) - torch.amin(loc_map, (1,2), keepdim=True) + 1e-5)

        # frg_feature = img * loc_map.unsqueeze(1)
        frg_feature = img * (loc_map>self.high_conf_thresh).unsqueeze(1) # foreground visual features
        frg_feature = frg_feature.flatten(-2, -1).mean(dim=-1) 
        frg_attn = (frg_feature@frg_feature.transpose(0,1)) / self.tau

        logits = Slogits.flatten(-2, -1).max(dim=-1)[0]
        labels = torch.arange(B).long().to(img.device)
        
        loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits.permute(1, 0), labels)

        fnac_loss1 = F.l1_loss(torch.softmax(aud_attn, dim=1), torch.softmax(logits, dim=1)) # FNS-1
        fnac_loss2 = F.l1_loss(torch.softmax(aud_attn, dim=1), torch.softmax(frg_attn, dim=1)) # TNS
        fnac_loss3 = F.l1_loss(torch.softmax(img_attn, dim=1), torch.softmax(logits, dim=1)) # FNS-2
    
        return [loss, fnac_loss1, fnac_loss2,  fnac_loss3], Slogits

    def forward(self, image, audio, name=None):
        # Image b*3*h*w 
        img = self.imgnet(image).unflatten(1, (512, 7, 7))
        img = self.img_dropout(img)
        img = self.img_proj(img) # b*512*7*7
        img = nn.functional.normalize(img, dim=1)

        # Audio b*1*h*w
        aud = self.audnet(audio)
        aud = self.aud_dropout(aud)
        aud = self.aud_proj(aud) # b*512
        aud = nn.functional.normalize(aud, dim=1)

        # Compute loss
        loss, logits = self.calculate_loss(img, aud, name=name)

        # Compute avl maps
        with torch.no_grad():
            B = img.shape[0]
            Savl = logits[torch.arange(B), torch.arange(B)]

        return loss, Savl
