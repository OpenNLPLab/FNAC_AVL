from model import EZVSL
import torch
import numpy as np


model = EZVSL(tau=0.03, dim=512)
model.cuda()

pseudo_img = torch.rand(4,3,224,224).cuda(non_blocking=True)
pseudo_audio = torch.rand(4,3,224,224).cuda(non_blocking=True)
print(pseudo_audio.shape, pseudo_img.shape)
output = model(pseudo_img, pseudo_audio)
