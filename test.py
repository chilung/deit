import torch
import timm

m = timm.create_model('divervit_base_patch16_224', pretrained=False)
m(torch.rand([4, 3, 224, 224]))
