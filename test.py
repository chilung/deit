import torch
import timm

# m = timm.create_model('divervit_base_patch16_224', pretrained=False)
# m(torch.rand([4, 3, 224, 224]))

a = torch.ones([2, 3])
print('==== a ====\n{}'.format(a))
b = a
b[0][1] = 10
print('==== a ====\n{}'.format(a))
print('==== b ====\n{}'.format(b))
