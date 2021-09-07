import torch
import timm

input1 = torch.randn(2, 5, 128)
input2 = torch.randn(2, 5, 128)
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
output = cos(input1, input2)
print(output)

# m = timm.create_model('divervit_base_patch16_224', pretrained=False)
# m.cuda()
# m(torch.rand([4, 3, 224, 224]).cuda())
