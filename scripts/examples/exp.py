import torch


m = torch.nn.Softmax(dim=1)
input = torch.randn(1, 3, 4, 4)
output = m(input)
print(input)
print(output)
print(output[0, :, 0, 0].sum())