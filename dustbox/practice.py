import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


torch.manual_seed(1)
"""
y = torch.rand(100).cuda()
W = torch.rand(100,200).cuda()
x = torch.rand(200,requires_grad=True).cuda()
b = torch.rand(100).cuda()
optimizer = torch.optim.AdamW([x], lr=1e-4)
loss = torch.nn.MSELoss(reduction='mean')
for epoch in range(10000):
    optimizer.zero_grad()
    output = torch.matmul(W,x)+b
    output_loss = loss(output,y)
    output_loss.backward()
    optimizer.step()
    if epoch %1000 == 0:
        print(output_loss)
print(y)
print(y-(torch.matmul(W,x)+b))
print(x.grad)
print(x.is_leaf)
print(W.grad)
print(b.grad)
print(y.grad)
"""

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768,768)
        self.fc2 = nn.Linear(768,30000)
    def forward(self,feature,**kwargs):
        x = self.fc1(feature)
        x = F.gelu(x)
        x = self.fc2(x)
        return x
net = Net().cuda()
x = torch.rand(768,requires_grad=True,device="cuda")
ansx = torch.rand(768,device="cuda")
ansy = torch.tensor(net(ansx).detach().cpu().numpy(),device="cuda")
optimizer = torch.optim.AdamW([x], lr=1e-4)
loss = torch.nn.MSELoss(reduction='sum')
for epoch in range(10000):
    optimizer.zero_grad()
    output = net(x)
    output_loss = loss(output,ansy)
    output_loss.backward()
    optimizer.step()
    if epoch %1000 == 0:
        print(output.requires_grad)
        print(output_loss)
