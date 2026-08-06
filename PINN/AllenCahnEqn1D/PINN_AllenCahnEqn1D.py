import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
class ann(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.l1=torch.nn.Linear(2,64)
    self.l2=torch.nn.Linear(64,128)
    self.l3=torch.nn.Linear(128,64)
    self.l4=torch.nn.Linear(64,1)
    self.tanh=torch.nn.Tanh()
  def forward(self,x:torch.Tensor)->torch.Tensor:
    return self.l4(self.tanh(self.l3(self.tanh(self.l2(self.tanh(self.l1(x)))))))

model_0=ann()
optimizer=torch.optim.Adam(params=model_0.parameters(),lr=0.001)

epochs=600
epoch_cnt=[]
phy_loss_epoch=[]
bc_loss_epoch=[]
input_loss_epoch=[]
loss_epoch=[]
for epoch in range(epochs):
  model_0.train()
  x=torch.rand(300,1)*2-1
  t=torch.rand(300,1)
  x.requires_grad=True
  t.requires_grad=True
  input=torch.cat([x,t],dim=1)
  u=model_0(input)
  du_dt=torch.autograd.grad(
      inputs=t,
      outputs=u,
      grad_outputs=torch.ones_like(u),
      create_graph=True
  )[0]
  du_dx=torch.autograd.grad(
      inputs=x,
      outputs=u,
      grad_outputs=torch.ones_like(u),
      create_graph=True
  )[0]
  d2u_dx2=torch.autograd.grad(
      inputs=x,
      outputs=du_dx,
      grad_outputs=torch.ones_like(du_dx),
      create_graph=True
  )[0]
  alpha=0.1
  phy_loss=du_dt-alpha*alpha*d2u_dx2+(u-u**3)
  phy_loss=torch.mean(phy_loss**2)
  t_initial=torch.zeros_like(x)
  input_initial=torch.cat([x,t_initial],dim=1)
  u_initial=model_0(input_initial)
  initial_loss=torch.mean((u_initial-torch.sin(torch.pi*x))**2)
  x_bc1=torch.ones_like(t)*-1
  x_bc2=torch.ones_like(t)
  input_bc1=torch.cat([x_bc1,t],dim=1)
  input_bc2=torch.cat([x_bc2,t],dim=1)
  u_bc1=model_0(input_bc1)
  u_bc2=model_0(input_bc2)
  bc_loss=torch.mean(u_bc1**2)+torch.mean(u_bc2**2)
  loss=phy_loss+initial_loss+bc_loss
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  with torch.inference_mode():
    epoch_cnt.append(epoch)
    phy_loss_epoch.append(phy_loss)
    bc_loss_epoch.append(bc_loss)
    input_loss_epoch.append(initial_loss)
    loss_epoch.append(loss)
with torch.inference_mode():
  plt.plot(epoch_cnt,phy_loss_epoch,label="Physics loss")
  plt.plot(epoch_cnt,bc_loss_epoch,label="BC loss")
  plt.plot(epoch_cnt,input_loss_epoch,label="Input loss")
  plt.plot(epoch_cnt,loss_epoch,label="Total loss")
  plt.legend()
