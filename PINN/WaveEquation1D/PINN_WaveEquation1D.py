import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ann(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.l1=torch.nn.Linear(2,256)
    self.l2=torch.nn.Linear(256,128)
    self.l3=torch.nn.Linear(128,128)
    self.l4=torch.nn.Linear(128,1)
    self.tanh=torch.nn.Tanh()
  def forward(self,x:torch.Tensor)->torch.Tensor:
    return self.l4(self.tanh(self.l3(self.tanh(self.l2(self.tanh(self.l1(x)))))))
model_0=ann()
optimizer=torch.optim.Adam(params=model_0.parameters(),lr=0.001)
epochs=1000
epoch_cnt=[]
phy_loss_epoch=[]
initial_loss_epoch=[]
bc_loss_epoch=[]
loss_epoch=[]
for epoch in range(epochs):
  model_0.train()
  x=torch.rand(300,1)
  t=torch.rand(300,1)
  x.requires_grad=True
  t.requires_grad=True
  input=torch.cat([x,t],dim=1)
  u=model_0(input)
  du_dx=torch.autograd.grad(
      inputs=x,
      outputs=u,
      grad_outputs=torch.ones_like(u),
      create_graph=True
  )[0]
  du_dt=torch.autograd.grad(
      inputs=t,
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
  d2u_dt2=torch.autograd.grad(
      inputs=t,
      outputs=du_dt,
      grad_outputs=torch.ones_like(du_dt),
      create_graph=True
  )[0]
  c=1
  phy_loss=d2u_dt2-c*c*d2u_dx2
  phy_loss=torch.mean(phy_loss**2)
  t_initial=torch.zeros_like(x)
  t_initial.requires_grad=True
  input_initial=torch.cat([x,t_initial],dim=1)
  u_initial=model_0(input_initial)
  du_initial_dt=torch.autograd.grad(
      inputs=t_initial,
      outputs=u_initial,
      grad_outputs=torch.ones_like(u_initial),
      create_graph=True
  )[0]
  initial_loss=torch.mean(du_initial_dt**2)+torch.mean((u_initial-torch.sin(torch.pi*x))**2)
  x_bc1=torch.zeros_like(t)
  x_bc2=torch.ones_like(t)
  input_bc1=torch.cat([x_bc1,t],dim=1)
  input_bc2=torch.cat([x_bc2,t],dim=1)
  u_bc1=model_0(input_bc1)
  u_bc2=model_0(input_bc2)
  bc_loss=torch.mean((u_bc1)**2)+torch.mean((u_bc2)**2)
  loss=phy_loss+initial_loss+bc_loss
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  with torch.inference_mode():
    epoch_cnt.append(epoch)
    phy_loss_epoch.append(phy_loss)
    bc_loss_epoch.append(bc_loss)
    initial_loss_epoch.append(initial_loss)
    loss_epoch.append(loss)
  with torch.inference_mode():
  plt.plot(epoch_cnt,phy_loss_epoch,label="phy_loss")
  plt.plot(epoch_cnt,bc_loss_epoch,label="bc_loss")
  plt.plot(epoch_cnt,initial_loss_epoch,label="initial_loss")
  plt.plot(epoch_cnt,loss_epoch,label="loss")
  plt.legend()
  x_test=torch.rand(1000,1)
  t_test=torch.rand(1000,1)
  input_test=torch.cat([x_test,t_test],dim=1)
  u_test_predicted=model_0(input_test)
  u_test_actual=torch.sin(torch.pi*x_test)*torch.cos(torch.pi*t_test)
  with torch.inference_mode():
   plt.scatter(x_test,u_test_predicted,label="predicted")
   plt.scatter(x_test,u_test_actual,label="actual")
   plt.legend()
  with torch.inference_mode():
   plt.scatter(t_test,u_test_predicted,label="predicted")
   plt.scatter(t_test,u_test_actual,label="actual")
   plt.legend()
