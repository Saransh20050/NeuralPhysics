import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
class ann(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.l1=torch.nn.Linear(2,32)
    self.l2=torch.nn.Linear(32,128)
    self.l3=torch.nn.Linear(128,128)
    self.l4=torch.nn.Linear(128,1)
    self.tanh=torch.nn.Tanh()
  def forward(self,x:torch.Tensor)->torch.Tensor:
    return self.l4(self.tanh(self.l3(self.tanh(self.l2(self.tanh(self.l1(x)))))))
model_0=ann()
optimizer=torch.optim.Adam(model_0.parameters(),lr=0.001)
epochs=2000
epoch_cnt=[]
phy_loss_epoch=[]
bc_loss_epoch=[]
initial_loss_epoch=[]
loss_epoch=[]
for epoch in range(epochs):
  model_0.train()
  x=torch.rand(300,1)
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
  alpha=1
  phy_loss=du_dt-alpha*d2u_dx2
  phy_loss=torch.mean(phy_loss**2)
  t_initial=torch.zeros_like(x)
  input_initial=torch.cat([x,t_initial],dim=1)
  u_initial=model_0(input_initial)
  u_initial_actual=torch.sin(torch.pi*x)
  initial_loss=torch.mean((u_initial-u_initial_actual)**2)
  bc1_x=torch.zeros_like(t)
  bc1_input=torch.cat([bc1_x,t],dim=1)
  bc1_u=model_0(bc1_input)
  bc1_loss=torch.mean((bc1_u)**2)
  bc2_x=torch.ones_like(t)
  bc2_input=torch.cat([bc2_x,t],dim=1)
  bc2_u=model_0(bc2_input)
  bc2_loss=torch.mean((bc2_u)**2)
  bc_loss=bc1_loss+bc2_loss
  loss=phy_loss+bc_loss+initial_loss
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
  plt.plot(epoch_cnt,phy_loss_epoch,label="physics loss")
  plt.plot(epoch_cnt,bc_loss_epoch,label="bc_loss")
  plt.plot(epoch_cnt,initial_loss_epoch,label="initial_loss")
  plt.plot(epoch_cnt,loss_epoch,label="loss")
  plt.legend()
  x_test=torch.rand(300,1)
  t_test=torch.rand(300,1)
  input_test=torch.cat([x_test,t_test],dim=1)
  u_test_predicted=model_0(input_test)
  u_test_actual=(torch.e**(-torch.pi*torch.pi*t_test))*torch.sin(torch.pi*x_test)
  with torch.inference_mode():
   plt.scatter(x_test,u_test_actual,label="Actual")
   plt.scatter(x_test,u_test_predicted,label="Predicted")
   plt.legend()

