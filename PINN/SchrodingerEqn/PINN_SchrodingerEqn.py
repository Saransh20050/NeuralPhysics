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
    self.l4=torch.nn.Linear(64,2)
    self.tanh=torch.nn.Tanh()
  def forward(self,x:torch.Tensor)->torch.Tensor:
    return self.l4(self.tanh(self.l3(self.tanh(self.l2(self.tanh(self.l1(x)))))))
model_0=ann()
optimizer=torch.optim.Adam(params=model_0.parameters(),lr=0.001)

epochs=3000
epoch_cnt=[]
loss_epoch=[]
phy_loss_epoch=[]
bc_loss_epoch=[]
initial_loss_epoch=[]
for epoch in range(epochs):
  model_0.train()
  t=torch.rand(300,1)*torch.pi/2
  x=torch.rand(300,1)*10-5
  x.requires_grad=True
  t.requires_grad=True
  input=torch.cat([x,t],dim=1)
  output=model_0(input)
  u=output[:,0:1]
  v=output[:,1:2]
  dv_dt=torch.autograd.grad(
      inputs=t,
      outputs=v,
      grad_outputs=torch.ones_like(v),
      create_graph=True
  )[0]
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
  dv_dx=torch.autograd.grad(
      inputs=x,
      outputs=v,
      grad_outputs=torch.ones_like(v),
      create_graph=True
  )[0]
  d2v_dx2=torch.autograd.grad(
      inputs=x,
      outputs=dv_dx,
      grad_outputs=torch.ones_like(dv_dx),
      create_graph=True
  )[0]
  phy_loss1=-1*dv_dt+d2u_dx2*0.5+u**3+u*v**2
  phy_loss2=du_dt+d2v_dx2*0.5+v*u**2+v**3
  phy_loss1=torch.mean(phy_loss1**2)
  phy_loss2=torch.mean(phy_loss2**2)
  phy_loss=phy_loss1+phy_loss2
  bc_x1=torch.ones_like(t)*5
  bc_input1=torch.cat([bc_x1,t],dim=1)
  bc_output1=model_0(bc_input1)
  bc_u1=bc_output1[:,0:1]
  bc_v1=bc_output1[:,1:2]
  bc_loss1=torch.mean(bc_u1**2)+torch.mean(bc_v1**2)
  bc_x2=torch.ones_like(t)*-5
  bc_input2=torch.cat([bc_x2,t],dim=1)
  bc_output2=model_0(bc_input2)
  bc_u2=bc_output2[:,0:1]
  bc_v2=bc_output2[:,1:2]
  bc_loss2=torch.mean(bc_u2**2)+torch.mean(bc_v2**2)
  bc_loss=bc_loss1+bc_loss2
  initial_t1=torch.zeros_like(x)
  initial_input1=torch.cat([x,initial_t1],dim=1)
  initial_output1=model_0(initial_input1)
  initial_u1=initial_output1[:,0:1]
  initial_v1=initial_output1[:,1:2]
  initial_loss=torch.mean((initial_u1-2*1/torch.cosh(x))**2)+torch.mean(initial_v1**2)
  loss=initial_loss+phy_loss+bc_loss
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  with torch.inference_mode():
    epoch_cnt.append(epoch)
    loss_epoch.append(loss)
    phy_loss_epoch.append(phy_loss)
    bc_loss_epoch.append(bc_loss)
    initial_loss_epoch.append(initial_loss)


  

  

  
with torch.inference_mode():
  plt.plot(epoch_cnt,loss_epoch,label="loss")
  plt.plot(epoch_cnt,phy_loss_epoch,label="phy_loss")
  plt.plot(epoch_cnt,bc_loss_epoch,label="bc_loss")
  plt.plot(epoch_cnt,initial_loss_epoch,label="initial_loss")
  plt.legend()
  
