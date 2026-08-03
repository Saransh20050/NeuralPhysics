import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class ann(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.l1=torch.nn.Linear(1,32)
    self.l2=torch.nn.Linear(32,128)
    self.l3=torch.nn.Linear(128,2)
    self.tanh=torch.nn.Tanh()
  def forward(self,x:torch.Tensor)->torch.Tensor:
    return self.l3(self.tanh(self.l2(self.tanh(self.l1(x)))))
model_0=ann()
optimizer=torch.optim.Adam(params=model_0.parameters(),lr=0.001)

epochs=5000
epoch_cnt=[]
eqn1_loss_epoch=[]
eqn2_loss_epoch=[]
bc_loss_epoch=[]
loss_epoch=[]
for epoch in range(epochs):
  t=torch.rand(300,1)*(torch.pi*2)
  t.requires_grad=True
  output=model_0(t)
  x=output[:,0:1]
  y=output[:,1:2]
  dx_dt=torch.autograd.grad(
      outputs=x,
      inputs=t,
      grad_outputs=torch.ones_like(x),
      create_graph=True
  )[0]
  dy_dt=torch.autograd.grad(
      outputs=y,
      inputs=t,
      grad_outputs=torch.ones_like(y),
      create_graph=True
  )[0]
  eqn1_loss=torch.mean((dx_dt-y)**2)
  eqn2_loss=torch.mean((dy_dt+x)**2)
  eqn_loss=eqn1_loss+eqn2_loss
  t_bc=torch.tensor([0],dtype=torch.float32)
  output=model_0(t_bc)
  x_bc=output[0]
  y_bc=output[1]
  bc_loss=torch.mean((x_bc-1)**2+(y_bc)**2)
  loss=eqn_loss+bc_loss
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  with torch.inference_mode():
    epoch_cnt.append(epoch)
    eqn1_loss_epoch.append(eqn1_loss)
    eqn2_loss_epoch.append(eqn2_loss)
    bc_loss_epoch.append(bc_loss)
    loss_epoch.append(loss)
    
  with torch.inference_mode():
  plt.plot(epoch_cnt,eqn1_loss_epoch,label="eqn1_loss")
  plt.plot(epoch_cnt,eqn2_loss_epoch,label="eqn2_loss")
  plt.plot(epoch_cnt,bc_loss_epoch,label="bc_loss")
  plt.plot(epoch_cnt,loss_epoch,label="loss")
  plt.legend()

 t_test=torch.rand(300,1)*torch.pi*2
 output_pred=model_0(t_test)
 x_pred=output_pred[:,0:1]
 y_pred=output_pred[:,1:2]
 x_actual=torch.cos(t_test)
 y_actual=-1*torch.sin(t_test)
 with torch.inference_mode():
  plt.scatter(t_test,x_pred,label="x_predicted")
  plt.scatter(t_test,y_pred,label="y_predicted")
  plt.scatter(t_test,x_actual,label="x_actual")
  plt.scatter(t_test,y_actual,label="y_actual")  
  plt.legend()


