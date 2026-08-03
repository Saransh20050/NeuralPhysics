import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ann(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.l1=torch.nn.Linear(1,32)
    self.l2=torch.nn.Linear(32,64)
    self.l3=torch.nn.Linear(64,2)
    self.tanh=torch.nn.Tanh()
  def forward(self,x:torch.Tensor)->torch.Tensor:
    return self.l3(self.tanh(self.l2(self.tanh(self.l1(x)))))
model_0=ann()
optimizer=torch.optim.Adam(params=model_0.parameters(),lr=0.001)

epochs=300
epoch_cnt=[]
eqn1_loss_epoch=[]
eqn2_loss_epoch=[]
bc_loss_epoch=[]
for epoch in range(epochs):
  t=torch.rand(300,1)*5
  t.requires_grad=True
  out=model_0(t)
  x=out[:,0:1]
  y=out[:,1:2]
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
  eqn1_loss=torch.mean((dx_dt+2*x+y)**2)
  eqn2_loss=torch.mean((dy_dt+x+2*y)**2)
  eqn_loss=eqn1_loss+eqn2_loss  
  t_bc=torch.tensor([0],dtype=torch.float32)
  output_bc=model_0(t_bc)
  x_bc=output_bc[0]
  y_bc=output_bc[1]
  bc_loss=torch.mean((x_bc-1)**2)+torch.mean((y_bc)**2)
  loss=eqn_loss+bc_loss
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  with torch.inference_mode():
   epoch_cnt.append(epoch)
   eqn1_loss_epoch.append(eqn1_loss)
   eqn2_loss_epoch.append(eqn2_loss)
   bc_loss_epoch.append(bc_loss)

  
with torch.inference_mode():
  plt.plot(epoch_cnt,eqn1_loss_epoch,label="eqn1_loss")
  plt.plot(epoch_cnt,eqn2_loss_epoch,label="eqn2_loss")
  plt.plot(epoch_cnt,bc_loss_epoch,label="bc_loss")
  plt.legend()



t_test=torch.rand(300,1)*5
out_test=model_0(t_test)
x_test=out_test[:,0:1]
y_test=out_test[:,1:2]
x_actual=0.5*(torch.e**(-t_test))+0.5*(torch.e**(-3*t_test))
y_actual=-0.5*(torch.e**(-t_test))+0.5*(torch.e**(-3*t_test))
with torch.inference_mode():
  plt.scatter(t_test,x_test,c="r",label="x_pred")
  plt.scatter(t_test,y_test,c="y",label="y_pred")
  plt.scatter(t_test,x_actual,c="g",label="x_actual")
  plt.scatter(t_test,y_actual,c="b",label="y_actual")
  plt.legend()



