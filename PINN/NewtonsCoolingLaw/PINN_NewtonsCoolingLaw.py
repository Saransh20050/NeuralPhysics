import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
class ann(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.l1=torch.nn.Linear(1,32)
    self.l2=torch.nn.Linear(32,64)
    self.l3=torch.nn.Linear(64,1)
    self.tanh=torch.nn.Tanh()
  def forward(self,x:torch.Tensor)->torch.Tensor:
    return self.l3(self.tanh(self.l2(self.tanh(self.l1(x)))))

model_0=ann()
optimizer=torch.optim.Adam(model_0.parameters(),lr=0.001)


epochs=6000
epoch_cnt=[]
phy_loss_epoch=[]
bc_loss_epoch=[]
loss_epoch=[]
k=0.45
y0=27
for epoch in range(epochs):
  x=torch.rand(300,1)*10
  x.requires_grad=True
  y=model_0(x)
  dy_dx=torch.autograd.grad(
      outputs=y,
      inputs=x,
      grad_outputs=torch.ones_like(y),
      create_graph=True
  )[0]
  phy_loss=dy_dx-k*(y0-y)
  phy_loss=torch.mean(phy_loss**2)
  x1=torch.tensor([0],dtype=torch.float32)
  bc_loss=(model_0(x1)-250)**2
  loss=phy_loss+bc_loss
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  with torch.inference_mode():
    epoch_cnt.append(epoch)
    phy_loss_epoch.append(phy_loss)
    bc_loss_epoch.append(bc_loss)
    loss_epoch.append(loss)



with torch.inference_mode():
  plt.plot(epoch_cnt,phy_loss_epoch,label="phy_loss")
  plt.plot(epoch_cnt,bc_loss_epoch,label="bc_loss")
  plt.plot(epoch_cnt,loss_epoch,label="loss")
  plt.legend()



x_test=torch.rand(300,1)*10
y_pred=model_0(x_test)
to_raise=-0.45*x_test
y_actual=27+223*(torch.e**to_raise)
with torch.inference_mode():
 plt.scatter(x_test,y_pred,c="r",label="predicted")
 plt.scatter(x_test,y_actual,c="g",label="actual")
 plt.legend()

 
