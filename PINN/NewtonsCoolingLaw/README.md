# PINN — Newton's Law of Cooling

A lightweight PyTorch implementation of a **Physics-Informed Neural Network (PINN)** that learns to solve the ordinary differential equation (ODE) for **Newton's Law of Cooling** without relying on labeled simulation data.

---

##  Problem Overview

The temperature $T(t)$ of an object cooling in an environment at ambient temperature $T_{\text{ambient}}$ satisfies the first-order differential equation:

$$\frac{dT}{dt} = -k (T - T_{\text{ambient}})$$

### System Parameters
* **Ambient Temperature ($T_{\text{ambient}}$):** $27^\circ\text{C}$
* **Initial Temperature ($T(0)$):** $250^\circ\text{C}$
* **Cooling Rate Constant ($k$):** $0.45\text{ s}^{-1}$

### Analytical Solution
$$T(t) = T_{\text{ambient}} + (T(0) - T_{\text{ambient}}) e^{-kt}$$

---

##  How It Works

Instead of training on labeled dataset points, the neural network optimizes a composite loss function enforced by two physics constraints:

$$\text{Loss}_{\text{Total}} = \text{Loss}_{\text{Physics}} + \text{Loss}_{\text{BC}}$$

1. **Physics Residual Loss ($\text{Loss}_{\text{Physics}}$):** Evaluates how well the network satisfies the ODE across random time points using automatic differentiation (`torch.autograd.grad`):
   $$\text{Loss}_{\text{Physics}} = \frac{1}{N} \sum \left( \frac{d\hat{T}}{dt} - k(T_{\text{ambient}} - \hat{T}) \right)^2$$
2. **Boundary Condition Loss ($\text{Loss}_{\text{BC}}$):** Forces the model to satisfy the initial temperature at $t = 0$:
   $$\text{Loss}_{\text{BC}} = \left( \hat{T}(0) - 250 \right)^2$$

---

##  Model Architecture

* **Input:** $t$ (Time step)
* **Output:** $\hat{T}(t)$ (Predicted Temperature)
* **Network Structure:** Fully Connected Neural Network (`1` $\to$ `32` $\to$ `64` $\to$ `1`)
* **Activation Function:** `nn.Tanh()` (enables smooth higher-order autograd derivatives)
* **Optimizer:** Adam ($\text{lr} = 0.001$)

---
