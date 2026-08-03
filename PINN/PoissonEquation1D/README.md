# 1D Poisson Equation using PINNs

This project implements a **Physics-Informed Neural Network (PINN)** in PyTorch to solve the one-dimensional Poisson equation.

The network is trained by minimizing the PDE residual and enforcing the boundary conditions, without using labeled solution data. Automatic differentiation is used to compute the required derivatives during training.

## Equation

d²y/dx² + π² sin(πx) = 0,  x ∈ [-1, 1]

Boundary Conditions:

y(-1) = 0

y(1) = 0

Analytical Solution:

y(x) = sin(πx)
---

*Part of the **NeuralPhysics** repository exploring Scientific Machine Learning.*
