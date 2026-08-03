# 1D Poisson Equation using PINNs

This project implements a **Physics-Informed Neural Network (PINN)** in PyTorch to solve the one-dimensional Poisson equation.

The network is trained by minimizing the PDE residual and enforcing the boundary conditions, without using labeled solution data. Automatic differentiation is used to compute the required derivatives during training.

## Equation

\[
\frac{d^2y}{dx^2} + \pi^2 \sin(\pi x) = 0,
\qquad x \in [-1,1]
\]

with

\[
y(-1)=0,\qquad y(1)=0.
\]

## Result

The learned solution closely matches the analytical solution,

\[
y(x)=\sin(\pi x).
\]

---

*Part of the **NeuralPhysics** repository exploring Scientific Machine Learning.*
