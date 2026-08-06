# 1D Burgers' Equation using Physics-Informed Neural Networks

This project implements a **Physics-Informed Neural Network (PINN)** in PyTorch to solve the one-dimensional **viscous Burgers' equation**, a classical nonlinear partial differential equation (PDE) widely used as a benchmark for Scientific Machine Learning (SciML).

Unlike linear PDEs (such as the Heat or Wave equations), Burgers' equation introduces a nonlinear convection term ($u \frac{\partial u}{\partial x}$), leading to steep gradient formation (shock dynamics) and making optimization significantly more challenging.

---

## Governing Equation

The one-dimensional viscous Burgers' equation is defined as:

$$\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}, \qquad x \in [-1, 1], \quad t \in [0, 1]$$

where the kinematic viscosity parameter is given by:

$$\nu = \frac{0.01}{\pi}$$

---

## Initial and Boundary Conditions

### Initial Condition
$$u(x,0) = -\sin(\pi x)$$

### Boundary Conditions
Fixed Dirichlet boundary conditions at the spatial domain endpoints:
$$u(-1,t) = 0, \qquad u(1,t) = 0$$

---

## PINN Formulation

The neural network takes spatial position $x$ and temporal coordinate $t$ as inputs and predicts the scalar field $u(x,t)$:

$$\text{Input: } (x, t) \longrightarrow \text{Output: } u(x,t)$$

### Total Training Loss
The total loss objective function minimizes:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{physics}} + \lambda_{\text{ic}} \mathcal{L}_{\text{ic}} + \lambda_{\text{bc}} \mathcal{L}_{\text{bc}}$$

* **Physics Loss ($\mathcal{L}_{\text{physics}}$):** Derived from the PDE residual using automatic differentiation.
* **Initial Condition Loss ($\mathcal{L}_{\text{ic}}$):** Enforces $u(x,0) = -\sin(\pi x)$.
* **Boundary Condition Loss ($\mathcal{L}_{\text{bc}}$):** Enforces Dirichlet boundaries at $x = \pm 1$.

> **Note on Optimization:** To prevent gradient pathology caused by steep spatial derivatives near the shock front, the initial condition loss is assigned a higher weight ($\lambda_{\text{ic}} = 10.0$) relative to the residual physics loss.

---

## Results

* The trained PINN successfully learns the non-linear spatio-temporal dynamics governed by Burgers' equation, accurately capturing the shock formation near $x = 0$ while strictly adhering to boundary constraints.
* Demonstrates the capability of PINNs to resolve non-linear transport behavior without relying on labeled simulation grid data.

---
