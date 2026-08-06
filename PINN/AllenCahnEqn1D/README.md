# 1D Allen–Cahn Equation using Physics-Informed Neural Networks

This project implements a **Physics-Informed Neural Network (PINN)** in PyTorch to solve the one-dimensional **Allen–Cahn equation**, a non-linear reaction–diffusion partial differential equation (PDE) commonly used to model phase separation and interface dynamics in materials science.

Instead of relying on labeled simulation data, the neural network directly learns the spatio-temporal solution by minimizing the PDE residual alongside initial and boundary condition errors.

---

## Governing Equation

The 1D Allen–Cahn equation is defined as:

$$\frac{\partial u}{\partial t} = \epsilon^2 \frac{\partial^2 u}{\partial x^2} + u - u^3, \qquad x \in [-1, 1], \quad t \in [0, 1]$$

where the interfacial width parameter is set to:

$$\epsilon = 0.1$$

---

## Initial and Boundary Conditions

### Initial Condition
$$u(x,0) = \sin(\pi x)$$

### Boundary Conditions
Fixed Dirichlet boundary conditions at the spatial domain endpoints:
$$u(-1,t) = 0, \qquad u(1,t) = 0$$

---

## PINN Formulation

The neural network takes spatial position $x$ and temporal coordinate $t$ as inputs and predicts the continuous scalar field $u(x,t)$:

$$\text{Input: } (x, t) \longrightarrow \text{Output: } u(x,t)$$

### Objective Function
The total loss is optimized using automatic differentiation to calculate spatial and temporal derivatives directly from the network parameters:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{physics}} + \mathcal{L}_{\text{ic}} + \mathcal{L}_{\text{bc}}$$

* **Physics Loss ($\mathcal{L}_{\text{physics}}$):** Minimizes the PDE residual $\left(\frac{\partial u}{\partial t} - \epsilon^2 \frac{\partial^2 u}{\partial x^2} - u + u^3\right)^2$.
* **Initial Condition Loss ($\mathcal{L}_{\text{ic}}$):** Enforces $u(x,0) = \sin(\pi x)$.
* **Boundary Condition Loss ($\mathcal{L}_{\text{bc}}$):** Enforces Dirichlet boundaries at $x = \pm 1$.

---

## Results

* The trained PINN successfully captures the phase separation dynamics dictated by the non-linear reaction term $u - u^3$ while upholding energy stability bounds.
* Demonstrates the applicability of PINNs to stiff, non-linear reaction–diffusion systems without requiring traditional discretization mesh grids.

---
