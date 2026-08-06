# 1D Wave Equation using Physics-Informed Neural Networks

This project implements a **Physics-Informed Neural Network (PINN)** in PyTorch to solve the one-dimensional **Wave Equation**. The network learns the displacement field by minimizing the governing partial differential equation together with the prescribed initial and boundary conditions, without using labeled solution data.

---

### Governing Equation

The one-dimensional wave equation is given by:

$$\frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2}, \qquad x \in [0,1], \quad t \in [0,1]$$

where:
* $u(x,t)$ is the displacement.
* $c = 1$ is the wave speed.

#### Initial Conditions
* **Initial Displacement:** $u(x,0) = \sin(\pi x)$
* **Initial Velocity:** $\frac{\partial u}{\partial t}(x,0) = 0$

#### Boundary Conditions
Fixed-end boundary conditions:
* $u(0,t) = 0$
* $u(1,t) = 0$

---

### Exact Analytical Solution

The exact analytical solution is:

$$u(x,t) = \sin(\pi x)\cos(\pi t)$$

---

### PINN Formulation

The neural network takes both **position** and **time** as inputs and predicts the displacement:

$$\text{Input: } (x, t) \longrightarrow \text{Output: } u(x,t)$$

The total training loss consists of:
1. **Physics Loss** derived from the Wave Equation PDE residual
2. **Initial Displacement Loss**
3. **Initial Velocity Loss**
4. **Boundary Condition Loss**

---

### Results

The trained PINN successfully learns the oscillatory spatio-temporal solution and closely matches the analytical solution throughout the computational domain.

Heatmap visualizations of the predicted solution, analytical solution, and absolute error are included for qualitative comparison.

