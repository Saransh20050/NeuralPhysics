# 1D Heat Equation using Physics-Informed Neural Networks

This project implements a Physics-Informed Neural Network (PINN) in PyTorch to solve the one-dimensional Heat Equation. Instead of relying on labeled training data, the network learns the solution by minimizing the governing partial differential equation together with the prescribed initial and boundary conditions.

---

### Governing Equation

The one-dimensional heat equation is given by:

$$ \frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}, \qquad x \in [0,1], \quad t \in [0,1] $$

where:
* $u(x,t)$ is the temperature distribution.
* $\alpha = 1$ is the thermal diffusivity.

#### Initial Condition
$$ u(x,0) = \sin(\pi x) $$

#### Boundary Conditions
$$ u(0,t) = 0 $$
$$ u(1,t) = 0 $$

---

### Exact Analytical Solution

The analytical solution to this system is:

$$ u(x,t) = e^{-\pi^2 t}\sin(\pi x) $$

---

### PINN Formulation

The neural network takes both position and time as inputs and predicts the temperature:

$$\text{Input: } (x, t) \longrightarrow \text{Output: } u(x,t)$$

The total training loss consists of:
1. **Physics Loss** derived from the PDE residual
2. **Initial Condition Loss**
3. **Boundary Condition Loss**

---

### Results

The trained PINN successfully learns the spatio-temporal temperature distribution and closely matches the analytical solution across the entire domain.
