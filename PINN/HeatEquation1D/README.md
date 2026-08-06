# 1D Heat Equation using Physics-Informed Neural Networks

This project implements a **Physics-Informed Neural Network (PINN)** in PyTorch to solve the one-dimensional **Heat Equation**. Instead of relying on labeled training data, the network learns the solution by minimizing the governing partial differential equation together with the prescribed initial and boundary conditions.

## Governing Equation

The one-dimensional heat equation is given by

$$
\frac{\partial u}{\partial t}
=
\alpha
\frac{\partial^2 u}{\partial x^2},
\qquad
x \in [0,1],\;
t \in [0,1]
$$

where

- $u(x,t)$ is the temperature distribution,
- $\alpha = 1$ is the thermal diffusivity.

---

## Initial Condition

$$
u(x,0)=\sin(\pi x)
$$

---

## Boundary Conditions

$$
u(0,t)=0
$$

$$
u(1,t)=0
$$

---

## Exact Analytical Solution

The analytical solution of the problem is

$$
u(x,t)
=
e^{-\pi^2 t}\sin(\pi x).
$$

---

## PINN Formulation

The neural network takes both **position** and **time** as inputs and predicts the temperature.

```
Input : (x, t)

↓

Output : u(x,t)
```

The total training loss consists of

- Physics loss from the Heat Equation
- Initial condition loss
- Boundary condition loss

---

## Result

The trained PINN successfully learns the spatio-temporal temperature distribution and closely matches the analytical solution across the entire domain.

---

*Part of the **NeuralPhysics** repository exploring Scientific Machine Learning.*
