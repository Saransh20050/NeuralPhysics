# Nonlinear Schrödinger Equation using Physics-Informed Neural Networks

This project implements a **Physics-Informed Neural Network (PINN)** in PyTorch to solve the one-dimensional **Nonlinear Schrödinger Equation (NLS)**, a nonlinear complex-valued partial differential equation that models wave propagation in nonlinear media and quantum systems.

Since the solution is complex-valued, the neural network simultaneously predicts the **real** and **imaginary** components of the wave function while satisfying the governing equation, initial condition, and boundary conditions.

---

## Table of Contents
- [Governing Equation](#governing-equation)
- [Initial Conditions](#initial-conditions)
- [Boundary Conditions](#boundary-conditions)
- [PINN Architecture & Formulation](#pinn-architecture--formulation)
- [Loss Function Definition](#loss-function-definition)
- [Results](#results)

---

## Governing Equation

The 1D Nonlinear Schrödinger Equation is defined as:

$$i \frac{\partial h}{\partial t} + \frac{1}{2} \frac{\partial^2 h}{\partial x^2} + |h|^2 h = 0$$

Decomposing the complex wave function $h(x,t)$ into real and imaginary components:

$$h(x,t) = u(x,t) + i \, v(x,t)$$

where:
* $x \in [-5, 5]$
* $t \in \left[0, \frac{\pi}{2}\right]$

Substituting $h(x,t) = u + iv$ into the governing PDE yields two coupled real-valued partial differential equations:

* **Real Residual ($\mathcal{f}_u$):**  
  $$-\frac{\partial v}{\partial t} + \frac{1}{2} \frac{\partial^2 u}{\partial x^2} + (u^2 + v^2)u = 0$$

* **Imaginary Residual ($\mathcal{f}_v$):**  
  $$\frac{\partial u}{\partial t} + \frac{1}{2} \frac{\partial^2 v}{\partial x^2} + (u^2 + v^2)v = 0$$

---

## Initial Conditions

At $t = 0$, the system is initialized with a bright soliton solution:

$$h(x,0) = 2 \, \text{sech}(x)$$

Decomposed into real and imaginary parts:

$$\begin{aligned}
u(x,0) &= 2 \, \text{sech}(x) \\
v(x,0) &= 0
\end{aligned}$$

---

## Boundary Conditions

At both spatial boundaries ($x = -5$ and $x = 5$):

$$\begin{aligned}
u(-5,t) &= u(5,t) = 0 \\
v(-5,t) &= v(5,t) = 0
\end{aligned}$$

---

## PINN Architecture & Formulation

The neural network acts as a universal function approximator, mapping space-time coordinates to the real and imaginary states.
