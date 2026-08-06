# Allen–Cahn Equation using Physics-Informed Neural Networks

This project implements a **Physics-Informed Neural Network (PINN)** in PyTorch to solve the one-dimensional **Allen–Cahn equation**, a nonlinear reaction–diffusion partial differential equation commonly used to model phase separation and interface dynamics in materials.

Instead of relying on labeled training data, the neural network learns the solution by minimizing the residual of the governing equation while satisfying the prescribed initial and boundary conditions.

## Governing Equation

The Allen–Cahn equation is

$$
\frac{\partial u}{\partial t}
=
\epsilon^2
\frac{\partial^2 u}{\partial x^2}
+
u-u^3,
$$

where

$$
x \in [-1,1], \qquad
t \in [0,1],
$$

and

$$
\epsilon = 0.1.
$$

---

## Initial Condition

$$
u(x,0)=\sin(\pi x).
$$

---

## Boundary Conditions

$$
u(-1,t)=0
$$

$$
u(1,t)=0.
$$

---

## PINN Formulation

The neural network receives both spatial and temporal coordinates as input and predicts the scalar field.

```
Input : (x, t)

↓

Output : u(x,t)
```

The total training loss consists of

- Physics loss from the Allen–Cahn equation
- Initial condition loss
- Boundary condition loss

Automatic differentiation is used to compute the required spatial and temporal derivatives directly from the neural network.

---

## Result

The trained PINN successfully satisfies the governing equation while enforcing the prescribed initial and boundary conditions, demonstrating the application of PINNs to nonlinear reaction–diffusion systems.

---
