# Coupled Linear ODE System using PINNs

This project implements a **Physics-Informed Neural Network (PINN)** in PyTorch to solve a system of two coupled first-order ordinary differential equations.

Instead of relying on labeled training data, the neural network learns the solution by minimizing the residuals of both governing equations while satisfying the prescribed initial conditions.

## Governing Equations

dx/dt + 2x + y = 0

dy/dt + x + 2y = 0

for

t ∈ [0, 5]

### Initial Conditions

x(0) = 1

y(0) = 0

## Exact Solution

x(t) = ½e⁻ᵗ + ½e⁻³ᵗ

y(t) = −½e⁻ᵗ + ½e⁻³ᵗ

## PINN Formulation

The network takes **time (t)** as input and simultaneously predicts both state variables:

```
Input:  t
Output: [x(t), y(t)]
```

The total training loss consists of:

- Physics loss from both differential equations
- Initial condition loss

## Result

The trained PINN successfully learns both solution trajectories and closely matches the analytical solution.

---
