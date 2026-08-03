# Simple Harmonic Oscillator using PINNs

This project implements a **Physics-Informed Neural Network (PINN)** in PyTorch to solve the classical **Simple Harmonic Oscillator (SHO)** formulated as a system of coupled first-order ordinary differential equations.

The neural network learns the solution by minimizing the residuals of the governing equations while satisfying the prescribed initial conditions, without using labeled training data.

## Governing Equations

```text
dx/dt = y

dy/dt = -x
```

for

```text
t ∈ [0, 2π]
```

### Initial Conditions

```text
x(0) = 1

y(0) = 0
```

## Exact Solution

```text
x(t) = cos(t)

y(t) = -sin(t)
```
The total training loss consists of:

- Physics loss from both governing equations
- Initial condition loss

## Result

The trained PINN successfully learns both oscillatory trajectories and closely matches the analytical solution.
