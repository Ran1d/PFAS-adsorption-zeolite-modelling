# PFAS Adsorption on Zeolites – Computational Modelling Suite

**Author:** Rand Ahmad Bushnaq  
**Purpose:** Independent project to demonstrate adsorption modelling skills for PhD application (TU Delft – Zeolites for PFAS Removal)

##  What is this?

An interactive, browser‑based modelling suite for PFAS (PFOA) adsorption on H‑ZSM‑5 zeolite. All computations run locally in your browser – no backend required.

### Implemented models

- **Isotherms:** Langmuir & Freundlich – fitted via nonlinear least‑squares (Levenberg‑Marquardt)
- **Kinetics:** Pseudo‑first‑order (PFO) & pseudo‑second‑order (PSO) – includes half‑life estimation
- **Breakthrough curve:** Thomas analytical model – predicts column service time
- **Sensitivity analysis:** One‑factor‑at‑a‑time (flow rate, bed mass, C₀) + tornado chart
- **Numerical 1D ADE**: Two finite-difference solvers - linearised retardation vs. fully-coupled nonlinear LDF:
  Solver A: Linearised retardation (local equilibrium approximation) Solver B: Fully-coupled nonlinear LDF (explicit q field)

<img width="593" height="1714" alt="image" src="https://github.com/user-attachments/assets/82ee2761-c453-43cb-93f3-8bfea0214cbe" />

<img width="616" height="1733" alt="image" src="https://github.com/user-attachments/assets/58871a8c-1c04-41a4-8419-f471601c4c0b" />







