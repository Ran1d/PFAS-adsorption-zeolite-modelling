# PFAS Adsorption on Zeolites – Computational Modelling Suite

**Author:** Rand Ahmad Bushnaq  
**Purpose:** Independent project to demonstrate adsorption modelling skills for PhD application (TU Delft – Zeolites for PFAS Removal)

## 🔬 What is this?

An interactive, browser‑based modelling suite for PFAS (PFOA) adsorption on H‑ZSM‑5 zeolite. All computations run locally in your browser – no backend required.

### Implemented models

- **Isotherms:** Langmuir & Freundlich – fitted via nonlinear least‑squares (Levenberg‑Marquardt)
- **Kinetics:** Pseudo‑first‑order (PFO) & pseudo‑second‑order (PSO) – includes half‑life estimation
- **Breakthrough curve:** Thomas analytical model – predicts column service time
- **Sensitivity analysis:** One‑factor‑at‑a‑time (flow rate, bed mass, C₀) + tornado chart
- **Numerical 1D ADE:** Two finite‑difference solvers compared side‑by‑side
  - *Linearised retardation* (local Langmuir slope approximation)
  - *Fully‑coupled nonlinear LDF* (explicit solid loading q(x,t) per cell)

## 🚀
