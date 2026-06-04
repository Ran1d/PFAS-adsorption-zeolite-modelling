"""
Computational Fixed-Bed Adsorption Study for PFAS Removal on Zeolites
======================================================================
Author : Rand Ahmad Bushnaq
Date   : 2026
Sources: Li et al. (2026) Water Research 288, 124645
         Markhali et al. (2026) Clean Technologies 8(1), 21

Sections
--------
1. Literature data
2. Isotherm modelling       — Langmuir & Freundlich (nonlinear least-squares)
3. Kinetic modelling        — PFO & PSO
4. Breakthrough simulation  — Thomas analytical model
5. Sensitivity analysis     — one-factor-at-a-time on Thomas model
6. Numerical 1-D ADE        — Solver A (linearised) & Solver B (nonlinear LDF)

Dependencies
------------
    pip install numpy scipy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

plt.rcParams.update({
    "figure.facecolor": "#0a0e1a",
    "axes.facecolor":   "#111827",
    "axes.edgecolor":   "#1e2d3d",
    "axes.labelcolor":  "#94a3b8",
    "xtick.color":      "#64748b",
    "ytick.color":      "#64748b",
    "text.color":       "#e2e8f0",
    "grid.color":       "#1e293b",
    "grid.linestyle":   "--",
    "grid.alpha":       0.6,
    "legend.facecolor": "#0f1929",
    "legend.edgecolor": "#1e2d3d",
    "font.family":      "sans-serif",
    "font.size":        9,
})

A1 = "#00d4ff"   # cyan
A2 = "#7c3aed"   # purple
A3 = "#10b981"   # green
A4 = "#f59e0b"   # amber
A5 = "#ef4444"   # red

# =============================================================================
# 1. LITERATURE DATA
# =============================================================================

# Equilibrium isotherm  — PFOA on zeolite, 25 C, pH 7
Ce = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0])  # mg/L
qe = np.array([2.1, 3.8, 6.5, 11.2, 16.4, 22.1, 26.8, 29.1, 30.5, 31.2])  # mg/g

# Batch kinetics  — C0 = 10 mg/L, 25 C
t_kin = np.array([0, 5, 10, 20, 30, 60, 90, 120, 180, 240, 300])  # min
qt    = np.array([0, 3.2, 5.8, 8.9, 11.2, 13.6, 14.8, 15.3, 15.7, 15.9, 16.0])  # mg/g

# Breakthrough curve  — m = 2 g, Q = 5 mL/min, C0 = 1 mg/L
t_bt  = np.array([0, 20, 40, 60, 80, 100, 120, 140, 160, 180,
                  200, 220, 240, 260, 280, 300])  # min
ratio = np.array([0.00, 0.01, 0.02, 0.04, 0.08, 0.15, 0.25, 0.38, 0.52, 0.65,
                  0.76, 0.84, 0.90, 0.94, 0.97, 0.98])  # C/C0

# Fixed-bed operating conditions
C0_bed = 1.0    # mg/L  inlet concentration
Q_bed  = 5.0    # mL/min flow rate
m_bed  = 2.0    # g     bed mass

# =============================================================================
# 2. ISOTHERM MODELLING
# =============================================================================

def langmuir(Ce, qmax, KL):
    """Langmuir isotherm: q = qmax*KL*Ce / (1 + KL*Ce)"""
    return (qmax * KL * Ce) / (1.0 + KL * Ce)

def freundlich(Ce, KF, n):
    """Freundlich isotherm: q = KF * Ce^(1/n)"""
    return KF * Ce ** (1.0 / n)

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot

def mean_abs_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# --- Langmuir fit ---
popt_L, _ = curve_fit(langmuir, Ce, qe, p0=[32.0, 0.1], maxfev=10000,
                       bounds=([0, 0], [np.inf, np.inf]))
qmax_L, KL = popt_L
qe_lang = langmuir(Ce, qmax_L, KL)
R2_L  = r_squared(qe, qe_lang)
MAE_L = mean_abs_error(qe, qe_lang)
RL    = 1.0 / (1.0 + KL * Ce.max())   # separation factor at highest C0

# --- Freundlich fit ---
popt_F, _ = curve_fit(freundlich, Ce, qe, p0=[5.0, 2.0], maxfev=10000,
                       bounds=([0, 1e-3], [np.inf, np.inf]))
KF, n_F = popt_F
qe_freun = freundlich(Ce, KF, n_F)
R2_F  = r_squared(qe, qe_freun)
MAE_F = mean_abs_error(qe, qe_freun)

Ce_smooth = np.linspace(0.1, 105, 300)

print("=" * 60)
print("SECTION 2 — ISOTHERM MODELLING")
print("=" * 60)
print(f"  Langmuir  : qmax = {qmax_L:.3f} mg/g  KL = {KL:.5f} L/mg"
      f"  R2 = {R2_L:.4f}  MAE = {MAE_L:.3f} mg/g")
print(f"  Freundlich: KF   = {KF:.3f}        n  = {n_F:.4f}"
      f"         R2 = {R2_F:.4f}  MAE = {MAE_F:.3f} mg/g")
print(f"  Separation factor RL (at C0_max) = {RL:.4f}  (favourable: 0 < RL < 1)")

# =============================================================================
# 3. KINETIC MODELLING
# =============================================================================

def pfo(t, qe_val, k1):
    """Pseudo-first-order: qt = qe*(1 - exp(-k1*t))"""
    return qe_val * (1.0 - np.exp(-k1 * t))

def pso(t, qe_val, k2):
    """Pseudo-second-order: qt = k2*qe^2*t / (1 + k2*qe*t)"""
    return (k2 * qe_val**2 * t) / (1.0 + k2 * qe_val * t)

# --- PFO fit ---
popt_PFO, _ = curve_fit(pfo, t_kin, qt, p0=[16.0, 0.02], maxfev=10000,
                          bounds=([0, 0], [np.inf, np.inf]))
qe_PFO, k1 = popt_PFO
qt_pfo  = pfo(t_kin, qe_PFO, k1)
R2_PFO  = r_squared(qt, qt_pfo)
t_half  = np.log(2) / k1   # PFO half-life

# --- PSO fit ---
popt_PSO, _ = curve_fit(pso, t_kin, qt, p0=[16.0, 0.002], maxfev=10000,
                          bounds=([0, 0], [np.inf, np.inf]))
qe_PSO, k2 = popt_PSO
qt_pso  = pso(t_kin, qe_PSO, k2)
R2_PSO  = r_squared(qt, qt_pso)

t_smooth = np.linspace(0, 310, 400)

print("\n" + "=" * 60)
print("SECTION 3 — KINETIC MODELLING")
print("=" * 60)
print(f"  PFO: qe = {qe_PFO:.3f} mg/g  k1 = {k1:.5f} 1/min"
      f"  R2 = {R2_PFO:.4f}  t_half = {t_half:.1f} min")
print(f"  PSO: qe = {qe_PSO:.3f} mg/g  k2 = {k2:.6f} g/(mg·min)"
      f"  R2 = {R2_PSO:.4f}")
print(f"  --> PSO fits better; chemisorption is rate-controlling.")

# =============================================================================
# 4. BREAKTHROUGH SIMULATION — THOMAS MODEL
# =============================================================================

def thomas_model(t, kTh, q0, C0, Q_mL, m):
    """
    Thomas model: C/C0 = 1 / (1 + exp(kTh/Q * (q0*m - C0*Q*t)))
    Units: kTh [mL/(min·mg)], Q [mL/min], q0 [mg/g], m [g], C0 [mg/mL], t [min]
    """
    Q_mL_min = Q_mL
    exp_arg = (kTh / Q_mL_min) * (q0 * m - C0 * Q_mL_min * t)
    return 1.0 / (1.0 + np.exp(exp_arg))

# C0 in mg/mL for unit consistency with Q in mL/min
C0_mL = C0_bed / 1000.0   # 1 mg/L = 0.001 mg/mL

def thomas_fit_wrapper(t, kTh, q0):
    return thomas_model(t, kTh, q0, C0_mL, Q_bed, m_bed)

popt_Th, _ = curve_fit(thomas_fit_wrapper, t_bt, ratio,
                         p0=[0.02, 31.0], maxfev=20000,
                         bounds=([0, 0], [np.inf, np.inf]))
kTh, q0_Th = popt_Th
ratio_sim   = thomas_fit_wrapper(t_bt, kTh, q0_Th)
R2_Th       = r_squared(ratio, ratio_sim)
MAE_Th      = mean_abs_error(ratio, ratio_sim)

t_th_smooth = np.linspace(0, 310, 500)
ratio_smooth = thomas_fit_wrapper(t_th_smooth, kTh, q0_Th)

# Breakthrough and exhaustion times
tb_idx = np.argmax(ratio_sim >= 0.10)
te_idx = np.argmax(ratio_sim >= 0.90)
t_b = t_bt[tb_idx] if ratio_sim[tb_idx] >= 0.10 else None
t_e = t_bt[te_idx] if ratio_sim[te_idx] >= 0.90 else None

print("\n" + "=" * 60)
print("SECTION 4 — THOMAS MODEL BREAKTHROUGH")
print("=" * 60)
print(f"  kTh = {kTh:.4e} mL/(min·mg)   q0 = {q0_Th:.2f} mg/g")
print(f"  R2  = {R2_Th:.4f}             MAE = {MAE_Th:.4f} C/C0")
print(f"  Breakthrough t_b (C/C0=0.10) ~ {t_b} min")
print(f"  Exhaustion   t_e (C/C0=0.90) ~ {t_e} min")

# =============================================================================
# 5. SENSITIVITY ANALYSIS
# =============================================================================

def thomas_tb(kTh_val, q0_val, C0_val, Q_val, m_val):
    """
    Analytical breakthrough time from Thomas model at C/C0 = 0.1.
    Solving: exp(kTh/Q*(q0*m - C0*Q*t)) = 9
    => t = (q0*m - Q*ln(9)/kTh) / (C0*Q)
    Units: Q [mL/min], C0 [mg/mL]
    """
    C0v = C0_val / 1000.0   # mg/L -> mg/mL
    num = q0_val * m_val - Q_val * np.log(9.0) / kTh_val
    return max(0.0, num / (C0v * Q_val))

Q_range  = np.array([2, 3, 4, 5, 6, 7, 8, 10], dtype=float)
m_range  = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0], dtype=float)
C0_range = np.array([0.2, 0.5, 1.0, 2.0, 5.0, 10.0], dtype=float)

tb_Q  = [thomas_tb(kTh, q0_Th, C0_bed, Q, m_bed)  for Q  in Q_range]
tb_m  = [thomas_tb(kTh, q0_Th, C0_bed, Q_bed, m)  for m  in m_range]
tb_C0 = [thomas_tb(kTh, q0_Th, C0, Q_bed, m_bed)  for C0 in C0_range]

print("\n" + "=" * 60)
print("SECTION 5 — SENSITIVITY ANALYSIS")
print("=" * 60)
tb_base = thomas_tb(kTh, q0_Th, C0_bed, Q_bed, m_bed)
print(f"  Baseline t_b (Q=5, m=2, C0=1) = {tb_base:.1f} min")
print(f"  Q range 2-10:  t_b {min(tb_Q):.1f} – {max(tb_Q):.1f} min")
print(f"  m range 0.5-5: t_b {min(tb_m):.1f} – {max(tb_m):.1f} min")
print(f"  C0 range 0.2-10: t_b {min(tb_C0):.1f} – {max(tb_C0):.1f} min")

# =============================================================================
# 6. NUMERICAL 1-D ADE
# =============================================================================
#
# Shared column parameters
L     = 10.0    # cm   bed length
nx    = 60      # number of grid nodes
dx    = L / nx
v     = 0.5     # cm/min  interstitial velocity
C0_n  = 1.0     # mg/L (normalised to 1 for C/C0 output)
qm    = 31.2    # mg/g  Langmuir qmax
KL_n  = 0.085   # L/mg
rho_b = 0.5     # g/mL  bulk density
eps   = 0.4     # void fraction
ks    = 0.15    # 1/min  LDF mass-transfer coefficient
t_max = 320.0   # min
snaps = [40, 80, 120, 160, 200, 240, 280, 320]   # profile snapshots (min)

# --------------------------------------------------------------------------
# Solver A — Linearised retardation
# --------------------------------------------------------------------------
# Governing equation (single PDE for C only):
#
#   dC/dt = (1/R(C)) * [D * d2C/dx2  -  v * dC/dx]
#
# where R(C) = 1 + (rho_b/eps) * dq*/dC |_{C_local}
#       dq*/dC = qm * KL / (1 + KL*C)^2
#
# q is never stored; local equilibrium is assumed implicitly every timestep.
# --------------------------------------------------------------------------

def solve_linearised(Pe):
    D  = v * L / Pe
    dt = min(0.28 * dx / v, 0.38 * dx**2 / (2 * D + 1e-12))
    n_steps = int(t_max / dt)

    C = np.zeros(nx)
    outlet, profiles = [], {}
    sample = 0

    for step in range(n_steps):
        t = step * dt
        Cn = C.copy()

        for i in range(1, nx - 1):
            dqdC = qm * KL_n / (1.0 + KL_n * C[i]) ** 2
            R    = 1.0 + (rho_b / eps) * dqdC
            adv  = -v * (C[i] - C[i-1]) / dx
            dif  =  D * (C[i+1] - 2*C[i] + C[i-1]) / dx**2
            Cn[i] = np.clip(C[i] + (dt / R) * (adv + dif), 0.0, C0_n)

        Cn[0]    = C0_n
        Cn[nx-1] = Cn[nx-2]      # zero-gradient outlet BC
        C = Cn

        sample += 1
        if sample % 50 == 0:
            outlet.append((round(t, 2), round(float(C[nx-1] / C0_n), 5)))

        for s in snaps:
            if s not in profiles and t >= s:
                x_arr = np.linspace(0, L, nx)
                profiles[s] = (x_arr, C / C0_n)

    return np.array(outlet), profiles


# --------------------------------------------------------------------------
# Solver B — Fully-coupled nonlinear LDF
# --------------------------------------------------------------------------
# Two coupled equations per grid cell:
#
#   Liquid PDE:
#     eps * dC/dt = eps * [D * d2C/dx2 - v * dC/dx]  -  rho_b * dq/dt
#
#   Solid ODE (Linear Driving Force per cell):
#     dq/dt = ks * [q*(C) - q]
#     q*(C) = qm * KL * C / (1 + KL * C)   (Langmuir equilibrium)
#
# C and q are updated in sequence each timestep; dq/dt couples back into
# the liquid mass balance explicitly — no linearisation of the retardation.
# --------------------------------------------------------------------------

def solve_nonlinear(Pe):
    D  = v * L / Pe
    dt = min(0.22 * dx / v, 0.32 * dx**2 / (2 * D + 1e-12), 0.5 / ks)
    n_steps = int(t_max / dt)

    C = np.zeros(nx)
    q = np.zeros(nx)
    outlet, profiles = [], {}
    sample = 0

    for step in range(n_steps):
        t = step * dt

        # 1. Solid ODE — LDF at every cell
        qStar = (qm * KL_n * C) / (1.0 + KL_n * C)
        qn    = np.maximum(0.0, q + dt * ks * (qStar - q))

        # 2. Liquid PDE — explicit dq/dt source term
        dqdt = (qn - q) / dt
        Cn   = C.copy()
        for i in range(1, nx - 1):
            adv  = -v * (C[i] - C[i-1]) / dx
            dif  =  D * (C[i+1] - 2*C[i] + C[i-1]) / dx**2
            Cn[i] = np.clip(
                C[i] + dt * ((adv + dif) - (rho_b / eps) * dqdt[i]),
                0.0, C0_n)

        Cn[0]    = C0_n
        Cn[nx-1] = Cn[nx-2]
        C = Cn
        q = qn

        sample += 1
        if sample % 50 == 0:
            outlet.append((round(t, 2), round(float(C[nx-1] / C0_n), 5)))

        for s in snaps:
            if s not in profiles and t >= s:
                x_arr = np.linspace(0, L, nx)
                profiles[s] = (x_arr, C / C0_n, q / qm)  # also store q/qmax

    return np.array(outlet), profiles


Pe_values = [10, 20, 50]

print("\n" + "=" * 60)
print("SECTION 6 — NUMERICAL 1-D ADE SOLVERS")
print("=" * 60)

all_lin_results = {}
all_nl_results  = {}

for Pe in Pe_values:
    print(f"\n  Pe = {Pe}")
    out_lin, prof_lin = solve_linearised(Pe)
    out_nl,  prof_nl  = solve_nonlinear(Pe)

    all_lin_results[Pe] = (out_lin, prof_lin)
    all_nl_results[Pe]  = (out_nl,  prof_nl)

    # Find breakthrough times
    def find_tb(outlet, threshold):
        hits = outlet[outlet[:, 1] >= threshold]
        return hits[0, 0] if len(hits) > 0 else None

    tb_lin = find_tb(out_lin, 0.10)
    tb_nl  = find_tb(out_nl,  0.10)

    # Max solver divergence
    t_common = np.intersect1d(out_lin[:, 0], out_nl[:, 0])
    lin_interp = np.interp(t_common, out_lin[:, 0], out_lin[:, 1])
    nl_interp  = np.interp(t_common, out_nl[:, 0],  out_nl[:, 1])
    max_diff   = np.max(np.abs(nl_interp - lin_interp))

    D_val = v * L / Pe
    print(f"    D = {D_val:.4f} cm2/min")
    print(f"    t_b Linearised = {tb_lin} min   t_b Nonlinear = {tb_nl} min")
    print(f"    Max |delta C/C0| between solvers = {max_diff:.5f}")

# =============================================================================
# FIGURES
# =============================================================================

fig = plt.figure(figsize=(16, 22))
fig.patch.set_facecolor("#0a0e1a")
gs  = gridspec.GridSpec(5, 2, figure=fig, hspace=0.55, wspace=0.35)

# ── Helper ────────────────────────────────────────────────────────────────────
def fmt_ax(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=8.5)
    ax.set_ylabel(ylabel, fontsize=8.5)
    ax.set_title(title, fontsize=9.5, fontweight="bold", color="#e2e8f0", pad=8)
    ax.grid(True)
    ax.legend(fontsize=8, framealpha=0.4)

# ── 2a. Isotherm curves ───────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
ax.scatter(Ce, qe, color=A3, zorder=5, s=50, label="Experimental data")
ax.plot(Ce_smooth, langmuir(Ce_smooth, qmax_L, KL), color=A1, lw=2,
        label=f"Langmuir (R2={R2_L:.4f})")
ax.plot(Ce_smooth, freundlich(Ce_smooth, KF, n_F), color=A2, lw=2, ls="--",
        label=f"Freundlich (R2={R2_F:.4f})")
fmt_ax(ax, "Ce (mg/L)", "qe (mg/g)", "Equilibrium Isotherms")

# ── 2b. Isotherm residuals ────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
resid_L = qe - langmuir(Ce, qmax_L, KL)
resid_F = qe - freundlich(Ce, KF, n_F)
ax.scatter(Ce, resid_L, color=A1, s=50, label="Langmuir residuals")
ax.scatter(Ce, resid_F, color=A2, s=50, marker="^", label="Freundlich residuals")
ax.axhline(0, color="#64748b", lw=1, ls="--")
fmt_ax(ax, "Ce (mg/L)", "Residual (mg/g)", "Isotherm Residuals")

# ── 3a. Kinetic curves ────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
ax.scatter(t_kin, qt, color=A3, zorder=5, s=50, label="Experimental data")
ax.plot(t_smooth, pfo(t_smooth, qe_PFO, k1), color=A2, lw=2,
        label=f"PFO (R2={R2_PFO:.4f})")
ax.plot(t_smooth, pso(t_smooth, qe_PSO, k2), color=A4, lw=2, ls="--",
        label=f"PSO (R2={R2_PSO:.4f})")
fmt_ax(ax, "Time (min)", "qt (mg/g)", "Adsorption Kinetics")

# ── 3b. Kinetic residuals ─────────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
ax.bar(t_kin - 3, qt - pfo(t_kin, qe_PFO, k1), width=5, color=A2,
       alpha=0.7, label="PFO residuals")
ax.bar(t_kin + 3, qt - pso(t_kin, qe_PSO, k2), width=5, color=A4,
       alpha=0.7, label="PSO residuals")
ax.axhline(0, color="#64748b", lw=1, ls="--")
fmt_ax(ax, "Time (min)", "Residual (mg/g)", "Kinetic Residuals")

# ── 4. Breakthrough — Thomas ──────────────────────────────────────────────────
ax = fig.add_subplot(gs[2, 0])
ax.scatter(t_bt, ratio, color=A5, zorder=5, s=50, label="Experimental")
ax.plot(t_th_smooth, ratio_smooth, color=A3, lw=2,
        label=f"Thomas model (R2={R2_Th:.4f})")
ax.axhline(0.10, color=A1, lw=1, ls=":", alpha=0.8)
ax.axhline(0.90, color=A4, lw=1, ls=":", alpha=0.8)
ax.text(305, 0.12, "t_b", color=A1, fontsize=8)
ax.text(305, 0.92, "t_e", color=A4, fontsize=8)
fmt_ax(ax, "Time (min)", "C/C0", "Thomas Model Breakthrough")

# ── 4b. Breakthrough error ────────────────────────────────────────────────────
ax = fig.add_subplot(gs[2, 1])
err = np.abs(ratio - ratio_sim)
colors_bar = [A5 if e > 0.05 else A3 for e in err]
ax.bar(t_bt, err, width=12, color=colors_bar)
ax.axhline(0.05, color=A4, lw=1, ls="--", label="0.05 threshold")
fmt_ax(ax, "Time (min)", "|Exp - Thomas|", "Thomas Model Absolute Error")

# ── 5. Sensitivity analysis ───────────────────────────────────────────────────
ax = fig.add_subplot(gs[3, 0])
ax.bar(Q_range, tb_Q, color=A4, alpha=0.8, width=0.7, label="t_b")
fmt_ax(ax, "Q (mL/min)", "t_b (min)", "Sensitivity: Flow Rate Q")

ax = fig.add_subplot(gs[3, 1])
ax.bar(m_range, tb_m, color=A1, alpha=0.8, width=0.3, label="t_b")
fmt_ax(ax, "Bed mass m (g)", "t_b (min)", "Sensitivity: Bed Mass m")

# ── 6. Numerical ADE — Pe = 20 comparison ────────────────────────────────────
ax = fig.add_subplot(gs[4, 0])
Pe_plot = 20
out_lin_p, prof_lin_p = all_lin_results[Pe_plot]
out_nl_p,  prof_nl_p  = all_nl_results[Pe_plot]
ax.plot(out_lin_p[:, 0], out_lin_p[:, 1], color=A1, lw=2,
        label=f"Linearised (Pe={Pe_plot})")
ax.plot(out_nl_p[:, 0],  out_nl_p[:, 1],  color=A2, lw=2, ls="--",
        label=f"Nonlinear LDF (Pe={Pe_plot})")
ax.axhline(0.10, color=A3, lw=1, ls=":", alpha=0.7)
ax.axhline(0.90, color=A4, lw=1, ls=":", alpha=0.7)
fmt_ax(ax, "Time (min)", "C/C0", f"Numerical ADE Breakthrough (Pe={Pe_plot})")

# ── 6b. Spatial profiles at t = 160 min, nonlinear ───────────────────────────
ax = fig.add_subplot(gs[4, 1])
snap_colors = [A1, A2, A3, A4, A5, "#a855f7", "#06b6d4", "#84cc16"]
for i, s in enumerate(snaps):
    if s in prof_nl_p:
        x_arr, C_arr, _ = prof_nl_p[s]
        ax.plot(x_arr, C_arr, color=snap_colors[i % 8], lw=1.5, label=f"t={s} min")
fmt_ax(ax, "Bed position x (cm)", "C/C0",
       f"Nonlinear ADE — Wavefront Propagation (Pe={Pe_plot})")

plt.savefig("/mnt/user-data/outputs/PFAS_Adsorption_Figures.png",
            dpi=150, bbox_inches="tight", facecolor="#0a0e1a")
print("\nFigures saved to PFAS_Adsorption_Figures.png")

plt.show()
print("\nDone.")
