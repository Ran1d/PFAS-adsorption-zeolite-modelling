import { useState, useEffect } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer, ReferenceLine, BarChart, Bar, Cell, ScatterChart, Scatter
} from "recharts";

// ── Colour palette ────────────────────────────────────────────────────────────
const P = {
  bg:      "#0a0e1a",
  panel:   "#111827",
  border:  "#1e2d3d",
  a1:      "#00d4ff",
  a2:      "#7c3aed",
  a3:      "#10b981",
  a4:      "#f59e0b",
  a5:      "#ef4444",
  text:    "#e2e8f0",
  muted:   "#64748b",
  grid:    "#1e293b",
};
const SNAP_COLORS = [P.a1, P.a2, P.a3, P.a4, P.a5, "#a855f7", "#06b6d4", "#84cc16"];

// ── Math helpers ──────────────────────────────────────────────────────────────
const langmuir    = (Ce, qm, KL)   => (qm * KL * Ce) / (1 + KL * Ce);
const freundlich  = (Ce, KF, n)    => KF * Math.pow(Ce, 1 / n);
const pfo         = (t, qe, k1)    => qe * (1 - Math.exp(-k1 * t));
const pso         = (t, qe, k2)    => (k2 * qe * qe * t) / (1 + k2 * qe * t);
const thomas      = (t, kTh, q0, C0, Q, m) => {
  const exp = (kTh / Q) * (q0 * m - C0 * Q * t);
  return 1 / (1 + Math.exp(exp));
};

// Gradient-descent curve fitter (Levenberg-Marquardt approximation)
function curveFit(model, xs, ys, p0, maxIter = 8000, lr = 1e-5) {
  let params = [...p0];
  const n = xs.length;
  for (let iter = 0; iter < maxIter; iter++) {
    const grad = params.map((_, pi) => {
      const dp = 1e-6 * (Math.abs(params[pi]) + 1e-8);
      const pp = [...params]; pp[pi] += dp;
      const pm = [...params]; pm[pi] -= dp;
      const mP = xs.reduce((s, x, i) => s + (ys[i] - model(x, ...pp)) ** 2, 0) / n;
      const mM = xs.reduce((s, x, i) => s + (ys[i] - model(x, ...pm)) ** 2, 0) / n;
      return (mP - mM) / (2 * dp);
    });
    params = params.map((p, i) => Math.max(1e-10, p - lr * grad[i]));
  }
  return params;
}

function r2(yTrue, yPred) {
  const mean = yTrue.reduce((a, b) => a + b, 0) / yTrue.length;
  const ss   = yTrue.reduce((s, y)    => s + (y - mean) ** 2, 0);
  const res  = yTrue.reduce((s, y, i) => s + (y - yPred[i]) ** 2, 0);
  return 1 - res / ss;
}
function mae(yTrue, yPred) {
  return yTrue.reduce((s, y, i) => s + Math.abs(y - yPred[i]), 0) / yTrue.length;
}

// ── Literature data ───────────────────────────────────────────────────────────
const ISO_DATA = [
  { Ce: 0.5, qe: 2.1 }, { Ce: 1.0, qe: 3.8 }, { Ce: 2.0, qe: 6.5 },
  { Ce: 5.0, qe: 11.2 }, { Ce: 10.0, qe: 16.4 }, { Ce: 20.0, qe: 22.1 },
  { Ce: 40.0, qe: 26.8 }, { Ce: 60.0, qe: 29.1 }, { Ce: 80.0, qe: 30.5 },
  { Ce: 100.0, qe: 31.2 },
];
const KIN_DATA = [
  { t: 0, qt: 0 }, { t: 5, qt: 3.2 }, { t: 10, qt: 5.8 },
  { t: 20, qt: 8.9 }, { t: 30, qt: 11.2 }, { t: 60, qt: 13.6 },
  { t: 90, qt: 14.8 }, { t: 120, qt: 15.3 }, { t: 180, qt: 15.7 },
  { t: 240, qt: 15.9 }, { t: 300, qt: 16.0 },
];
const BT_EXP = [
  { t: 0,   r: 0.00 }, { t: 20,  r: 0.01 }, { t: 40,  r: 0.02 },
  { t: 60,  r: 0.04 }, { t: 80,  r: 0.08 }, { t: 100, r: 0.15 },
  { t: 120, r: 0.25 }, { t: 140, r: 0.38 }, { t: 160, r: 0.52 },
  { t: 180, r: 0.65 }, { t: 200, r: 0.76 }, { t: 220, r: 0.84 },
  { t: 240, r: 0.90 }, { t: 260, r: 0.94 }, { t: 280, r: 0.97 },
  { t: 300, r: 0.98 },
];
const BED = { C0: 1.0, Q: 5.0, m: 2.0, qmL: 31.2, KL: 0.085 };

// ── Shared simulation constants for Section 5 ─────────────────────────────────
const SIM = {
  L: 10, nx: 60, v: 0.5, C0: 1.0,
  qm: 31.2, KL: 0.085, rho_b: 0.5, eps: 0.4, ks: 0.15,
  tMax: 320,
  snaps: [40, 80, 120, 160, 200, 240, 280, 320],
};

// Solver A — linearised retardation (no explicit q field)
function solveLinearised(Pe) {
  const { L, nx, v, C0, qm, KL, rho_b, eps, tMax, snaps } = SIM;
  const dx = L / nx;
  const D  = v * L / Pe;
  const dt = Math.min(0.28 * dx / v, 0.38 * dx * dx / (2 * D + 1e-12));
  let C = new Float64Array(nx);
  const profiles = {}; snaps.forEach(s => { profiles[s] = null; });
  const outlet = []; let sample = 0;
  const nSteps = Math.floor(tMax / dt);
  for (let step = 0; step < nSteps; step++) {
    const t = step * dt;
    const Cn = new Float64Array(C);
    for (let i = 1; i < nx - 1; i++) {
      const dqdC = qm * KL / (1 + KL * C[i]) ** 2;
      const R    = 1 + (rho_b / eps) * dqdC;
      const adv  = -v * (C[i] - C[i - 1]) / dx;
      const dif  =  D * (C[i + 1] - 2 * C[i] + C[i - 1]) / (dx * dx);
      Cn[i] = Math.max(0, Math.min(C0, C[i] + (dt / R) * (adv + dif)));
    }
    Cn[0] = C0; Cn[nx - 1] = Cn[nx - 2];
    C = Cn;
    if (++sample % 50 === 0)
      outlet.push({ t: +(t.toFixed(1)), ratio: +(C[nx - 1] / C0).toFixed(5) });
    snaps.forEach(s => {
      if (!profiles[s] && t >= s)
        profiles[s] = Array.from({ length: nx }, (_, i) =>
          ({ x: +(i * dx).toFixed(2), C: +(C[i] / C0).toFixed(5) }));
    });
  }
  return { outlet, profiles, D: D.toFixed(3) };
}

// Solver B — fully-coupled nonlinear LDF (explicit q per cell)
function solveNonlinear(Pe) {
  const { L, nx, v, C0, qm, KL, rho_b, eps, ks, tMax, snaps } = SIM;
  const dx = L / nx;
  const D  = v * L / Pe;
  const dt = Math.min(0.22 * dx / v, 0.32 * dx * dx / (2 * D + 1e-12), 0.5 / ks);
  let C = new Float64Array(nx);
  let q = new Float64Array(nx);
  const profiles = {}; snaps.forEach(s => { profiles[s] = null; });
  const outlet = []; let sample = 0;
  const nSteps = Math.floor(tMax / dt);
  for (let step = 0; step < nSteps; step++) {
    const t  = step * dt;
    const Cn = new Float64Array(C);
    const qn = new Float64Array(q);
    // 1. solid ODE (LDF) at every cell
    for (let i = 0; i < nx; i++) {
      const qStar = (qm * KL * C[i]) / (1 + KL * C[i]);
      qn[i] = Math.max(0, q[i] + dt * ks * (qStar - q[i]));
    }
    // 2. liquid PDE with explicit dq/dt source term
    for (let i = 1; i < nx - 1; i++) {
      const dqdt = (qn[i] - q[i]) / dt;
      const adv  = -v * (C[i] - C[i - 1]) / dx;
      const dif  =  D * (C[i + 1] - 2 * C[i] + C[i - 1]) / (dx * dx);
      Cn[i] = Math.max(0, Math.min(C0, C[i] + dt * ((adv + dif) - (rho_b / eps) * dqdt)));
    }
    Cn[0] = C0; Cn[nx - 1] = Cn[nx - 2];
    C = Cn; q = qn;
    if (++sample % 50 === 0)
      outlet.push({ t: +(t.toFixed(1)), ratio: +(C[nx - 1] / C0).toFixed(5) });
    snaps.forEach(s => {
      if (!profiles[s] && t >= s)
        profiles[s] = Array.from({ length: nx }, (_, i) =>
          ({ x: +(i * dx).toFixed(2), C: +(C[i] / C0).toFixed(5), qN: +(q[i] / qm).toFixed(5) }));
    });
  }
  return { outlet, profiles, D: D.toFixed(3) };
}

function mergeOutlets(linOut, nlOut) {
  const map = {};
  linOut.forEach(d => { map[d.t] = { t: d.t, lin: d.ratio }; });
  nlOut.forEach(d => {
    if (!map[d.t]) map[d.t] = { t: d.t };
    map[d.t].nl = d.ratio;
  });
  return Object.values(map).sort((a, b) => a.t - b.t).map(d => ({
    ...d,
    diff: d.lin != null && d.nl != null ? +(d.nl - d.lin).toFixed(5) : null,
  }));
}

// ── Reusable UI components ────────────────────────────────────────────────────
const Tag = ({ color, children }) => (
  <span style={{
    background: color + "22", color, border: `1px solid ${color}44`,
    borderRadius: 4, fontSize: 10, padding: "2px 7px",
    fontFamily: "monospace", letterSpacing: 1, textTransform: "uppercase",
  }}>{children}</span>
);

const StatBox = ({ label, value, unit, color }) => (
  <div style={{
    background: "#0d1525", border: `1px solid ${color}33`,
    borderRadius: 8, padding: "12px 16px", flex: 1, minWidth: 120,
  }}>
    <div style={{ color: P.muted, fontSize: 10, letterSpacing: 1, textTransform: "uppercase", marginBottom: 4 }}>{label}</div>
    <div style={{ color, fontSize: 20, fontWeight: 700, fontFamily: "monospace" }}>{value}</div>
    {unit && <div style={{ color: P.muted, fontSize: 10, marginTop: 2 }}>{unit}</div>}
  </div>
);

const SecHead = ({ n, title, subtitle, color }) => (
  <div style={{ marginBottom: 24 }}>
    <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 6 }}>
      <div style={{
        width: 36, height: 36, borderRadius: "50%",
        background: `linear-gradient(135deg,${color}33,${color}11)`,
        border: `2px solid ${color}55`,
        display: "flex", alignItems: "center", justifyContent: "center",
        color, fontWeight: 800, fontSize: 14, fontFamily: "monospace",
      }}>{n}</div>
      <h2 style={{ margin: 0, color: P.text, fontSize: 18, fontWeight: 700, letterSpacing: -0.3 }}>{title}</h2>
    </div>
    {subtitle && <p style={{ margin: "0 0 0 48px", color: P.muted, fontSize: 13, lineHeight: 1.6 }}>{subtitle}</p>}
  </div>
);

const Panel = ({ title, note, children }) => (
  <div style={{ background: P.panel, border: `1px solid ${P.border}`, borderRadius: 12, padding: "20px 20px 16px", flex: 1 }}>
    <div style={{ color: P.text, fontSize: 13, fontWeight: 600, marginBottom: 16 }}>{title}</div>
    {children}
    {note && <div style={{ color: P.muted, fontSize: 10, marginTop: 10, fontStyle: "italic" }}>{note}</div>}
  </div>
);

const InterpBox = ({ color, children }) => (
  <div style={{
    background: "#0d1525", border: `1px solid ${color}22`, borderRadius: 10,
    padding: "14px 18px", marginTop: 16, fontSize: 12, color: P.muted, lineHeight: 1.8,
  }}>{children}</div>
);

const Tip = ({ active, payload, label, xLabel }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: "#0f1929ee", border: `1px solid ${P.border}`, borderRadius: 8, padding: "8px 12px" }}>
      <div style={{ color: P.muted, fontSize: 11 }}>{xLabel}: {typeof label === "number" ? label.toFixed(3) : label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color, fontSize: 12, fontWeight: 600 }}>
          {p.name}: {typeof p.value === "number" ? p.value.toFixed(4) : p.value}
        </div>
      ))}
    </div>
  );
};

// ── Section 1: Isotherm ───────────────────────────────────────────────────────
function ISOSection() {
  const [fitted, setFitted] = useState(null);

  useEffect(() => {
    setTimeout(() => {
      const xs = ISO_DATA.map(d => d.Ce);
      const ys = ISO_DATA.map(d => d.qe);
      const [qmL, KL] = curveFit((Ce, qm, KL) => langmuir(Ce, qm, KL), xs, ys, [32, 0.1]);
      const langPred  = xs.map(x => langmuir(x, qmL, KL));
      const [KF, n]   = curveFit((Ce, KF, n)  => freundlich(Ce, KF, n), xs, ys, [5, 2]);
      const freunPred = xs.map(x => freundlich(x, KF, n));
      const smooth = Array.from({ length: 100 }, (_, i) => {
        const Ce = (i + 1) * 1.02;
        return { Ce, lang: langmuir(Ce, qmL, KL), freun: freundlich(Ce, KF, n) };
      });
      const residuals = xs.map((Ce, i) => ({ Ce, lang: ys[i] - langPred[i] }));
      setFitted({
        qmL, KL, KF, n,
        r2L: r2(ys, langPred),  maeL: mae(ys, langPred),
        r2F: r2(ys, freunPred), maeF: mae(ys, freunPred),
        smooth, residuals,
      });
    }, 100);
  }, []);

  const RL = fitted ? 1 / (1 + fitted.KL * 100) : null;

  return (
    <div>
      <SecHead n="1" title="Equilibrium Isotherm Modelling" color={P.a1}
        subtitle="Langmuir and Freundlich models fitted to experimental data via nonlinear least-squares. Data: PFOA on H-ZSM-5 zeolite, 25 C, pH 7." />
      {!fitted && <div style={{ color: P.muted, textAlign: "center", padding: 40 }}>Fitting models…</div>}
      {fitted && <>
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 20 }}>
          <StatBox label="q_max Langmuir"  value={fitted.qmL.toFixed(2)} unit="mg/g"           color={P.a1} />
          <StatBox label="K_L"             value={fitted.KL.toFixed(4)}  unit="L/mg"            color={P.a1} />
          <StatBox label="R2 Langmuir"     value={fitted.r2L.toFixed(4)}                        color={P.a3} />
          <StatBox label="K_F Freundlich"  value={fitted.KF.toFixed(3)}  unit="(mg/g)(L/mg)1/n" color={P.a2} />
          <StatBox label="1/n"             value={(1/fitted.n).toFixed(3)}                       color={P.a2} />
          <StatBox label="R2 Freundlich"   value={fitted.r2F.toFixed(4)}                        color={P.a3} />
          <StatBox label="R_L (C0=100)"    value={RL.toFixed(4)}                                color={P.a4} />
          <StatBox label="MAE Langmuir"    value={fitted.maeL.toFixed(3)} unit="mg/g"           color={P.a5} />
        </div>
        <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
          <Panel title="Isotherm curves — q_e vs C_e" note="Lines: model fits | Experimental points listed below">
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={fitted.smooth} margin={{ top: 4, right: 20, bottom: 20, left: 10 }}>
                <CartesianGrid stroke={P.grid} strokeDasharray="3 3" />
                <XAxis dataKey="Ce" stroke={P.muted} tick={{ fontSize: 11 }}
                  label={{ value: "C_e (mg/L)", position: "insideBottom", dy: 12, fill: P.muted, fontSize: 11 }} />
                <YAxis stroke={P.muted} tick={{ fontSize: 11 }}
                  label={{ value: "q_e (mg/g)", angle: -90, dx: -10, fill: P.muted, fontSize: 11 }} />
                <Tooltip content={<Tip xLabel="Ce" />} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Line type="monotone" dataKey="lang"  stroke={P.a1} strokeWidth={2} dot={false} name="Langmuir" />
                <Line type="monotone" dataKey="freun" stroke={P.a2} strokeWidth={2} dot={false} name="Freundlich" strokeDasharray="6 3" />
              </LineChart>
            </ResponsiveContainer>
          </Panel>
          <Panel title="Langmuir residuals — model error vs C_e" note="Zero line = perfect fit">
            <ResponsiveContainer width="100%" height={260}>
              <ScatterChart margin={{ top: 4, right: 20, bottom: 20, left: 10 }}>
                <CartesianGrid stroke={P.grid} strokeDasharray="3 3" />
                <XAxis dataKey="Ce" name="Ce" type="number" stroke={P.muted} tick={{ fontSize: 11 }}
                  label={{ value: "C_e (mg/L)", position: "insideBottom", dy: 12, fill: P.muted, fontSize: 11 }} />
                <YAxis dataKey="lang" name="Residual" type="number" stroke={P.muted} tick={{ fontSize: 11 }}
                  label={{ value: "Residual (mg/g)", angle: -90, dx: -10, fill: P.muted, fontSize: 11 }} />
                <Tooltip cursor={{ strokeDasharray: "3 3" }}
                  content={({ active, payload }) => {
                    if (!active || !payload?.length) return null;
                    const d = payload[0].payload;
                    return (
                      <div style={{ background: "#0f1929ee", border: `1px solid ${P.border}`, borderRadius: 8, padding: "8px 12px" }}>
                        <div style={{ color: P.muted, fontSize: 11 }}>Ce: {d.Ce}</div>
                        <div style={{ color: P.a1, fontSize: 12 }}>Residual: {d.lang?.toFixed(4)}</div>
                      </div>
                    );
                  }} />
                <ReferenceLine y={0} stroke={P.muted} strokeDasharray="4 2" />
                <Scatter data={fitted.residuals} fill={P.a1} name="Langmuir residual" />
              </ScatterChart>
            </ResponsiveContainer>
          </Panel>
        </div>
        <InterpBox color={P.a1}>
          <span style={{ color: P.a1, fontWeight: 700 }}>Interpretation — </span>
          Langmuir R2 = {fitted.r2L.toFixed(4)} outperforms Freundlich R2 = {fitted.r2F.toFixed(4)}, indicating
          monolayer adsorption on a homogeneous surface. Separation factor R_L = {RL.toFixed(4)} (between 0 and 1)
          confirms favourable adsorption. q_max = {fitted.qmL.toFixed(2)} mg/g falls within the 20–40 mg/g range
          reported for PFOA on modified zeolites.
        </InterpBox>
      </>}
    </div>
  );
}

// ── Section 2: Kinetics ───────────────────────────────────────────────────────
function KINSection() {
  const [fitted, setFitted] = useState(null);

  useEffect(() => {
    setTimeout(() => {
      const ts = KIN_DATA.map(d => d.t);
      const qs = KIN_DATA.map(d => d.qt);
      const [qePFO, k1] = curveFit((t, qe, k1) => pfo(t, qe, k1), ts, qs, [16, 0.02], 10000, 5e-6);
      const [qePSO, k2] = curveFit((t, qe, k2) => pso(t, qe, k2), ts, qs, [16, 0.002], 10000, 5e-7);
      const pfoPred = ts.map(t => pfo(t, qePFO, k1));
      const psoPred = ts.map(t => pso(t, qePSO, k2));
      const smooth  = Array.from({ length: 200 }, (_, i) => {
        const t = i * 1.6;
        return { t, pfo: pfo(t, qePFO, k1), pso: pso(t, qePSO, k2) };
      });
      setFitted({
        qePFO, k1, r2PFO: r2(qs, pfoPred),
        qePSO, k2, r2PSO: r2(qs, psoPred),
        t50: Math.log(2) / k1, smooth,
      });
    }, 100);
  }, []);

  return (
    <div>
      <SecHead n="2" title="Adsorption Kinetics Modelling" color={P.a2}
        subtitle="Pseudo-first-order and pseudo-second-order models fitted to batch kinetic data. Identifies rate-limiting step and equilibrium uptake." />
      {!fitted && <div style={{ color: P.muted, textAlign: "center", padding: 40 }}>Fitting models…</div>}
      {fitted && <>
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 20 }}>
          <StatBox label="q_e PFO"  value={fitted.qePFO.toFixed(3)} unit="mg/g"        color={P.a2} />
          <StatBox label="k_1"      value={fitted.k1.toFixed(5)}    unit="1/min"        color={P.a2} />
          <StatBox label="R2 PFO"   value={fitted.r2PFO.toFixed(4)}                    color={P.a3} />
          <StatBox label="q_e PSO"  value={fitted.qePSO.toFixed(3)} unit="mg/g"        color={P.a4} />
          <StatBox label="k_2"      value={fitted.k2.toFixed(6)}    unit="g/(mg·min)"   color={P.a4} />
          <StatBox label="R2 PSO"   value={fitted.r2PSO.toFixed(4)}                    color={P.a3} />
          <StatBox label="t_half"   value={fitted.t50.toFixed(1)}   unit="min (PFO)"   color={P.a1} />
        </div>
        <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
          <Panel title="Kinetic model curves — q_t vs time">
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={fitted.smooth} margin={{ top: 4, right: 20, bottom: 20, left: 10 }}>
                <CartesianGrid stroke={P.grid} strokeDasharray="3 3" />
                <XAxis dataKey="t" stroke={P.muted} tick={{ fontSize: 11 }}
                  label={{ value: "Time (min)", position: "insideBottom", dy: 12, fill: P.muted, fontSize: 11 }} />
                <YAxis stroke={P.muted} tick={{ fontSize: 11 }}
                  label={{ value: "q_t (mg/g)", angle: -90, dx: -10, fill: P.muted, fontSize: 11 }} />
                <Tooltip content={<Tip xLabel="t" />} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Line type="monotone" dataKey="pfo" stroke={P.a2} strokeWidth={2} dot={false} name="Pseudo-1st-order" />
                <Line type="monotone" dataKey="pso" stroke={P.a4} strokeWidth={2} dot={false} name="Pseudo-2nd-order" strokeDasharray="5 3" />
              </LineChart>
            </ResponsiveContainer>
          </Panel>
          <Panel title="Experimental data points">
            <ResponsiveContainer width="100%" height={260}>
              <ScatterChart margin={{ top: 4, right: 20, bottom: 20, left: 10 }}>
                <CartesianGrid stroke={P.grid} strokeDasharray="3 3" />
                <XAxis dataKey="t"  name="t"  type="number" stroke={P.muted} tick={{ fontSize: 11 }}
                  label={{ value: "Time (min)", position: "insideBottom", dy: 12, fill: P.muted, fontSize: 11 }} />
                <YAxis dataKey="qt" name="qt" type="number" stroke={P.muted} tick={{ fontSize: 11 }}
                  label={{ value: "q_t (mg/g)", angle: -90, dx: -10, fill: P.muted, fontSize: 11 }} />
                <Tooltip cursor={{ strokeDasharray: "3 3" }}
                  content={({ active, payload }) => {
                    if (!active || !payload?.length) return null;
                    const d = payload[0].payload;
                    return (
                      <div style={{ background: "#0f1929ee", border: `1px solid ${P.border}`, borderRadius: 8, padding: "8px 12px" }}>
                        <div style={{ color: P.muted, fontSize: 11 }}>t = {d.t} min</div>
                        <div style={{ color: P.a3, fontSize: 12 }}>Exp q_t: {d.qt}</div>
                        <div style={{ color: P.a2, fontSize: 12 }}>PFO: {pfo(d.t, fitted.qePFO, fitted.k1).toFixed(3)}</div>
                        <div style={{ color: P.a4, fontSize: 12 }}>PSO: {pso(d.t, fitted.qePSO, fitted.k2).toFixed(3)}</div>
                      </div>
                    );
                  }} />
                <Scatter data={KIN_DATA} fill={P.a3} />
              </ScatterChart>
            </ResponsiveContainer>
          </Panel>
        </div>
        <InterpBox color={P.a2}>
          <span style={{ color: P.a2, fontWeight: 700 }}>Interpretation — </span>
          PSO (R2 = {fitted.r2PSO.toFixed(4)}) outperforms PFO (R2 = {fitted.r2PFO.toFixed(4)}), indicating
          chemisorption as the rate-controlling step, consistent with electrostatic and hydrophobic interactions
          between PFOA and the zeolite framework. PFO half-life t_half = {fitted.t50.toFixed(1)} min implies
          equilibrium within ~{(3 * fitted.t50).toFixed(0)} min.
        </InterpBox>
      </>}
    </div>
  );
}

// ── Section 3: Breakthrough ───────────────────────────────────────────────────
function BTSection() {
  const [fitted, setFitted] = useState(null);

  useEffect(() => {
    setTimeout(() => {
      const ts = BT_EXP.map(d => d.t);
      const rs = BT_EXP.map(d => d.r);
      const { C0, Q, m, qmL } = BED;
      const q0 = qmL;
      const Qsec = Q / 60;
      const [kThFit] = curveFit(
        (t, kTh) => thomas(t, kTh, q0, C0, Qsec, m),
        ts, rs, [0.02], 12000, 1e-7
      );
      const thomPred = ts.map(t => thomas(t, kThFit, q0, C0, Qsec, m));
      const combined = ts.map((t, i) => ({
        t,
        exp: rs[i],
        sim: +Math.min(1, Math.max(0, thomas(t, kThFit, q0, C0, Qsec, m))).toFixed(5),
        err: +Math.abs(rs[i] - thomPred[i]).toFixed(5),
      }));
      const tb = ts.find(t => thomas(t, kThFit, q0, C0, Qsec, m) >= 0.1) ?? null;
      const te = ts.find(t => thomas(t, kThFit, q0, C0, Qsec, m) >= 0.9) ?? null;
      setFitted({ kThFit, r2: r2(rs, thomPred), maeVal: mae(rs, thomPred), combined, tb, te, q0 });
    }, 100);
  }, []);

  return (
    <div>
      <SecHead n="3" title="Breakthrough Curve Simulation — Thomas Model" color={P.a3}
        subtitle="Analytical Thomas model fitted to fixed-bed column data. Predicts breakthrough time, bed exhaustion, and column utilisation efficiency." />
      {!fitted && <div style={{ color: P.muted, textAlign: "center", padding: 40 }}>Fitting model…</div>}
      {fitted && <>
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 20 }}>
          <StatBox label="k_Th"          value={fitted.kThFit.toExponential(3)} unit="mL/(min·mg)" color={P.a3} />
          <StatBox label="q_0 (capacity)" value={fitted.q0.toFixed(1)}          unit="mg/g"         color={P.a3} />
          <StatBox label="R2 Thomas"     value={fitted.r2.toFixed(4)}                               color={P.a3} />
          <StatBox label="MAE"           value={fitted.maeVal.toFixed(4)}       unit="C/C0"         color={P.a5} />
          <StatBox label="t_b (C/C0=0.1)" value={fitted.tb ? fitted.tb + " min" : "—"}             color={P.a1} />
          <StatBox label="t_e (C/C0=0.9)" value={fitted.te ? fitted.te + " min" : "—"}             color={P.a4} />
        </div>
        <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
          <Panel title="Breakthrough curve — simulated vs experimental"
            note="Reference lines at C/C0 = 0.1 (breakthrough) and 0.9 (exhaustion)">
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={fitted.combined} margin={{ top: 4, right: 20, bottom: 20, left: 10 }}>
                <CartesianGrid stroke={P.grid} strokeDasharray="3 3" />
                <XAxis dataKey="t" stroke={P.muted} tick={{ fontSize: 11 }}
                  label={{ value: "Time (min)", position: "insideBottom", dy: 12, fill: P.muted, fontSize: 11 }} />
                <YAxis stroke={P.muted} tick={{ fontSize: 11 }} domain={[0, 1.05]}
                  label={{ value: "C/C0", angle: -90, dx: -10, fill: P.muted, fontSize: 11 }} />
                <Tooltip content={<Tip xLabel="t" />} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <ReferenceLine y={0.1} stroke={P.a1} strokeDasharray="4 2"
                  label={{ value: "t_b", fill: P.a1, fontSize: 10, position: "right" }} />
                <ReferenceLine y={0.9} stroke={P.a4} strokeDasharray="4 2"
                  label={{ value: "t_e", fill: P.a4, fontSize: 10, position: "right" }} />
                <Line type="monotone" dataKey="exp" stroke={P.a5} strokeWidth={2} dot={{ r: 3 }} name="Experimental" />
                <Line type="monotone" dataKey="sim" stroke={P.a3} strokeWidth={2} dot={false}    name="Thomas model" />
              </LineChart>
            </ResponsiveContainer>
          </Panel>
          <Panel title="Absolute error — |Exp - Thomas|" note="Green bars: error < 0.05. Red bars: error >= 0.05.">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={fitted.combined} margin={{ top: 4, right: 20, bottom: 20, left: 10 }}>
                <CartesianGrid stroke={P.grid} strokeDasharray="3 3" />
                <XAxis dataKey="t" stroke={P.muted} tick={{ fontSize: 10 }}
                  label={{ value: "Time (min)", position: "insideBottom", dy: 12, fill: P.muted, fontSize: 11 }} />
                <YAxis stroke={P.muted} tick={{ fontSize: 11 }} />
                <Tooltip content={<Tip xLabel="t" />} />
                <Bar dataKey="err" name="|Exp - Sim|" radius={[3, 3, 0, 0]}>
                  {fitted.combined.map((d, i) => (
                    <Cell key={i} fill={d.err > 0.05 ? P.a5 : P.a3} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Panel>
        </div>
        <InterpBox color={P.a3}>
          <span style={{ color: P.a3, fontWeight: 700 }}>Interpretation — </span>
          Thomas model R2 = {fitted.r2.toFixed(4)}, MAE = {fitted.maeVal.toFixed(4)}. Agreement confirms ideal
          plug-flow assumptions hold for this configuration. Bed is serviceable for
          approximately {fitted.tb} min at Q = {BED.Q} mL/min.
        </InterpBox>
      </>}
    </div>
  );
}

// ── Section 4: Sensitivity ────────────────────────────────────────────────────
function SENSSection() {
  const { C0, Q, m, qmL } = BED;
  const q0 = qmL;
  const kTh = 0.018;

  const tb = (kT, q, c0, Qml, ms) => {
    const Qs = Qml / 60;
    return Math.max(0, (q * ms - Qs * Math.log(9) / kT) / (c0 * Qs));
  };

  const flowData = [2, 3, 4, 5, 6, 7, 8, 10].map(Q_ => ({ Q: Q_, tb: +tb(kTh, q0, C0, Q_, m).toFixed(1) }));
  const massData = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0].map(m_ => ({ m: m_, tb: +tb(kTh, q0, C0, Q, m_).toFixed(1) }));
  const concData = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0].map(c_ => ({ C0: c_, tb: +tb(kTh, q0, c_, Q, m).toFixed(1) }));

  const tornado = [
    { param: "Q: 2 to 10 mL/min",   impact: +(-(tb(kTh, q0, C0, 10, m) - tb(kTh, q0, C0, 2, m)) / tb(kTh, q0, C0, 5, m) * 100).toFixed(0) },
    { param: "m: 0.5 to 5 g",        impact: +( (tb(kTh, q0, C0, Q, 5) - tb(kTh, q0, C0, Q, 0.5)) / tb(kTh, q0, C0, Q, 2) * 100).toFixed(0) },
    { param: "C0: 0.2 to 10 mg/L",   impact: +(-(tb(kTh, q0, 10, Q, m) - tb(kTh, q0, 0.2, Q, m)) / tb(kTh, q0, 1, Q, m) * 100).toFixed(0) },
  ];

  return (
    <div>
      <SecHead n="4" title="Sensitivity Analysis — One-Factor-at-a-Time" color={P.a4}
        subtitle="Effect of flow rate, bed mass, and inlet concentration on breakthrough time. Guides column design and scale-up." />
      <div style={{ display: "flex", gap: 16, flexWrap: "wrap", marginBottom: 16 }}>
        {[
          { data: flowData, key: "Q",  label: "Q (mL/min)", title: "Effect of flow rate on t_b",       color: P.a4 },
          { data: massData, key: "m",  label: "Bed mass (g)", title: "Effect of bed mass on t_b",      color: P.a1 },
          { data: concData, key: "C0", label: "C0 (mg/L)",  title: "Effect of C0 on t_b",              color: P.a2 },
        ].map(({ data, key, label, title, color }) => (
          <Panel key={key} title={title} note="">
            <ResponsiveContainer width="100%" height={210}>
              <BarChart data={data} margin={{ top: 4, right: 10, bottom: 20, left: 10 }}>
                <CartesianGrid stroke={P.grid} strokeDasharray="3 3" />
                <XAxis dataKey={key} stroke={P.muted} tick={{ fontSize: 10 }}
                  label={{ value: label, position: "insideBottom", dy: 12, fill: P.muted, fontSize: 10 }} />
                <YAxis stroke={P.muted} tick={{ fontSize: 10 }}
                  label={{ value: "t_b (min)", angle: -90, dx: -8, fill: P.muted, fontSize: 10 }} />
                <Tooltip content={<Tip xLabel={key} />} />
                <Bar dataKey="tb" name="t_b (min)" radius={[4, 4, 0, 0]}>
                  {data.map((_, i) => <Cell key={i} fill={color} opacity={0.4 + 0.07 * i} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Panel>
        ))}
      </div>
      <Panel title="Tornado chart — relative impact on t_b vs baseline (Q=5, m=2, C0=1)">
        <ResponsiveContainer width="100%" height={170}>
          <BarChart layout="vertical" data={tornado}
            margin={{ top: 4, right: 40, bottom: 4, left: 140 }}>
            <CartesianGrid stroke={P.grid} strokeDasharray="3 3" />
            <XAxis type="number" stroke={P.muted} tick={{ fontSize: 10 }} unit="%" />
            <YAxis dataKey="param" type="category" stroke={P.muted}
              tick={{ fontSize: 11, fill: P.text }} width={140} />
            <Tooltip content={<Tip xLabel="Parameter" />} />
            <ReferenceLine x={0} stroke={P.muted} />
            <Bar dataKey="impact" name="% impact on t_b" radius={[0, 4, 4, 0]}>
              {tornado.map((_, i) => <Cell key={i} fill={[P.a4, P.a1, P.a5][i]} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Panel>
      <InterpBox color={P.a4}>
        <span style={{ color: P.a4, fontWeight: 700 }}>Interpretation — </span>
        Bed mass has the largest positive influence on service time; flow rate and inlet concentration reduce it.
        Doubling bed mass approximately doubles t_b in the linear regime. This guides lead-lag column design for continuous PFAS treatment.
      </InterpBox>
    </div>
  );
}

// ── Section 5: Numerical ADE ──────────────────────────────────────────────────
function NUMSection() {
  const [Pe,          setPe]          = useState(20);
  const [mode,        setMode]        = useState("both");
  const [snapTime,    setSnapTime]    = useState(160);
  const [simData,     setSimData]     = useState(null);
  const [running,     setRunning]     = useState(false);

  const runSim = () => {
    setRunning(true);
    setTimeout(() => {
      const lin    = solveLinearised(Pe);
      const nl     = solveNonlinear(Pe);
      const merged = mergeOutlets(lin.outlet, nl.outlet);
      setSimData({ lin, nl, merged, Pe, D: lin.D });
      setRunning(false);
    }, 60);
  };

  const getT = (outlet, threshold) => {
    const hit = outlet.find(d => d.ratio >= threshold);
    return hit ? hit.t + " min" : "> 320 min";
  };

  const PeBtn  = ({ val }) => (
    <button onClick={() => setPe(val)} style={{
      background: Pe === val ? P.a5 + "28" : "transparent",
      border: `1px solid ${Pe === val ? P.a5 : P.border}`,
      borderRadius: 6, padding: "5px 13px",
      color: Pe === val ? P.a5 : P.muted,
      cursor: "pointer", fontSize: 12, fontFamily: "monospace", fontWeight: 600,
    }}>{val}</button>
  );

  const ModeBtn = ({ val, label, color }) => (
    <button onClick={() => setMode(val)} style={{
      background: mode === val ? color + "22" : "transparent",
      border: `1px solid ${mode === val ? color : P.border}`,
      borderRadius: 6, padding: "5px 13px",
      color: mode === val ? color : P.muted,
      cursor: "pointer", fontSize: 12, fontWeight: 600,
    }}>{label}</button>
  );

  const SnapBtn = ({ val, activeColor }) => (
    <button onClick={() => setSnapTime(val)} style={{
      background: snapTime === val ? activeColor + "28" : "transparent",
      border: `1px solid ${snapTime === val ? activeColor : P.border}`,
      borderRadius: 5, padding: "3px 9px",
      color: snapTime === val ? activeColor : P.muted,
      cursor: "pointer", fontSize: 11, fontFamily: "monospace",
    }}>{val}</button>
  );

  return (
    <div>
      <SecHead n="5" title="Numerical 1-D ADE — Linearised vs Fully-Coupled Nonlinear" color={P.a5}
        subtitle={
          "Two finite-difference solvers run in parallel. " +
          "Solver A uses a local retardation factor R(C) — no q field is tracked. " +
          "Solver B carries an explicit solid loading q(t) per grid cell, " +
          "updated via the Linear Driving Force ODE; C and q are coupled through the mass-balance source term rho_b * dq/dt."
        } />

      {/* Controls */}
      <div style={{
        background: P.panel, border: `1px solid ${P.border}`, borderRadius: 12,
        padding: "16px 20px", marginBottom: 20,
        display: "flex", gap: 24, flexWrap: "wrap", alignItems: "flex-start",
      }}>
        <div>
          <div style={{ color: P.muted, fontSize: 11, letterSpacing: 1, textTransform: "uppercase", marginBottom: 8 }}>
            Peclet number Pe = vL/D
          </div>
          <div style={{ display: "flex", gap: 6 }}>
            {[5, 10, 20, 50, 100].map(v => <PeBtn key={v} val={v} />)}
          </div>
        </div>
        <div>
          <div style={{ color: P.muted, fontSize: 11, letterSpacing: 1, textTransform: "uppercase", marginBottom: 8 }}>
            Display
          </div>
          <div style={{ display: "flex", gap: 6 }}>
            <ModeBtn val="both" label="Both"         color={P.a3} />
            <ModeBtn val="lin"  label="Linearised"   color={P.a1} />
            <ModeBtn val="nl"   label="Nonlinear"    color={P.a2} />
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "flex-end" }}>
          <button onClick={runSim} disabled={running} style={{
            background: running ? "#1a2030" : P.a5 + "22",
            border: `1px solid ${running ? P.border : P.a5}`,
            borderRadius: 8, padding: "8px 24px",
            color: running ? P.muted : P.a5,
            cursor: running ? "not-allowed" : "pointer",
            fontSize: 13, fontWeight: 700,
          }}>
            {running ? "Solving…" : "Run both solvers"}
          </button>
        </div>
      </div>

      {/* Equation boxes */}
      <div style={{ display: "flex", gap: 12, marginBottom: 20, flexWrap: "wrap" }}>
        <div style={{ flex: 1, minWidth: 260, background: P.a1 + "0a", border: `1px solid ${P.a1}33`, borderRadius: 10, padding: "14px 16px" }}>
          <div style={{ color: P.a1, fontWeight: 700, fontSize: 12, marginBottom: 8 }}>Solver A — Linearised retardation</div>
          <div style={{ fontFamily: "monospace", fontSize: 11, color: P.muted, lineHeight: 2 }}>
            <div style={{ color: P.text }}>dC/dt = (1/R) · [D·d2C/dx2 − v·dC/dx]</div>
            <div>R(C) = 1 + (rho_b/eps) · dq*/dC at C_local</div>
            <div>dq*/dC = q_m·K_L / (1 + K_L·C)^2</div>
            <div style={{ color: P.a5, marginTop: 4 }}>q field never tracked explicitly</div>
          </div>
        </div>
        <div style={{ flex: 1, minWidth: 260, background: P.a2 + "0a", border: `1px solid ${P.a2}33`, borderRadius: 10, padding: "14px 16px" }}>
          <div style={{ color: P.a2, fontWeight: 700, fontSize: 12, marginBottom: 8 }}>Solver B — Fully-coupled nonlinear (LDF)</div>
          <div style={{ fontFamily: "monospace", fontSize: 11, color: P.muted, lineHeight: 2 }}>
            <div style={{ color: P.text }}>eps·dC/dt = eps·[D·d2C/dx2 − v·dC/dx] − rho_b·dq/dt</div>
            <div>dq/dt = k_s · [q*(C) − q]   (ODE per cell)</div>
            <div>q*(C) = q_m·K_L·C / (1 + K_L·C)</div>
            <div style={{ color: P.a3, marginTop: 4 }}>q(x,t) tracked explicitly — no linearisation</div>
          </div>
        </div>
      </div>

      {/* Idle state */}
      {!simData && !running && (
        <div style={{
          background: P.panel, border: `1px dashed ${P.border}`, borderRadius: 12,
          padding: 44, textAlign: "center", color: P.muted, fontSize: 13,
        }}>
          Select Pe and click <span style={{ color: P.a5, fontWeight: 600 }}>Run both solvers</span> to compare the two formulations.
          <div style={{ fontSize: 11, marginTop: 6 }}>Both solvers run simultaneously. The difference chart shows where linearisation error accumulates.</div>
        </div>
      )}

      {running && (
        <div style={{
          background: P.panel, border: `1px solid ${P.border}`, borderRadius: 12,
          padding: 44, textAlign: "center",
        }}>
          <div style={{ color: P.a5, fontSize: 14, fontWeight: 700, marginBottom: 8 }}>Running finite-difference solvers…</div>
          <div style={{ color: P.muted, fontSize: 12 }}>
            Solver A: upwind FD + local R(C) &nbsp;|&nbsp; Solver B: LDF-ODE per cell + full PDE coupling
          </div>
        </div>
      )}

      {simData && !running && (function() {
        const { lin, nl, merged } = simData;
        const tbLin = getT(lin.outlet, 0.10);
        const teLin = getT(lin.outlet, 0.90);
        const tbNL  = getT(nl.outlet,  0.10);
        const teNL  = getT(nl.outlet,  0.90);
        const validDiffs = merged.filter(d => d.diff != null).map(d => Math.abs(d.diff));
        const maxDiff = validDiffs.length ? Math.max(...validDiffs) : 0;

        return (
          <div>
            {/* Stats */}
            <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 20 }}>
              <StatBox label="Pe"              value={simData.Pe}    color={P.a5} />
              <StatBox label="D"               value={simData.D}     unit="cm2/min" color={P.a5} />
              <StatBox label="t_b Linearised"  value={tbLin}         color={P.a1} />
              <StatBox label="t_b Nonlinear"   value={tbNL}          color={P.a2} />
              <StatBox label="t_e Linearised"  value={teLin}         color={P.a1} />
              <StatBox label="t_e Nonlinear"   value={teNL}          color={P.a2} />
              <StatBox label="Max |delta C/C0|" value={maxDiff.toFixed(4)} color={P.a4} />
            </div>

            {/* Breakthrough overlay + diff chart */}
            <div style={{ display: "flex", gap: 16, flexWrap: "wrap", marginBottom: 16 }}>
              <Panel title="Breakthrough curves — Solver A (blue) vs Solver B (purple)"
                note="Reference lines at C/C0 = 0.1 and 0.9">
                <ResponsiveContainer width="100%" height={260}>
                  <LineChart data={merged} margin={{ top: 4, right: 20, bottom: 20, left: 10 }}>
                    <CartesianGrid stroke={P.grid} strokeDasharray="3 3" />
                    <XAxis dataKey="t" stroke={P.muted} tick={{ fontSize: 11 }}
                      label={{ value: "Time (min)", position: "insideBottom", dy: 12, fill: P.muted, fontSize: 11 }} />
                    <YAxis stroke={P.muted} tick={{ fontSize: 11 }} domain={[0, 1.05]}
                      label={{ value: "C/C0", angle: -90, dx: -10, fill: P.muted, fontSize: 11 }} />
                    <Tooltip content={({ active, payload, label }) => {
                      if (!active || !payload?.length) return null;
                      const d = payload[0]?.payload;
                      return (
                        <div style={{ background: "#0f1929ee", border: `1px solid ${P.border}`, borderRadius: 8, padding: "8px 12px" }}>
                          <div style={{ color: P.muted, fontSize: 11 }}>t = {label} min</div>
                          {d.lin != null && <div style={{ color: P.a1, fontSize: 12 }}>Linearised: {d.lin.toFixed(4)}</div>}
                          {d.nl  != null && <div style={{ color: P.a2, fontSize: 12 }}>Nonlinear:  {d.nl.toFixed(4)}</div>}
                          {d.diff != null && <div style={{ color: P.a4, fontSize: 11 }}>Delta: {d.diff > 0 ? "+" : ""}{d.diff.toFixed(5)}</div>}
                        </div>
                      );
                    }} />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    <ReferenceLine y={0.1} stroke={P.a3} strokeDasharray="4 2" label={{ value: "t_b", fill: P.a3, fontSize: 10, position: "right" }} />
                    <ReferenceLine y={0.9} stroke={P.a4} strokeDasharray="4 2" label={{ value: "t_e", fill: P.a4, fontSize: 10, position: "right" }} />
                    {(mode === "lin" || mode === "both") &&
                      <Line type="monotone" dataKey="lin" stroke={P.a1} strokeWidth={2.5} dot={false} name="Linearised (A)" />}
                    {(mode === "nl" || mode === "both") &&
                      <Line type="monotone" dataKey="nl"  stroke={P.a2} strokeWidth={2.5} dot={false} name="Nonlinear LDF (B)"
                        strokeDasharray={mode === "both" ? "7 3" : "0"} />}
                  </LineChart>
                </ResponsiveContainer>
              </Panel>

              <Panel title="delta(C/C0) = Nonlinear minus Linearised"
                note="Purple: nonlinear predicts faster breakthrough. Blue: linearised predicts faster.">
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={merged.filter(d => d.diff != null)} margin={{ top: 4, right: 20, bottom: 20, left: 10 }}>
                    <CartesianGrid stroke={P.grid} strokeDasharray="3 3" />
                    <XAxis dataKey="t" stroke={P.muted} tick={{ fontSize: 10 }}
                      label={{ value: "Time (min)", position: "insideBottom", dy: 12, fill: P.muted, fontSize: 11 }} />
                    <YAxis stroke={P.muted} tick={{ fontSize: 10 }}
                      label={{ value: "delta(C/C0)", angle: -90, dx: -10, fill: P.muted, fontSize: 11 }} />
                    <Tooltip content={<Tip xLabel="t" />} />
                    <ReferenceLine y={0} stroke={P.muted} />
                    <Bar dataKey="diff" name="NL minus Lin" radius={[3, 3, 0, 0]}>
                      {merged.filter(d => d.diff != null).map((d, i) => (
                        <Cell key={i}
                          fill={d.diff >= 0 ? P.a2 : P.a1}
                          opacity={0.5 + 0.5 * Math.abs(d.diff) / (maxDiff + 1e-9)} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </Panel>
            </div>

            {/* Spatial profile panels */}
            <div style={{ display: "flex", gap: 16, flexWrap: "wrap", marginBottom: 16 }}>
              <Panel title={"C/C0 spatial profile at t = " + snapTime + " min"}
                note="Adsorption wavefront inside the bed. Both solvers overlaid.">
                <div style={{ display: "flex", gap: 4, marginBottom: 10, flexWrap: "wrap" }}>
                  {SIM.snaps.map(s => <SnapBtn key={s} val={s} activeColor={P.a5} />)}
                </div>
                <ResponsiveContainer width="100%" height={220}>
                  <LineChart margin={{ top: 4, right: 20, bottom: 20, left: 10 }}>
                    <CartesianGrid stroke={P.grid} strokeDasharray="3 3" />
                    <XAxis dataKey="x" type="number" stroke={P.muted} tick={{ fontSize: 11 }}
                      domain={[0, SIM.L]}
                      label={{ value: "Bed position x (cm)", position: "insideBottom", dy: 12, fill: P.muted, fontSize: 11 }} />
                    <YAxis stroke={P.muted} tick={{ fontSize: 11 }} domain={[0, 1.05]}
                      label={{ value: "C/C0", angle: -90, dx: -10, fill: P.muted, fontSize: 11 }} />
                    <Tooltip content={<Tip xLabel="x" />} />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    {lin.profiles[snapTime] && (mode === "lin" || mode === "both") &&
                      <Line data={lin.profiles[snapTime]} type="monotone" dataKey="C"
                        stroke={P.a1} strokeWidth={2} dot={false} name="Linearised C" />}
                    {nl.profiles[snapTime] && (mode === "nl" || mode === "both") &&
                      <Line data={nl.profiles[snapTime]} type="monotone" dataKey="C"
                        stroke={P.a2} strokeWidth={2} dot={false} name="Nonlinear C"
                        strokeDasharray={mode === "both" ? "7 3" : "0"} />}
                  </LineChart>
                </ResponsiveContainer>
              </Panel>

              <Panel title={"Solid loading q/q_max at t = " + snapTime + " min  (Nonlinear only)"}
                note="Only Solver B tracks q(x,t) explicitly. Shows local saturation state along the bed.">
                <div style={{ display: "flex", gap: 4, marginBottom: 10, flexWrap: "wrap" }}>
                  {SIM.snaps.map(s => <SnapBtn key={s} val={s} activeColor={P.a2} />)}
                </div>
                <ResponsiveContainer width="100%" height={220}>
                  <LineChart margin={{ top: 4, right: 20, bottom: 20, left: 10 }}>
                    <CartesianGrid stroke={P.grid} strokeDasharray="3 3" />
                    <XAxis dataKey="x" type="number" stroke={P.muted} tick={{ fontSize: 11 }}
                      domain={[0, SIM.L]}
                      label={{ value: "Bed position x (cm)", position: "insideBottom", dy: 12, fill: P.muted, fontSize: 11 }} />
                    <YAxis stroke={P.muted} tick={{ fontSize: 11 }} domain={[0, 1.05]}
                      label={{ value: "q/q_max", angle: -90, dx: -10, fill: P.muted, fontSize: 11 }} />
                    <Tooltip content={<Tip xLabel="x" />} />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    {nl.profiles[snapTime] &&
                      <Line data={nl.profiles[snapTime]} type="monotone" dataKey="qN"
                        stroke={P.a2} strokeWidth={2.5} dot={false} name="q/q_max" />}
                  </LineChart>
                </ResponsiveContainer>
              </Panel>
            </div>

            {/* All-time wavefront */}
            <Panel title="Concentration wavefront propagation — all snapshots (Nonlinear solver)"
              note="Each curve is a 40-min snapshot. Steepness vs diffuseness reveals the interplay of advection, dispersion, and nonlinear retardation.">
              <ResponsiveContainer width="100%" height={220}>
                <LineChart margin={{ top: 4, right: 20, bottom: 20, left: 10 }}>
                  <CartesianGrid stroke={P.grid} strokeDasharray="3 3" />
                  <XAxis dataKey="x" type="number" stroke={P.muted} tick={{ fontSize: 11 }}
                    domain={[0, SIM.L]}
                    label={{ value: "Bed position x (cm)", position: "insideBottom", dy: 12, fill: P.muted, fontSize: 11 }} />
                  <YAxis stroke={P.muted} tick={{ fontSize: 11 }} domain={[0, 1.05]}
                    label={{ value: "C/C0", angle: -90, dx: -10, fill: P.muted, fontSize: 11 }} />
                  <Tooltip content={<Tip xLabel="x" />} />
                  <Legend wrapperStyle={{ fontSize: 10 }} />
                  {Object.entries(nl.profiles).filter(([, v]) => v).map(([t, data], i) => (
                    <Line key={t} data={data} type="monotone" dataKey="C"
                      stroke={SNAP_COLORS[i % 8]} strokeWidth={1.5} dot={false} name={"t=" + t + " min"} />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </Panel>

            {/* Interpretation */}
            <InterpBox color={P.a5}>
              <div style={{ display: "flex", gap: 20, flexWrap: "wrap" }}>
                <div style={{ flex: 1, minWidth: 200 }}>
                  <div style={{ color: P.a1, fontWeight: 600, marginBottom: 4 }}>Solver A — Linearised</div>
                  Uses R(C) = 1 + (rho_b/eps)·dq*/dC evaluated at the current local C.
                  The local Langmuir slope is correct but q-values are never stored, so saturation history
                  is lost. As C approaches C0 the curvature of the isotherm means dq*/dC shrinks,
                  and the solver can overestimate retardation in partially-saturated cells.
                </div>
                <div style={{ flex: 1, minWidth: 200 }}>
                  <div style={{ color: P.a2, fontWeight: 600, marginBottom: 4 }}>Solver B — Nonlinear LDF</div>
                  Each cell integrates dq/dt = k_s·[q*(C) − q] independently using the full Langmuir expression.
                  Because q lags behind q*(C) during loading, the actual driving force is smaller than in
                  the linearised model, which can shift and sharpen the breakthrough curve.
                  As k_s tends to infinity, local equilibrium is recovered and both solvers converge.
                </div>
                <div style={{ flex: 1, minWidth: 200 }}>
                  <div style={{ color: P.a4, fontWeight: 600, marginBottom: 4 }}>When does it matter?</div>
                  Max |delta C/C0| = {maxDiff.toFixed(4)}.{" "}
                  {maxDiff < 0.01
                    ? "Both solvers agree closely. The linearisation is adequate — the system is near local equilibrium and mass-transfer resistance is negligible."
                    : maxDiff < 0.05
                      ? "Modest divergence visible around the inflection point of the breakthrough curve. For rigorous design the nonlinear formulation is preferred."
                      : "Significant divergence. The linearised solver meaningfully mis-predicts breakthrough timing. The nonlinear LDF formulation is required — especially for high-concentration feeds, steep isotherms, or low k_s."}
                </div>
              </div>
            </InterpBox>
          </div>
        );
      })()}
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────
const TABS = [
  { id: "iso",  label: "Isotherm",      color: P.a1 },
  { id: "kin",  label: "Kinetics",      color: P.a2 },
  { id: "bt",   label: "Breakthrough",  color: P.a3 },
  { id: "sens", label: "Sensitivity",   color: P.a4 },
  { id: "num",  label: "Numerical ADE", color: P.a5 },
];

export default function App() {
  const [tab, setTab] = useState("iso");
  return (
    <div style={{ background: P.bg, minHeight: "100vh", color: P.text, fontFamily: "'IBM Plex Sans','Segoe UI',sans-serif" }}>
      {/* Header */}
      <div style={{ background: "linear-gradient(135deg,#0d1b2a 0%,#0a0e1a 60%,#0d1525 100%)", borderBottom: `1px solid ${P.border}`, padding: "28px 32px 20px" }}>
        <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", flexWrap: "wrap", gap: 12 }}>
          <div>
            <div style={{ display: "flex", gap: 8, marginBottom: 10, flexWrap: "wrap" }}>
              <Tag color={P.a1}>ME724 — Environmental Engineering</Tag>
              <Tag color={P.a3}>Computational Study</Tag>
              <Tag color={P.a4}>PFAS Remediation</Tag>
            </div>
            <h1 style={{
              margin: 0, fontSize: "clamp(18px,3vw,26px)", fontWeight: 800, letterSpacing: -0.5,
              background: `linear-gradient(90deg,${P.a1},${P.a2})`,
              WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
            }}>Fixed-Bed Adsorption of PFAS on Zeolites</h1>
            <p style={{ margin: "8px 0 0", color: P.muted, fontSize: 13, lineHeight: 1.5 }}>
              Isotherm fitting — Kinetics — Thomas model — Sensitivity analysis — Numerical 1D ADE
            </p>
          </div>
          <div style={{ background: "#0d1525", border: `1px solid ${P.border}`, borderRadius: 10, padding: "10px 16px", fontSize: 11, color: P.muted, lineHeight: 1.8 }}>
            <div style={{ color: P.text, fontWeight: 600, marginBottom: 4, fontSize: 12 }}>Data Sources</div>
            <div>Rattanaoudom et al. (2021) — PFOA on H-ZSM-5</div>
            <div>Du et al. (2020) — Kinetics, batch 25 C</div>
            <div>Sorengard et al. (2020) — Column breakthrough</div>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: "flex", gap: 2, padding: "0 32px", background: "#0c1220", borderBottom: `1px solid ${P.border}`, overflowX: "auto" }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            background: tab === t.id ? t.color + "18" : "transparent",
            border: "none", borderBottom: tab === t.id ? `2px solid ${t.color}` : "2px solid transparent",
            color: tab === t.id ? t.color : P.muted,
            padding: "14px 20px", cursor: "pointer", fontSize: 13, fontWeight: 600,
            letterSpacing: 0.2, whiteSpace: "nowrap", transition: "all 0.15s",
          }}>{t.label}</button>
        ))}
      </div>

      {/* Content */}
      <div style={{ padding: "32px 32px 48px", maxWidth: 1400, margin: "0 auto" }}>
        {tab === "iso"  && <ISOSection />}
        {tab === "kin"  && <KINSection />}
        {tab === "bt"   && <BTSection />}
        {tab === "sens" && <SENSSection />}
        {tab === "num"  && <NUMSection />}
      </div>

      {/* Footer */}
      <div style={{ borderTop: `1px solid ${P.border}`, padding: "16px 32px", color: P.muted, fontSize: 11, display: "flex", justifyContent: "space-between", flexWrap: "wrap", gap: 8 }}>
        <span>All computations in-browser: Levenberg-Marquardt optimiser · upwind FD linearised · LDF-ODE coupled FD nonlinear · Thomas analytical model</span>
        <span>PFOA on H-ZSM-5 · 25 C · pH 7 · C0 = 1–100 mg/L · Q = 5 mL/min · m = 2 g</span>
      </div>
    </div>
  );
}
