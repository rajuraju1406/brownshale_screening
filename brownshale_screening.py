"""
Supplementary code: Monte Carlo screening of CO2 huff-n-puff (HnP) in the Pematang Brown Shale
==================================================================================================
Companion code for "Between Miscibility and Fracture Pressure: A Monte Carlo Screening of CO2
Huff-n-Puff in the Pematang Brown Shale, Rokan Block, Central Sumatra Basin".

Purpose
    Screens the pressure window between the minimum miscibility pressure (MMP) and the fracture
    pressure at a representative depth of 2.6 km, and the CO2 retention needed for a
    combustion-inclusive carbon balance. It is a screening calculation on representative and
    analog data, not a field-measured evaluation.

Workflow (research questions, RQ)
    1.   Benchmark of three MMP correlations against two slim-tube MMPs (Song et al., 2024).
    RQ1  MMP, reservoir pressure, and pressure window, using the correlations directly ("direct")
         and anchored to the measured analog MMPs and extrapolated in temperature ("anchored").
    RQ2  Analog injection design (60 MPa into 35.5-37.05 MPa reservoirs) rescaled to the Brown
         Shale and tested against the anchored MMP of the SAME analog and the fracture pressure
         (paired test); an alternative MMP-ratio rescaling is reported as a sensitivity.
    RQ3  Sensitivity to nanoconfinement, reservoir-pressure gradient, geothermal gradient, depth,
         and oil priors.
    RQ4  Gross CO2 injection required for carbon neutrality at the analog storage rates, with
         project emissions either fixed per barrel or growing with gross injection, and the
         share of realizations in which the analog design (and thus its storage rate) applies.
    Figures  Figure 1 (schematic, not computed) and Figures 3-5 (computed from this run).

Input conventions (comments in the code)
    Source      value taken from a cited publication
    Analog      value borrowed from another basin
    Assumption  modelling choice that has no direct Brown Shale measurement behind it

Requirements   Python >= 3.9, NumPy, and Matplotlib (figures only).
Usage          python brownshale_screening.py
Outputs        console log of every step; results.json (numbers used in Tables 2-3 and Section 4);
               Figure1_concept.png, Figure3_mmp_temperature.png, Figure4_injection_transfer.png,
               Figure5_ranking.png.
Reproducibility  Fixed random seeds, N = 100,000 realizations; the run ends with a consistency
                 check against the values reported in the manuscript.

Principal sources
    Song et al. (2024), Adv. Geo-Energy Res. 13(2):106-118      analog slim-tube MMPs, injection design, storage rates
    Cronquist (1978); Yellig & Metcalfe (1980); Alston et al. (1985)   MMP correlations
    Xiong et al. (2026), Petroleum Science 23:777-790           confinement-related MMP reductions
    Lukman (2020), Jurnal Petro 9(1):21-29                      log-derived pore pressure, Pematang Group
    U.S. EIA (2015); USGS Fact Sheet 2015-3072                  pressure class, resource and GOR context
    U.S. EPA (2026)                                             combustion emission factor
    Wang et al. (2025), Chem. Eng. J. 506:160098                CO2 retention from molecular simulation
"""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ------------------------------------------------------------------------------------------
# Constants and settings
# ------------------------------------------------------------------------------------------
N = 100_000                       # Monte Carlo realizations
SEED = 42                         # base-case seed
FT, PSI = 0.3048, 0.006894757     # m per ft; MPa per psi
DEPTH_FT = 8559                   # Assumption: representative depth = planned total depth of Gulamo-1 (Pertamina, 2023)
ETA = 0.67                        # Assumption: poroelastic coupling dP_frac/dP_res (nu = 0.25, Biot coefficient = 1)
EF = 0.43                         # Source: t CO2 per barrel of crude oil burned (U.S. EPA, 2026); combustion only
STORAGE_RATES = {"Song oil #1": 0.4476, "Song oil #2": 0.6933}   # Source: Song et al. (2024)
CONFINEMENT_R = (0.0, 0.005, 0.09, 0.16, 0.32)   # Source: Xiong et al. (2026), 50/10/5 nm -> 0.47/9.01/16.31%;
                                                # T. Wan et al. (2024, RSC Adv. 14:27227), 31.25 -> 21.25 MPa from 50 to 5 nm (32%), upper scenario
SONG_MATRIX = {"Oil#1": dict(inj_mol=1171.05, oil_m3=0.43), "Oil#2": dict(inj_mol=401.66, oil_m3=0.44)}  # Song et al. (2024), Sections 5.3 and 5.5
PRESSURE_G = (1.0, 1.2, 1.4, 1.6)          # multipliers on hydrostatic reservoir pressure (scenarios)
LUKMAN_PSI = (4523, 4891)                  # Source: Lukman (2020), pore pressure at 2,600 and 2,640 m
GEOTHERMAL_G = (30, 40, 50, 60, 80)        # C/km, fixed-gradient scenarios inside the reported 30 to >120 C/km
E_PROP = (0.0, 0.05, 0.10)                 # Assumption: t CO2 emitted per t CO2 injected (compression/recycling)

q = lambda a, p: np.percentile(a, p)


def pr(label, a, unit="MPa", fmt="{:.1f}"):
    print(f"{label:46s} P10/50/90: " + " / ".join(fmt.format(q(a, p)) for p in (10, 50, 90)) + f" {unit}")


def header(text):
    print(f"\n=== {text} ===")


# ------------------------------------------------------------------------------------------
# MMP correlations (temperature in deg F, result in MPa)
# ------------------------------------------------------------------------------------------
def mmp_cronquist(TF, MW, c1):
    """Cronquist (1978). MW: C5+ molecular weight; c1: mol% C1 in live oil. Data range 21.7-120.8 C."""
    return 15.988 * TF ** (0.744206 + 0.0011038 * MW + 0.0015279 * c1) * PSI


def mmp_yellig(TF):
    """Yellig & Metcalfe (1980), temperature term only. Data range 35-88.9 C."""
    return (1833.7217 + 2.2518055 * TF + 0.01800674 * TF ** 2 - 103949.93 / TF) * PSI


def mmp_alston(TF, MW, vi):
    """Alston et al. (1985), pure CO2. vi = x(C1+N2) / x(C2-C4+CO2+H2S) in live oil."""
    return 0.000878 * TF ** 1.06 * MW ** 1.78 * vi ** 0.136 * PSI


# ------------------------------------------------------------------------------------------
# Analog oils (Song et al., 2024, Tables 1-2)
# ------------------------------------------------------------------------------------------
# vol = C1 + N2; inter = C2 + C3 + C4 + CO2; c56 = C5 + C6 (used only in a sensitivity check).
# MW(C5+) is recomputed from the C5-C38 fractions; Table 2 prints 887.64 for the C19-C23 fraction
# of oil #2, and 287.64 is used.
ANALOGS = {
    "Oil#1 (Ro 1.45%, volatile)": dict(T_K=374.72, mmp=20.00, c1=65.22, mw5=208.9, vol=65.22 + 0.17,
                                       inter=12.79 + 4.72 + 1.80 + 0.12, c56=0.42 + 0.66, Pres=35.50),
    "Oil#2 (Ro 1.15%, light)":    dict(T_K=377.25, mmp=27.50, c1=46.66, mw5=211.6, vol=46.66 + 0.27,
                                       inter=12.05 + 4.78 + 1.80 + 0.09, c56=0.63 + 1.17, Pres=37.05),
}
for _a in ANALOGS.values():
    _a["TF"] = (_a["T_K"] - 273.15) * 9 / 5 + 32


# ------------------------------------------------------------------------------------------
# Reservoir-state sampler and MMP estimates
# ------------------------------------------------------------------------------------------
def sample_state(depth_ft, seed=SEED, api=(30, 38), gor=(300, 600, 900), fixed_G=None):
    """Draws temperature, pressures, and oil properties for N realizations at a given depth.

    fixed_G (C/km) replaces the triangular gradient by a constant; the random stream is kept
    identical so that all other variables are unchanged between scenarios.
    """
    rng = np.random.default_rng(seed)
    depth_m = depth_ft * FT
    grad = rng.triangular(40, 50, 60, N) / 1000.0             # Assumption: 40/50/60 C/km; reported range 30 to >120 C/km (Siringoringo et al., 2025)
    if fixed_G is not None:
        grad = np.full(N, fixed_G / 1000.0)
    Ts = rng.uniform(27, 29, N)                               # Assumption: surface temperature, C
    T_C = Ts + grad * depth_m
    P_res = rng.uniform(1000, 1070, N) * 9.81 * depth_m / 1e6 # Assumption: hydrostatic ("normal" class, U.S. EIA, 2015)
    P_frac = rng.uniform(0.70, 0.80, N) * depth_ft * PSI      # Assumption: fracture gradient 0.70-0.80 psi/ft
    API = rng.uniform(*api, N)                                # Assumption: stock-tank gravity, deg API
    MW_c5 = 6084.0 / (API - 5.9)                              # Cragoe relation; stock-tank MW used as C5+ MW
    GOR = rng.triangular(*gor, N)                             # scf/STB; mode from USGS FS 2015-3072 mean (275 BCFG / 459 MMBO = 599), bounds assumed
    fC1 = rng.uniform(0.70, 0.85, N)                          # Assumption: C1 fraction of solution gas
    rho_o = 141.5 / (131.5 + API)                             # stock-tank density, kg/L
    n_oil = 158.987 * rho_o * 1000 / MW_c5                    # mol per STB
    n_gas = GOR / 379.5 * 453.59                              # mol per STB
    c1_mol = n_gas * fC1 / (n_gas + n_oil) * 100              # live-oil C1, mol%
    vol_int = fC1 / (1 - fC1)                                 # x(C1)/x(C2-C4) of the solution gas
    return dict(depth_ft=depth_ft, T_C=T_C, T_F=T_C * 9 / 5 + 32, P_res=P_res, P_frac=P_frac,
                MW_c5=MW_c5, c1_mol=c1_mol, vol_int=vol_int, API=API, GOR=GOR, fC1=fC1)


def mmp_sets(s):
    """Direct correlations and MMPs anchored to the measured analog values.

    Anchored: MMP = MMP_measured * f(T_Brown Shale) / f(T_analog), using each correlation's own
    temperature dependence with the analog composition held fixed.
    """
    direct = {"Cronquist": mmp_cronquist(s["T_F"], s["MW_c5"], s["c1_mol"]),
              "Yellig-Metcalfe (extrap.)": mmp_yellig(s["T_F"]),
              "Alston": mmp_alston(s["T_F"], s["MW_c5"], s["vol_int"])}
    anchored = {}
    for name, a in ANALOGS.items():
        tag, vi = name[:5], a["vol"] / a["inter"]
        anchored[f"{tag} x Cronquist"] = a["mmp"] * mmp_cronquist(s["T_F"], a["mw5"], a["c1"]) / mmp_cronquist(a["TF"], a["mw5"], a["c1"])
        anchored[f"{tag} x Yellig"] = a["mmp"] * mmp_yellig(s["T_F"]) / mmp_yellig(a["TF"])
        anchored[f"{tag} x Alston"] = a["mmp"] * mmp_alston(s["T_F"], a["mw5"], vi) / mmp_alston(a["TF"], a["mw5"], vi)
    return direct, anchored


def shares(P_res, P_frac, direct_all, anch_all, r=0.0):
    """Shares with P_res < MMP and with a closed window (MMP >= P_frac), pooled over correlations."""
    na, nb = len(anch_all) // len(P_res), len(direct_all) // len(P_res)
    a_, b_ = anch_all * (1 - r), direct_all * (1 - r)
    return dict(sh_a=np.mean(np.tile(P_res, na) < a_), sh_b=np.mean(np.tile(P_res, nb) < b_),
                cl_a=np.mean(np.tile(P_frac, na) - a_ <= 0), cl_b=np.mean(np.tile(P_frac, nb) - b_ <= 0),
                ex=np.mean(P_res > P_frac))


def analog_keys(anchored, tag):
    return [k for k in anchored if k.startswith(tag)]


# ------------------------------------------------------------------------------------------
# 1. Benchmark of the correlations against measured slim-tube MMPs
# ------------------------------------------------------------------------------------------
def step_benchmark():
    header("1. BENCHMARK: correlations vs measured slim-tube MMP (Song et al., 2024)")
    errors = {"Cronquist": [], "Yellig-Metcalfe": [], "Alston": []}
    for name, a in ANALOGS.items():
        vi = a["vol"] / a["inter"]
        pred = {"Cronquist": mmp_cronquist(a["TF"], a["mw5"], a["c1"]),
                "Yellig-Metcalfe": mmp_yellig(a["TF"]),
                "Alston": mmp_alston(a["TF"], a["mw5"], vi)}
        print(f"{name}: T = {a['TF']:.0f} F, measured MMP = {a['mmp']:.1f} MPa")
        for k, v in pred.items():
            e = (v - a["mmp"]) / a["mmp"] * 100
            errors[k].append(e)
            print(f"   {k:16s} predicted {v:5.1f} MPa  error {e:+6.1f}%")
    print("Mean absolute error:", {k: f"{np.mean(np.abs(v)):.0f}%" for k, v in errors.items()})
    alt = []
    for name, a in ANALOGS.items():
        p = mmp_alston(a["TF"], a["mw5"], a["vol"] / (a["inter"] + a["c56"]))
        alt.append(f"{name[:5]} {p:.1f} MPa ({(p - a['mmp']) / a['mmp'] * 100:+.1f}%)")
    print("Alston with C5-C6 added to the intermediate fraction: " + " ; ".join(alt))


# ------------------------------------------------------------------------------------------
# 2-3. RQ1: reservoir state, MMP, and pressure window
# ------------------------------------------------------------------------------------------
def step_rq1(S, direct, anch):
    depth_m = S["depth_ft"] * FT
    header(f"2. RESERVOIR STATE (n = {N:,}, depth {depth_m / 1000:.2f} km)")
    pr("Temperature", S["T_C"], "C", "{:.0f}")
    pr("Reservoir pressure (hydrostatic)", S["P_res"])
    pr("Fracture pressure (assumed gradient)", S["P_frac"])
    pr("C5+ MW (API 30-38, Cragoe)", S["MW_c5"], "", "{:.0f}")
    pr("Live-oil C1 (GOR 300-900)", S["c1_mol"], "mol%", "{:.0f}")

    direct_all = np.concatenate(list(direct.values()))
    anch_all = np.concatenate(list(anch.values()))
    P_res, P_frac, T_F, T_C = S["P_res"], S["P_frac"], S["T_F"], S["T_C"]

    header("3 (RQ1). BROWN SHALE BULK MMP")
    print(f"(a) Direct correlations (median T = {np.median(T_F):.0f} F, above the calibration range of each correlation):")
    for k, v in direct.items():
        pr("   " + k, v)
    pr("   pooled direct", direct_all)
    print("(b) Anchored to the measured analog MMP and extrapolated in temperature:")
    for k, v in anch.items():
        pr("   " + k, v)
    pr("   pooled anchored", anch_all)

    ratio = np.tile(P_res, len(anch)) / anch_all
    print(f"Pres/MMP (anchored) P10/50/90: {q(ratio, 10):.2f} / {q(ratio, 50):.2f} / {q(ratio, 90):.2f}")
    print(f"Share with Pres < MMP: anchored = {np.mean(ratio < 1):.2f} ; direct = {np.mean(np.tile(P_res, 3) < direct_all):.2f}")
    win_a = np.tile(P_frac, len(anch)) - anch_all
    win_d = np.tile(P_frac, 3) - direct_all
    print(f"Window (Pfrac - MMP), anchored: P10/50/90 = {q(win_a, 10):.1f} / {q(win_a, 50):.1f} / {q(win_a, 90):.1f} MPa ; closed in {np.mean(win_a <= 0) * 100:.0f}% of cases")
    print(f"Window (Pfrac - MMP), direct  : P10/50/90 = {q(win_d, 10):.1f} / {q(win_d, 50):.1f} / {q(win_d, 90):.1f} MPa ; closed in {np.mean(win_d <= 0) * 100:.0f}% of cases")
    print(f"At the median temperature ({np.median(T_C):.0f} C) the six anchored estimates span "
          f"{min(np.median(v) for v in anch.values()):.1f}-{max(np.median(v) for v in anch.values()):.1f} MPa")

    header("3A (RQ1). PER-METHOD BREAKDOWN (the pooled values above are mixtures of these)")
    print(f"{'estimate':30s} | {'MMP P50':>7s} | {'Pres<MMP':>8s} | {'closed':>6s} | {'W P10/50/90 (MPa)':>20s}")
    groups = list(direct.items()) + [(f"Anchored {t} (3 corr. pooled)", np.concatenate([anch[k] for k in analog_keys(anch, t)]))
                                     for t in ("Oil#1", "Oil#2")]
    for k, v in groups:
        rep = len(v) // N
        w = np.tile(P_frac, rep) - v
        print(f"{k:30s} | {np.median(v):7.1f} | {np.mean(np.tile(P_res, rep) < v) * 100:7.1f}% | {np.mean(w <= 0) * 100:5.1f}% | "
              f"{q(w, 10):6.1f} / {q(w, 50):5.1f} / {q(w, 90):5.1f}")
    return direct_all, anch_all, win_a, win_d


# ------------------------------------------------------------------------------------------
# 3B-3G. RQ3: sensitivity analyses
# ------------------------------------------------------------------------------------------
def step_confinement(S, anch, direct_all, anch_all):
    # Source: Xiong et al. (2026) report MMP reductions of about 0.5% (50 nm), 9% (10 nm), and up to 16% (5 nm)
    # in idealized pores at lower temperatures. Applied here uniformly to every realization and correlation
    # as a scenario, not as a Brown Shale estimate.
    P_res, P_frac = S["P_res"], S["P_frac"]
    header("3B (RQ3). CONFINEMENT SENSITIVITY: MMP x (1 - r)")
    print(f"{'r':>6s} | {'Pres<MMP anch.':>14s} {'closed anch.':>13s} | {'Pres<MMP direct':>15s} {'closed direct':>13s} | median window anch. (MPa)")
    for r in CONFINEMENT_R:
        sh = shares(P_res, P_frac, direct_all, anch_all, r)
        w = q(np.tile(P_frac, len(anch)) - anch_all * (1 - r), 50)
        print(f"{r * 100:5.1f}% | {sh['sh_a'] * 100:13.1f}% {sh['cl_a'] * 100:12.1f}% | {sh['sh_b'] * 100:14.1f}% {sh['cl_b'] * 100:12.1f}% | {w:6.1f}")
    print("Per analog (anchored, three correlations pooled), share with Pres < MMP:")
    for tag in ("Oil#1", "Oil#2"):
        keys = analog_keys(anch, tag)
        v = np.concatenate([anch[k] for k in keys])
        Pr = np.tile(P_res, len(keys))
        print(f"   {tag}: " + " ; ".join(f"r={r * 100:.1f}% -> {np.mean(Pr < v * (1 - r)) * 100:.0f}%" for r in CONFINEMENT_R))


def step_pressure_gradient(S, anch, direct_all, anch_all):
    # Reservoir pressure is scaled by a uniform multiplier g (g = 1.18-1.27 spans the log-derived
    # gradients of Lukman, 2020). Fracture pressure is either independent of pore pressure or coupled
    # to it (dP_frac = ETA x dP_res).
    P_res, P_frac = S["P_res"], S["P_frac"]
    header("3C (RQ3). RESERVOIR-PRESSURE GRADIENT SENSITIVITY: P_res x g")
    print(f"{'g':>5s} | {'Pres P50':>8s} {'psi/ft':>7s} | {'Pres<MMP anch.':>14s} {'Pres<MMP direct':>15s} | {'Pres>Pfrac indep.':>17s} {'coupled':>8s} | {'closed anch. indep./coupled':>27s}")
    for g in PRESSURE_G:
        Pg = P_res * g
        sa = shares(Pg, P_frac, direct_all, anch_all)
        sc = shares(Pg, P_frac + ETA * (Pg - P_res), direct_all, anch_all)
        print(f"{g:5.1f} | {q(Pg, 50):8.1f} {q(Pg, 50) / PSI / DEPTH_FT:7.2f} | {sa['sh_a'] * 100:13.1f}% {sa['sh_b'] * 100:14.1f}% | "
              f"{sa['ex'] * 100:16.1f}% {sc['ex'] * 100:7.1f}% | {sa['cl_a'] * 100:12.1f}% /{sc['cl_a'] * 100:6.1f}%")
    print("Per analog (anchored, three correlations pooled), share with Pres < MMP:")
    for tag in ("Oil#1", "Oil#2"):
        keys = analog_keys(anch, tag)
        v = np.concatenate([anch[k] for k in keys])
        print(f"   {tag}: " + " ; ".join(f"g={g:.1f} -> {np.mean(np.tile(P_res * g, len(keys)) < v) * 100:.0f}%" for g in PRESSURE_G))


def step_log_derived_pressure(S, anch, direct_all, anch_all):
    # Source: Lukman (2020), well TM-01, Pematang Group; Eaton method from logs (indirect, single well):
    # 4,523 psi at 2,600 m and 4,891 psi at 2,640 m (0.53-0.57 psi/ft). Treated as a scenario.
    P_res, P_frac = S["P_res"], S["P_frac"]
    P_emp = np.random.default_rng(7).uniform(LUKMAN_PSI[0] * PSI, LUKMAN_PSI[1] * PSI, N)
    header(f"3D (RQ3). LOG-DERIVED OVERPRESSURE SCENARIO: P_res ~ U({LUKMAN_PSI[0] * PSI:.1f}, {LUKMAN_PSI[1] * PSI:.1f}) MPa")
    for label, Pf in (("P_frac independent (0.70-0.80 psi/ft)", P_frac),
                      (f"P_frac coupled (+{ETA} x dP_res)", P_frac + ETA * (P_emp - P_res))):
        sh = shares(P_emp, Pf, direct_all, anch_all)
        print(f"  {label:40s}: Pres<MMP anch. {sh['sh_a'] * 100:5.1f}% | direct {sh['sh_b'] * 100:5.1f}% | "
              f"closed anch. {sh['cl_a'] * 100:4.1f}% / direct {sh['cl_b'] * 100:4.1f}% | Pres>Pfrac {sh['ex'] * 100:4.1f}%")
    for tag in ("Oil#1", "Oil#2"):
        keys = analog_keys(anch, tag)
        v = np.concatenate([anch[k] for k in keys])
        print(f"   {tag} (anchored, pooled): Pres<MMP = {np.mean(np.tile(P_emp, len(keys)) < v) * 100:.0f}%")


def step_depth():
    # Depths bracket the planned total depth of Gulamo-1 (8,559 ft MD, vertical well); the Brown Shale
    # itself may lie shallower. Hydrostatic pressure and all other priors are unchanged.
    header("3E (RQ3). DEPTH SENSITIVITY (hydrostatic P_res; other priors unchanged)")
    print(f"{'depth':>14s} | {'T50 C':>5s} {'Pres50':>6s} {'Pfrac50':>7s} | {'MMP anch. P50':>13s} {'direct P50':>10s} | {'Pres<MMP anch.':>14s} {'direct':>6s} | {'closed anch.':>12s} {'direct':>6s}")
    for dft in (6000, 6600, 7500, DEPTH_FT, 10000):
        s = sample_state(dft)
        d, a = mmp_sets(s)
        dd, aa = np.concatenate(list(d.values())), np.concatenate(list(a.values()))
        sh = shares(s["P_res"], s["P_frac"], dd, aa)
        tag = " (base)" if dft == DEPTH_FT else ""
        print(f"{dft:6d} ft{dft * FT / 1000:5.2f}km | {np.median(s['T_C']):5.0f} {np.median(s['P_res']):6.1f} {np.median(s['P_frac']):7.1f} | "
              f"{np.median(aa):13.1f} {np.median(dd):10.1f} | {sh['sh_a'] * 100:13.1f}% {sh['sh_b'] * 100:5.1f}% | {sh['cl_a'] * 100:11.1f}% {sh['cl_b'] * 100:5.1f}%{tag}")


def step_geothermal():
    # The base case samples 40-60 C/km, a central part of the reported 30 to >120 C/km (Siringoringo et al.,
    # 2025). Here the gradient is fixed at values across that range; depth (2.6 km), pressures, and oil priors
    # keep their base distributions and random draws.
    header("3F (RQ3). GEOTHERMAL-GRADIENT SENSITIVITY (fixed G; other priors unchanged)")
    print(f"{'G C/km':>6s} | {'T C':>4s} | {'MMP P50 anch. #1/#2':>19s} | {'Pres<MMP anch. #1/#2':>20s} | {'closed anch. #1/#2':>18s} | "
          f"{'closed direct C/YM/A':>21s} | {'pooled: Pres<MMP a/d, closed a/d':>32s}")
    for G in GEOTHERMAL_G:
        s = sample_state(DEPTH_FT, fixed_G=G)
        d, a = mmp_sets(s)
        Pr, Pf = s["P_res"], s["P_frac"]
        A1 = np.concatenate([a[k] for k in analog_keys(a, "Oil#1")])
        A2 = np.concatenate([a[k] for k in analog_keys(a, "Oil#2")])
        Pr3, Pf3 = np.tile(Pr, 3), np.tile(Pf, 3)
        dd, aa = np.concatenate(list(d.values())), np.concatenate(list(a.values()))
        sh = shares(Pr, Pf, dd, aa)
        cd = [np.mean(Pf <= v) * 100 for v in d.values()]
        print(f"{G:6d} | {np.median(s['T_C']):4.0f} | {np.median(A1):8.1f} / {np.median(A2):5.1f}  | "
              f"{np.mean(Pr3 < A1) * 100:8.0f}% / {np.mean(Pr3 < A2) * 100:4.0f}%   | {np.mean(Pf3 <= A1) * 100:7.0f}% / {np.mean(Pf3 <= A2) * 100:4.0f}% | "
              f"{cd[0]:5.0f}% /{cd[1]:4.0f}% /{cd[2]:4.0f}% | {sh['sh_a'] * 100:5.0f}% / {sh['sh_b'] * 100:3.0f}%, {sh['cl_a'] * 100:4.0f}% / {sh['cl_b'] * 100:3.0f}%")


def step_oil_priors():
    # The anchored MMP does not depend on the oil priors (analog composition is held fixed);
    # only the direct Cronquist and Alston estimates do.
    header("3G (RQ3). OIL-PRIOR SENSITIVITY on the direct correlations")
    print(f"{'scenario':34s} | {'MMP P50 Cron / YM / Alston':28s} | {'pooled P50':>10s} | {'Pres<MMP':>8s} | {'closed':>6s}")
    scenarios = (("base: API 30-38, GOR 300/600/900", {}),
                 ("heavier: API 25-30", dict(api=(25, 30))),
                 ("lighter: API 38-47", dict(api=(38, 47))),
                 ("GOR low 150/300/450", dict(gor=(150, 300, 450))),
                 ("GOR high 600/1200/1800", dict(gor=(600, 1200, 1800))))
    for label, kw in scenarios:
        s = sample_state(DEPTH_FT, **kw)
        d, _ = mmp_sets(s)
        dd = np.concatenate(list(d.values()))
        Pr, Pf = np.tile(s["P_res"], 3), np.tile(s["P_frac"], 3)
        med = [np.median(v) for v in d.values()]
        print(f"{label:34s} | {med[0]:7.1f} / {med[1]:5.1f} / {med[2]:6.1f}     | {np.median(dd):10.1f} | {np.mean(Pr < dd) * 100:7.1f}% | {np.mean(Pf - dd <= 0) * 100:5.1f}%")


def step_bubble_point(S):
    # Yellig & Metcalfe (1980) also use the bubble-point pressure; the temperature term alone is used above.
    # Diagnostic with the Standing (1947) correlation as a proxy. Assumption: gas gravity from fC1 with a
    # 40 g/mol non-C1 fraction. Standing is calibrated to about 258 F and is extrapolated here.
    T_F, P_res = S["T_F"], S["P_res"]
    header("3H. BUBBLE-POINT DIAGNOSTIC (Standing proxy; base results unchanged)")
    gg = (16.04 * S["fC1"] + 40.0 * (1 - S["fC1"])) / 28.96
    Pb = 18.2 * (((S["GOR"] / gg) ** 0.83) * 10 ** (0.00091 * T_F - 0.0125 * S["API"]) - 1.4) * PSI
    ym = mmp_yellig(T_F)
    pr("Bubble point (Standing proxy)", Pb)
    print(f"Share with Pb > Yellig-Metcalfe MMP: {np.mean(Pb > ym) * 100:.1f}% ; share with Pb > hydrostatic P_res: {np.mean(Pb > P_res) * 100:.1f}%")
    print(f"Yellig-Metcalfe median {np.median(ym):.1f} -> max(MMP, Pb) {np.median(np.maximum(ym, Pb)):.1f} MPa ; "
          f"Pres < MMP {np.mean(P_res < ym) * 100:.1f}% -> {np.mean(P_res < np.maximum(ym, Pb)) * 100:.1f}%")


# ------------------------------------------------------------------------------------------
# 4. RQ2: analog injection design rescaled to the Brown Shale
# ------------------------------------------------------------------------------------------
def step_rq2(S, anch, direct_all, anch_all):
    # Source: Song et al. (2024): 60 MPa injection and 20 MPa production in reservoirs at 35.5 / 37.05 MPa.
    # The injection pressure is rescaled by the ratio P_inj / P_res of each analog.
    P_res, P_frac = S["P_res"], S["P_frac"]
    header("4 (RQ2). ANALOG INJECTION DESIGN (Song et al., 2024: 60 MPa injection, 20 MPa production)")
    print("Paired test: each analog design vs the anchored MMP of the same analog (three correlations pooled).")
    print("Categories: inside = MMP < Pinj < Pfrac ; below MMP = Pinj <= MMP < Pfrac ; above Pfrac = Pinj >= Pfrac (sum to 100%).")
    paired = {}
    for name, a in ANALOGS.items():
        tag = name[:5]
        r_inj, r_prod = 60 / a["Pres"], 20 / a["Pres"]
        Pinj = r_inj * P_res
        v = np.concatenate([anch[k] for k in analog_keys(anch, tag)])
        Pi, Pf = np.tile(Pinj, 3), np.tile(P_frac, 3)
        inside, above = (Pi > v) & (Pi < Pf), Pi >= Pf
        below = ~inside & ~above
        paired[tag] = dict(inside=np.mean(inside), above=np.mean(above), below=np.mean(below), Pinj50=q(Pinj, 50))
        print(f"{name}: Pinj/Pres = {r_inj:.2f} -> Pinj P50 = {q(Pinj, 50):.1f} MPa (Pprod/Pres = {r_prod:.2f} -> {q(r_prod * P_res, 50):.1f} MPa)")
        print(f"     paired anchored: inside {np.mean(inside) * 100:4.1f}% | below MMP {np.mean(below) * 100:4.1f}% | above Pfrac {np.mean(above) * 100:4.1f}%")
        Pi6, Pf6 = np.tile(Pinj, len(anch)), np.tile(P_frac, len(anch))
        print(f"     (pooled over both analogs' MMPs: inside {np.mean((Pi6 > anch_all) & (Pi6 < Pf6)) * 100:4.1f}%)")
        for dname, dv in mmp_sets(S)[0].items():
            ins = np.mean((Pinj > dv) & (Pinj < P_frac)) * 100
            print(f"     vs direct {dname:26s}: inside {ins:4.1f}%")
    print("Sensitivity to the rescaling rule: Pinj = MMP_BS x (Pinj,a / MMP_slim,a), i.e. the same margin above MMP as the analog")
    for name, a in ANALOGS.items():
        tag = name[:5]
        v = np.concatenate([anch[k] for k in analog_keys(anch, tag)])
        Pi = 60 / a["mmp"] * v
        print(f"   {tag}: ratio {60 / a['mmp']:.2f} -> Pinj P50 {np.median(Pi):.1f} MPa ; above Pfrac in {np.mean(Pi >= np.tile(P_frac, 3)) * 100:.0f}%")
    print(f"Direct use of 60 MPa exceeds Pfrac in {np.mean(60 > P_frac) * 100:.0f}% of cases")
    print("Note: the analog reservoirs lie above their own MMP (35.5 > 20.0 and 37.05 > 27.5 MPa), "
          "whereas the Brown Shale median lies below its MMP in most realizations.")
    return paired


# ------------------------------------------------------------------------------------------
# 5. RQ4: carbon break-even
# ------------------------------------------------------------------------------------------
def step_rq4():
    header("5 (RQ4). CARBON BREAK-EVEN")
    mscf_per_t = 379.5 / 44.01 * 2204.62 / 1000     # Mscf of CO2 per tonne at 60 F and 14.696 psia
    print(f"1 t CO2 = {mscf_per_t:.1f} Mscf ; break-even retention for combustion alone = {EF * mscf_per_t:.1f} Mscf/bbl")
    for e_proj in (0.05, 0.10, 0.15):                # Assumption: project emissions, t CO2/bbl (capture, compression, transport, field energy)
        need = (EF + e_proj) * mscf_per_t
        row = f"project {e_proj:.2f} t/bbl -> retained CO2 needed {need:4.1f} Mscf/bbl ; gross injection per incremental bbl: "
        row += "  ".join(f"{need / r:4.1f} @ {r * 100:.2f}%" for r in STORAGE_RATES.values())
        print(row)
    need = (EF + 0.10) * mscf_per_t
    print("Retention-fraction bracket at 0.10 t/bbl (storage fractions of Song et al. are single-cycle, matrix-only, mole-based):")
    for label, r in (("Wang et al. (2025), calcite nanopore MD", 0.14), ("Song et al. (2024), oil #1", 0.4476), ("Song et al. (2024), oil #2", 0.6933)):
        print(f"   f = {r * 100:5.1f}%  ({label:42s}) -> gross injection {need / r:5.1f} Mscf/bbl")
    print("Project emissions growing with gross injection: E_p = 0.10 t/bbl fixed + e x (t CO2 injected per bbl).")
    print("Mass balance f*m = Ec + Ep0 + e*m  ->  m* = (Ec + Ep0) / (f - e) ; no solution when f <= e.")
    for e in E_PROP:
        row = f"   e = {e:.2f}: "
        for label, r in (("f=14%", 0.14), ("f=44.76%", 0.4476), ("f=69.33%", 0.6933)):
            row += f"{label} -> " + (f"{(EF + 0.10) / (r - e) * mscf_per_t:5.1f} Mscf/bbl   " if r > e else "no solution      ")
        print(row)
    inc = 11_300 * 0.01
    print(f"One recovery point (1% of 11.3 BBO in place) = {inc:.0f} MMbbl -> combustion CO2 = {inc * EF:.1f} Mt")
    return need


def step_injection_intensity(need):
    # Derived (not reported by Song et al.): CO2 injected into the matrix per barrel of cumulative oil after half a year.
    # Cumulative oil includes non-incremental oil, so the value per INCREMENTAL barrel is larger than shown here.
    header("5A (RQ4). ANALOG INJECTION INTENSITY u (derived from Song et al., 2024 matrix totals)")
    mscf_per_t = 379.5 / 44.01 * 2204.62 / 1000
    for tag, d in SONG_MATRIX.items():
        t_co2 = d["inj_mol"] * 44.01 / 1e6
        bbl = d["oil_m3"] * 6.2898
        u = t_co2 * mscf_per_t / bbl
        f = 0.4476 if tag == "Oil#1" else 0.6933
        print(f"   {tag}: {t_co2 * 1000:.1f} kg CO2 / {bbl:.2f} bbl -> u = {u:.2f} Mscf/bbl (cumulative basis); "
              f"q* = {need / f:.1f} Mscf/bbl -> u/q* = {u / (need / f) * 100:.1f}%")


def step_link(paired):
    # The analog storage fractions come from injection above MMP and below no stated fracture limit. They are
    # transferable only where the rescaled analog design is inside the Brown Shale window (paired test, Step 4).
    header("5B (RQ2-RQ4 link). SHARE OF REALIZATIONS IN WHICH EACH ANALOG STORAGE RATE APPLIES")
    for tag, f in (("Oil#1", 0.4476), ("Oil#2", 0.6933)):
        print(f"   {tag}: f = {f * 100:.2f}% applies in {paired[tag]['inside'] * 100:.0f}% of realizations "
              f"(above Pfrac {paired[tag]['above'] * 100:.0f}%, below MMP {paired[tag]['below'] * 100:.0f}%)")


def step_convergence(seeds=range(1, 11)):
    header("9. MONTE CARLO CONVERGENCE (SD of headline shares over seeds 1-10, percentage points)")
    names = ["Pres<MMP anch.", "Pres<MMP direct", "closed anch.", "closed direct", "Pinj#1>Pfrac", "Pinj#2>Pfrac"]
    for n in (10_000, N):
        res = []
        for sd in seeds:
            rng_state = sample_state(DEPTH_FT, seed=sd)
            if n < N:
                rng_state = {k: (v[:n] if isinstance(v, np.ndarray) else v) for k, v in rng_state.items()}
            d, a = mmp_sets(rng_state)
            D, A = np.concatenate(list(d.values())), np.concatenate(list(a.values()))
            P, F = rng_state["P_res"], rng_state["P_frac"]
            res.append([np.mean(np.tile(P, 6) < A), np.mean(np.tile(P, 3) < D), np.mean(np.tile(F, 6) <= A),
                        np.mean(np.tile(F, 3) <= D), np.mean(60 / 35.5 * P > F), np.mean(60 / 37.05 * P > F)])
        sdv = np.std(np.array(res) * 100, axis=0, ddof=1)
        print(f"   N = {n:>7,}: " + " ; ".join(f"{k} {s:.2f}" for k, s in zip(names, sdv)))


# ------------------------------------------------------------------------------------------
# 6. Resource context
# ------------------------------------------------------------------------------------------
def step_resources():
    # Source: Direktorat Jenderal Migas (2026): 11.3 BBO in place (P50, North Aman Trough); initial phase of
    # 425 wells targets 201 MMBO and 191 Bcf over the contract term; 724 MMBOE is the 2C P50 recoverable estimate.
    header("6. RESOURCE CONTEXT")
    print(f"724/11,300 = {724 / 11300 * 100:.1f}% ; 201/11,300 = {201 / 11300 * 100:.1f}% ; "
          f"plan EUR per well = {201 / 425:.2f} MMBO ; 113/201 = {113 / 201 * 100:.0f}%")


# ------------------------------------------------------------------------------------------
# 8. Ranking of unknowns (tornado) and critical geothermal gradient (Section 4.4)
# ------------------------------------------------------------------------------------------
def window_metrics(P_res, P_frac, mmp):
    """Shares (%) with P_res < MMP and with a closed window, and median window (MPa), for one or pooled estimates."""
    rep = len(mmp) // len(P_res)
    Pr, Pf = np.tile(P_res, rep), np.tile(P_frac, rep)
    return dict(below=float(np.mean(Pr < mmp) * 100), closed=float(np.mean(Pf <= mmp) * 100),
                W50=float(np.median(Pf - mmp)))


def step_ranking(S, direct, anch):
    """One-at-a-time ranges of the pooled anchored metrics; MMP-estimate choice spans all five estimates."""
    header("7 (RQ3). RANKING OF UNKNOWNS (one-at-a-time ranges of pooled anchored metrics)")
    pool = lambda a: np.concatenate(list(a.values()))
    P_res, P_frac = S["P_res"], S["P_frac"]
    A = pool(anch)
    cases = {}
    cases["Geothermal gradient (30–80 °C/km)"] = []
    for G in GEOTHERMAL_G:
        s = sample_state(DEPTH_FT, fixed_G=G)
        cases["Geothermal gradient (30–80 °C/km)"].append(window_metrics(s["P_res"], s["P_frac"], pool(mmp_sets(s)[1])))
    A1 = np.concatenate([anch[k] for k in analog_keys(anch, "Oil#1")])
    A2 = np.concatenate([anch[k] for k in analog_keys(anch, "Oil#2")])
    cases["MMP estimate (3 direct, 2 anchored)"] = [window_metrics(P_res, P_frac, v) for v in list(direct.values()) + [A1, A2]]
    cases["Depth (6,000–10,000 ft)"] = []
    for ft in (6000, 6600, 7500, DEPTH_FT, 10000):
        s = sample_state(ft)
        cases["Depth (6,000–10,000 ft)"].append(window_metrics(s["P_res"], s["P_frac"], pool(mmp_sets(s)[1])))
    cases["Pore pressure (×1.0–1.6, coupled)"] = [window_metrics(P_res * g, P_frac + ETA * (P_res * g - P_res), A)
                                                  for g in PRESSURE_G]
    cases["Nanoconfinement (r = 0–32%)"] = [window_metrics(P_res, P_frac, A * (1 - r)) for r in CONFINEMENT_R]
    base = window_metrics(P_res, P_frac, A)
    out = {}
    print(f"{'factor':38s} | {'closed (%)':>14s} | {'Pres<MMP (%)':>14s}   base closed {base['closed']:.1f}, Pres<MMP {base['below']:.1f}")
    for k, v in cases.items():
        c = (min(x["closed"] for x in v), max(x["closed"] for x in v))
        b = (min(x["below"] for x in v), max(x["below"] for x in v))
        out[k] = dict(closed=c, below=b)
        print(f"{k:38s} | {c[0]:5.1f} – {c[1]:5.1f} | {b[0]:5.1f} – {b[1]:5.1f}")
    return dict(base=base, ranges=out)


def anchored_curve(TC, tag):
    """Mean of the three anchored MMPs (MPa) of one analog at temperature TC (deg C)."""
    a = ANALOGS[[k for k in ANALOGS if k.startswith(tag)][0]]
    TF, vi = TC * 9 / 5 + 32, a["vol"] / a["inter"]
    vals = [a["mmp"] * mmp_cronquist(TF, a["mw5"], a["c1"]) / mmp_cronquist(a["TF"], a["mw5"], a["c1"]),
            a["mmp"] * mmp_yellig(TF) / mmp_yellig(a["TF"]),
            a["mmp"] * mmp_alston(TF, a["mw5"], vi) / mmp_alston(a["TF"], a["mw5"], vi)]
    return np.min(vals, axis=0), np.mean(vals, axis=0), np.max(vals, axis=0)


def step_critical_gradient(S):
    """Temperature and geothermal gradient at which each analog's anchored MMP reaches median P_res and median P_frac."""
    header("8 (RQ3). CRITICAL TEMPERATURE AND GEOTHERMAL GRADIENT (anchored, mean of three correlations)")
    z_km, Ts = DEPTH_FT * FT / 1000, 28.0
    out = {}
    for tag in ("Oil#1", "Oil#2"):
        res = {}
        for label, target in (("Pres", float(np.median(S["P_res"]))), ("Pfrac", float(np.median(S["P_frac"])))):
            lo, hi = 50.0, 350.0
            for _ in range(60):  # bisection
                mid = (lo + hi) / 2
                (lo, hi) = (mid, hi) if anchored_curve(mid, tag)[1] < target else (lo, mid)
            res[label] = dict(T=lo, G=(lo - Ts) / z_km)
        out[tag] = res
        print(f"   {tag}: MMP = median Pres at {res['Pres']['T']:.0f} C (G = {res['Pres']['G']:.0f} C/km); "
              f"MMP = median Pfrac at {res['Pfrac']['T']:.0f} C (G = {res['Pfrac']['G']:.0f} C/km)")
    return out


def export_results(S, direct, anch, direct_all, anch_all, paired, ranking, critical, need, path="results.json"):
    """Writes the numbers used in Section 4 so that tables and figures read one source."""
    P_res, P_frac = S["P_res"], S["P_frac"]
    res = dict(state={k: [q(S[k], p) for p in (10, 50, 90)] for k in ("T_C", "P_res", "P_frac", "MW_c5", "c1_mol")})
    rows = {k: v for k, v in direct.items()}
    for tag in ("Oil#1", "Oil#2"):
        rows[f"Anchored {tag}"] = np.concatenate([anch[k] for k in analog_keys(anch, tag)])
    rows["Pooled direct"], rows["Pooled anchored"] = direct_all, anch_all
    t2 = {}
    for k, v in rows.items():
        rep = len(v) // N
        w = np.tile(P_frac, rep) - v
        t2[k] = dict(mmp50=q(v, 50), mmp10=q(v, 10), mmp90=q(v, 90), below=float(np.mean(np.tile(P_res, rep) < v) * 100),
                     closed=float(np.mean(w <= 0) * 100), W10=q(w, 10), W50=q(w, 50), W90=q(w, 90))
    res["table2"] = t2
    # RQ2: exclusive categories inside / below MMP / above Pfrac for pooled, paired, and MMP-ratio cases
    rq2 = {}
    for name, a in ANALOGS.items():
        tag = name[:5]
        Pinj = 60 / a["Pres"] * P_res
        own = np.concatenate([anch[k] for k in analog_keys(anch, tag)])
        for case, mm, Pi in (("pooled", anch_all, np.tile(Pinj, len(anch_all) // N)),
                             ("paired", own, np.tile(Pinj, 3)),
                             ("MMP-ratio rule", own, 60 / a["mmp"] * own)):
            Pf = np.tile(P_frac, len(mm) // N)
            above = Pi >= Pf
            inside = (Pi > mm) & ~above
            rq2[f"{case} {tag}"] = dict(inside=float(np.mean(inside) * 100), above=float(np.mean(above) * 100),
                                         below=float(100 - np.mean(inside) * 100 - np.mean(above) * 100),
                                         Pinj50=float(np.median(Pi)))
    res["rq2"] = rq2
    res["ranking"] = {k: dict(closed=list(v["closed"]), below=list(v["below"])) for k, v in ranking["ranges"].items()}
    res["ranking_base"] = ranking["base"]
    res["critical"] = critical
    mscf = 379.5 / 44.01 * 2204.62 / 1000
    res["rq4"] = dict(need_mscf=need, q_star={f"{f}": {f"Ep{ep}": (EF + ep) * mscf / f for ep in (0.05, 0.10, 0.15)} for f in (0.14, 0.4476, 0.6933)},
                      q_star_e={f"{f}": {f"e{e}": ((EF + 0.10) / (f - e) * mscf if f > e else None) for e in E_PROP} for f in (0.14, 0.4476, 0.6933)},
                      u={tag: d["inj_mol"] * 44.01 / 1e6 * mscf / (d["oil_m3"] * 6.2898) for tag, d in SONG_MATRIX.items()},
                      applies={tag: paired[tag]["inside"] * 100 for tag in paired})
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(f"\nResults written to {path}")
    return res


# ------------------------------------------------------------------------------------------
# Figures (Figure 1 is a schematic; Figures 3-5 are drawn from the numbers computed above)
# ------------------------------------------------------------------------------------------
CM = 1 / 2.54
FIG_STYLE = {
    "font.family": "Liberation Serif", "mathtext.fontset": "stix", "font.size": 9,
    "axes.linewidth": 0.8, "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "legend.fontsize": 8, "legend.frameon": False, "savefig.dpi": 300,
}
C1, C2, GREY, NAVY = "#1f5fa8", "#c0392b", "#6b6b6b", "#1f3a68"


def figure3(S, crit, path="Figure3_mmp_temperature.png"):
    """MMP versus temperature at the representative depth, with the sampled pressure and temperature ranges."""
    T = np.linspace(90, 240, 301)
    z_km, Ts = DEPTH_FT * FT / 1000, 28.0
    fig, ax = plt.subplots(figsize=(16 * CM, 10.0 * CM))
    t10, t50, t90 = (q(S["T_C"], p) for p in (10, 50, 90))
    pr = [q(S["P_res"], p) for p in (10, 50, 90)]
    pf = [q(S["P_frac"], p) for p in (10, 50, 90)]
    ax.axvspan(t10, t90, color="#efe9da", zorder=0)
    ax.axvline(t50, color=GREY, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.fill_between(T, pf[0], pf[2], color=NAVY, alpha=0.18, lw=0, zorder=1)
    ax.axhline(pf[1], color=NAVY, lw=1.4, zorder=2)
    ax.fill_between(T, pr[0], pr[2], color=GREY, alpha=0.25, lw=0, zorder=1)
    ax.axhline(pr[1], color=GREY, lw=1.2, ls=(0, (6, 2)), zorder=2)
    # direct correlations at the median sampled oil properties
    TF = T * 9 / 5 + 32
    mw, c1, vi = (float(np.median(S[k])) for k in ("MW_c5", "c1_mol", "vol_int"))
    ax.plot(T, mmp_cronquist(TF, mw, c1), color="black", lw=0.9, ls=(0, (1, 1.5)), zorder=3)
    ax.plot(T, mmp_alston(TF, mw, vi), color="black", lw=0.9, ls=(0, (5, 1.5, 1, 1.5)), zorder=3)
    ax.plot(T, mmp_yellig(TF), color="black", lw=0.9, ls=(0, (3, 2)), zorder=3)
    # anchored bands (min-max of three correlations) and their means
    for tag, col in (("Oil#1", C1), ("Oil#2", C2)):
        lo, mean, hi = anchored_curve(T, tag)
        ax.fill_between(T, lo, hi, color=col, alpha=0.30, lw=0, zorder=3)
        ax.plot(T, mean, color=col, lw=1.6, zorder=4)
        a = ANALOGS[[k for k in ANALOGS if k.startswith(tag)][0]]
        ax.plot(a["T_K"] - 273.15, a["mmp"], marker="s", ms=5, mfc=col, mec="black", mew=0.6, ls="none", zorder=6)
    # crossings with median reservoir and fracture pressure
    for tag, col in (("Oil#1", C1), ("Oil#2", C2)):
        for lab, yv in (("Pres", pr[1]), ("Pfrac", pf[1])):
            Tc = crit[tag][lab]["T"]
            if T[0] <= Tc <= T[-1]:
                ax.plot(Tc, yv, marker="o", ms=5, mfc="white", mec=col, mew=1.3, zorder=7)
                right = Tc > 200
                ax.annotate(f"{crit[tag][lab]['G']:.0f} °C/km", (Tc, yv), textcoords="offset points",
                            xytext=(-6 if right else 6, -11 if lab == "Pres" else 8), ha="right" if right else "left",
                            fontsize=7.5, color=col)
    ax.set(xlim=(T[0], T[-1]), ylim=(10, 70), xlabel="Reservoir temperature at 2.6 km (°C)", ylabel="Pressure (MPa)")
    top = ax.secondary_xaxis("top", functions=(lambda t: (t - Ts) / z_km, lambda g: g * z_km + Ts))
    top.set_xlabel("Equivalent geothermal gradient (°C/km)")
    top.tick_params(direction="in")
    handles = [Patch(fc=C1, alpha=0.4, label="Anchored, analog oil #1 (3 correlations)"),
               Patch(fc=C2, alpha=0.4, label="Anchored, analog oil #2 (3 correlations)"),
               Line2D([], [], color="black", lw=0.9, ls=(0, (1, 1.5)), label="Cronquist, direct"),
               Line2D([], [], color="black", lw=0.9, ls=(0, (5, 1.5, 1, 1.5)), label="Alston, direct"),
               Line2D([], [], color="black", lw=0.9, ls=(0, (3, 2)), label="Yellig–Metcalfe, direct"),
               Line2D([], [], color=NAVY, lw=1.4, label="Fracture pressure (P10–P50–P90)"),
               Line2D([], [], color=GREY, lw=1.2, ls=(0, (6, 2)), label="Reservoir pressure (P10–P50–P90)"),
               Line2D([], [], marker="s", color="w", mfc="white", mec="black", ms=5, label="Measured slim-tube MMP"),
               Line2D([], [], marker="o", color="w", mfc="white", mec="black", ms=5, label="Crossing of median pressure"),
               Patch(fc="#efe9da", label="Sampled temperature P10–P90 (median dashed)")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, columnspacing=1.6,
              handlelength=2.4)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.895, bottom=0.335)
    fig.savefig(path)
    plt.close(fig)


def figure4(res, path="Figure4_injection_transfer.png"):
    """Exclusive outcome shares of the rescaled analog injection design."""
    order = [("pooled Oil#1", "Oil #1, pooled anchored MMP"), ("paired Oil#1", "Oil #1, own anchored MMP"),
             ("MMP-ratio rule Oil#1", "Oil #1, MMP-ratio rule (Eq. 6)"),
             ("pooled Oil#2", "Oil #2, pooled anchored MMP"), ("paired Oil#2", "Oil #2, own anchored MMP"),
             ("MMP-ratio rule Oil#2", "Oil #2, MMP-ratio rule (Eq. 6)")]
    cols = {"inside": "#4c8a58", "below": "#e0b04a", "above": "#a33a32"}
    names = {"inside": "Inside window", "below": "Below MMP", "above": "At or above fracture pressure"}
    fig, ax = plt.subplots(figsize=(16 * CM, 5.8 * CM))
    y = np.array([0, 1, 2, 3.6, 4.6, 5.6])
    for yi, (key, lab) in zip(y, order):
        r = res["rq2"][key]
        left = 0
        for cat in ("inside", "below", "above"):
            w = r[cat]
            ax.barh(yi, w, left=left, height=0.72, color=cols[cat], ec="white", lw=0.6)
            if w >= 6:
                ax.text(left + w / 2, yi, f"{w:.0f}%", ha="center", va="center", fontsize=8,
                        color="white" if cat != "below" else "black")
            left += w
        ax.text(101.5, yi, f"{r['Pinj50']:.1f} MPa", va="center", fontsize=8)
    ax.set_yticks(y, [lab for _, lab in order])
    ax.invert_yaxis()
    ax.set(xlim=(0, 100), xlabel="Share of realizations (%)")
    ax.text(101.5, -0.85, "Median $P_{inj,s}$", fontsize=8, va="center")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(handles=[Patch(fc=cols[c], label=names[c]) for c in ("inside", "below", "above")],
              loc="upper center", bbox_to_anchor=(0.42, -0.2), ncol=3)
    fig.tight_layout(pad=0.3)
    fig.savefig(path)
    plt.close(fig)


def figure5(res, path="Figure5_ranking.png"):
    """One-at-a-time ranges of the pooled anchored metrics."""
    rk, base = res["ranking"], res["ranking_base"]
    facs = sorted(rk, key=lambda k: rk[k]["closed"][1] - rk[k]["closed"][0], reverse=True)
    fig, axs = plt.subplots(1, 2, figsize=(16 * CM, 6.2 * CM), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, key, b, title in ((axs[0], "closed", base["closed"], "(a) Closed window, MMP ≥ $P_{frac}$"),
                              (axs[1], "below", base["below"], "(b) Reservoir pressure below MMP")):
        for i, f in enumerate(facs):
            lo, hi = rk[f][key]
            ax.barh(i, max(hi - lo, 0.6), left=lo if hi - lo >= 0.6 else lo - 0.3, height=0.55, color=NAVY, alpha=0.85)
            ax.text(hi + 1.5, i, f"{lo:.0f}–{hi:.0f}", va="center", fontsize=7.5)
        ax.axvline(b, color=C2, lw=1.0, ls=(0, (4, 2)))
        ax.set(xlim=(0, 115), xlabel="Share of realizations (%)")
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.set_title(title, fontsize=9, loc="left")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axs[0].set_yticks(range(len(facs)), facs)
    axs[0].text(base["closed"] + 1.5, -0.62, "base case", color=C2, fontsize=7.5, va="center")
    axs[1].text(base["below"] - 1.5, -0.62, "base case", color=C2, fontsize=7.5, ha="right", va="center")
    axs[0].set_ylim(len(facs) - 0.5, -0.9)
    fig.subplots_adjust(left=0.30, right=0.985, top=0.88, bottom=0.2)
    fig.savefig(path)
    plt.close(fig)

def figure1(path="Figure1_concept.png"):
    """Schematic of the two screens (not computed, not to scale). (a) At fixed depth the MMP rises with temperature,
    whereas reservoir and fracture pressure do not; (b) the gross injection q* at which retained CO2 offsets the
    emissions rises as the retained fraction f falls (Eq. 7)."""
    GREEN, LGREEN = "#4c8a58", "#9cc5a1"
    fig = plt.figure(figsize=(16 * CM, 4.6 * CM))
    axa = fig.add_axes([0.04, 0.165, 0.235, 0.70])
    axb = fig.add_axes([0.535, 0.165, 0.22, 0.70])
    # (a) pressure window
    x = np.linspace(0, 1, 200)
    p_frac, p_res = 0.66, 0.40
    mmp = 0.22 + 0.62 * x ** 1.8
    mmp_c = 0.80 * mmp
    axa.fill_between(x, mmp, p_frac, where=mmp <= p_frac, color=LGREEN, alpha=0.55, lw=0, interpolate=True)
    axa.fill_between(x, p_frac, mmp, where=mmp >= p_frac, color="#c9c9c9", lw=0, interpolate=True)
    axa.axhline(p_frac, color=NAVY, lw=1.6)
    axa.axhline(p_res, color=GREY, lw=1.2, ls=(0, (6, 2)))
    axa.plot(x, mmp, color=C2, lw=1.6)
    axa.plot(x, mmp_c, color=C2, lw=1.0, ls=(0, (4, 2)))
    axa.set_ylim(0, 0.95)
    axa.set_xlabel("Temperature at fixed depth →", fontweight="bold")
    axa.set_ylabel("Pressure", fontweight="bold")
    axa.set_title("(a) Pressure window", fontweight="bold", fontsize=9.5)
    axa.legend(handles=[Patch(fc="#c9c9c9", label="Window closed"),
                        Patch(fc=LGREEN, alpha=0.55, label="Pressure window"),
                        Line2D([], [], color=NAVY, lw=1.6, label="Fracture pressure"),
                        Line2D([], [], color=GREY, lw=1.2, ls=(0, (6, 2)), label="Reservoir pressure"),
                        Line2D([], [], color=C2, lw=1.6, label="MMP (bulk fluid)"),
                        Line2D([], [], color=C2, lw=1.0, ls=(0, (4, 2)), label="MMP with\nnanoconfinement")],
               loc="center left", bbox_to_anchor=(1.02, 0.5), handlelength=1.9, labelspacing=0.5,
               fontsize=7.5, borderaxespad=0)
    # (b) retention burden
    q = np.linspace(0, 1, 50)
    emis = 0.55
    for f, ls, lab in ((0.95, "-", "Retained CO₂, high $f$"), (0.62, (0, (5, 2)), "Retained CO₂, low $f$")):
        axb.plot(q, f * q, color=GREEN, lw=1.6, ls=ls, label=lab)
        qs = emis / f
        axb.plot([qs, qs], [0, emis], color="black", lw=0.6, ls=(0, (1, 1.5)))
        axb.plot(qs, emis, "o", ms=4.5, color="black", zorder=5)
        axb.text(qs - 0.02, 0.03, "$q^*$", ha="right", va="bottom", fontsize=8.5)
    axb.axhline(emis, color="black", lw=1.2, ls=(0, (5, 3)),
                label="Emissions to offset\n(combustion + project)")
    axb.plot([], [], "o", ms=4.5, color="black", label="Neutrality, $q^*$")
    axb.annotate("", xy=(emis / 0.62 - 0.015, 0.17), xytext=(emis / 0.95 + 0.015, 0.17),
                 arrowprops=dict(arrowstyle="->", lw=0.8, color="black"))
    axb.text((emis / 0.62 + emis / 0.95) / 2, 0.20, "lower $f$", ha="center", va="bottom", fontsize=8)
    axb.set_ylim(0, 0.95)
    axb.set_xlabel("CO₂ injected per incremental barrel", fontweight="bold")
    axb.set_ylabel("CO₂ mass", fontweight="bold")
    axb.set_title("(b) Retention burden", fontweight="bold", fontsize=9.5)
    axb.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), handlelength=1.9, labelspacing=0.5, fontsize=7.5,
               borderaxespad=0)
    for ax in (axa, axb):
        ax.set_xlim(0, 1)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.savefig(path)
    plt.close(fig)


def make_figures(S, crit, res):
    """Writes Figure 1 (schematic) and Figures 3-5 (from this run)."""
    plt.rcParams.update(FIG_STYLE)
    figure1()
    figure3(S, crit)
    figure4(res)
    figure5(res)
    print("Figures written: Figure1_concept.png, Figure3_mmp_temperature.png, Figure4_injection_transfer.png, "
          "Figure5_ranking.png")


# ------------------------------------------------------------------------------------------
# Consistency check against the values reported in the manuscript
# ------------------------------------------------------------------------------------------
def check_reported(S, direct, anch, direct_all, anch_all, win_a, win_d, need, paired):
    P_res, P_frac = S["P_res"], S["P_frac"]
    Pi1, Pi2 = (60 / 35.50) * P_res, (60 / 37.05) * P_res
    within = lambda Pi: np.mean((np.tile(Pi, 6) > anch_all) & (np.tile(Pi, 6) < np.tile(P_frac, 6))) * 100
    A1 = np.concatenate([anch[k] for k in analog_keys(anch, "Oil#1")])
    A2 = np.concatenate([anch[k] for k in analog_keys(anch, "Oil#2")])
    Pf3 = np.tile(P_frac, 3)
    closed_d = [round(np.mean(P_frac <= v) * 100) for v in direct.values()]
    checks = {
        "anchored MMP medians 30.3 (#1) / 40.6 (#2) MPa": abs(np.median(A1) - 30.3) < 0.05 and abs(np.median(A2) - 40.6) < 0.05,
        "closed window per method: Cronquist 57%, Y-M 0%, Alston 59%": closed_d == [57, 0, 59],
        "closed window anchored: #1 0%, #2 14%": round(np.mean(Pf3 <= A1) * 100) == 0 and round(np.mean(Pf3 <= A2) * 100) == 14,
        "median window anchored: #1 13.9, #2 3.6 MPa": abs(np.median(Pf3 - A1) - 13.9) < 0.05 and abs(np.median(Pf3 - A2) - 3.6) < 0.05,
        "paired RQ2 inside window: 41.1% / 55.5%": abs(paired["Oil#1"]["inside"] * 100 - 41.1) < 0.05 and abs(paired["Oil#2"]["inside"] * 100 - 55.5) < 0.05,
        "median temperature 159 C": round(float(np.median(S["T_C"]))) == 159,
        "median reservoir pressure 26.5 MPa": abs(float(np.median(P_res)) - 26.5) < 0.05,
        "anchored P10-P90 28.5-43.0 MPa": abs(q(anch_all, 10) - 28.5) < 0.1 and abs(q(anch_all, 90) - 43.0) < 0.1,
        "Pres < MMP: direct 91%, anchored 99%": round(np.mean(np.tile(P_res, 3) < direct_all) * 100) == 91
                                                and round(np.mean(np.tile(P_res, 6) < anch_all) * 100) == 99,
        "window closed: anchored 7%, direct 39%": round(np.mean(win_a <= 0) * 100) == 7 and round(np.mean(win_d <= 0) * 100) == 39,
        "median window: anchored 9.2, direct 3.3 MPa": abs(q(win_a, 50) - 9.2) < 0.1 and abs(q(win_d, 50) - 3.3) < 0.1,
        "analog injection exceeds Pfrac: 59% / 27%": round(np.mean(Pi1 > P_frac) * 100) == 59 and round(np.mean(Pi2 > P_frac) * 100) == 27,
        "analog injection within window (anchored): 39% / 64%": round(within(Pi1)) == 39 and round(within(Pi2)) == 64,
        "gross injection 14.5-22.5 Mscf/bbl": abs(need / STORAGE_RATES["Song oil #2"] - 14.5) < 0.1
                                              and abs(need / STORAGE_RATES["Song oil #1"] - 22.5) < 0.1,
    }
    header("CONSISTENCY CHECK against values reported in the manuscript")
    for k, v in checks.items():
        print(f"   [{'OK' if v else 'MISMATCH'}] {k}")


# ------------------------------------------------------------------------------------------
def main():
    step_benchmark()
    S = sample_state(DEPTH_FT)
    direct, anch = mmp_sets(S)
    direct_all, anch_all, win_a, win_d = step_rq1(S, direct, anch)
    step_confinement(S, anch, direct_all, anch_all)
    step_pressure_gradient(S, anch, direct_all, anch_all)
    step_log_derived_pressure(S, anch, direct_all, anch_all)
    step_depth()
    step_geothermal()
    step_oil_priors()
    step_bubble_point(S)
    paired = step_rq2(S, anch, direct_all, anch_all)
    need = step_rq4()
    step_injection_intensity(need)
    step_link(paired)
    step_resources()
    ranking = step_ranking(S, direct, anch)
    critical = step_critical_gradient(S)
    step_convergence()
    res = export_results(S, direct, anch, direct_all, anch_all, paired, ranking, critical, need)
    make_figures(S, critical, res)
    check_reported(S, direct, anch, direct_all, anch_all, win_a, win_d, need, paired)


if __name__ == "__main__":
    main()
