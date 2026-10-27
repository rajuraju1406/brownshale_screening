# Brown Shale CO₂ Huff-n-Puff Screening

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Dependencies](https://img.shields.io/badge/dependencies-NumPy%20%7C%20Matplotlib-informational)
![Reproducible](https://img.shields.io/badge/runs-reproducible%20%28fixed%20seed%29-success)

Supplementary code for the paper **“Between Miscibility and Fracture Pressure: A Monte Carlo Screening of CO₂ Huff-n-Puff in the Pematang Brown Shale, Rokan Block, Central Sumatra Basin”** (GECo UI 2026 Paper & Poster Competition).

A single Python script reproduces every number, table value, and results figure reported in the paper. One command runs it in a few seconds.

---

## Contents

- [Overview](#overview)
- [Quick start](#quick-start)
- [Base-case results](#base-case-results)
- [Method](#method)
- [Script structure](#script-structure)
- [Inputs](#inputs)
- [Outputs](#outputs)
- [Reproducibility and verification](#reproducibility-and-verification)
- [Limitations](#limitations)
- [References](#references)
- [Citation](#citation)

---

## Overview

CO₂ huff-n-puff (HnP) could serve two national goals in Indonesia at once: raising oil recovery and storing CO₂. In the Pematang Brown Shale, two screens decide whether it can do both:

1. **Pressure window.** Miscible displacement requires an injection pressure above the minimum miscibility pressure (MMP). Indonesian regulation caps injection below the formation fracture pressure. At a fixed depth, the MMP rises with temperature, but reservoir and fracture pressure do not. A hot reservoir can therefore leave no window at all.
2. **Carbon balance.** If the combustion of the produced oil is counted, each incremental barrel carries a CO₂ retention requirement. That requirement grows as the retained fraction of the injected CO₂ falls.

No Brown Shale slim-tube, PVT, production, or CO₂ HnP field data were found in public sources. The script therefore propagates the uncertainty with a **Monte Carlo screening of 100,000 realizations at a representative depth of 2.6 km**, and it answers four research questions:

| RQ | Question |
|---|---|
| RQ1 | How often is reservoir pressure below the MMP, and how wide (or closed) is the MMP–fracture-pressure window? |
| RQ2 | How often does the analog injection design, rescaled to Brown Shale pressure, fall inside the window? |
| RQ3 | How robust are these answers to nanoconfinement, pore pressure, geothermal gradient, depth, oil properties, and the rescaling rule? |
| RQ4 | How much CO₂ must be retained or injected per incremental barrel for combustion-inclusive neutrality? |

> **Scope.** The shares reported here are frequencies conditional on assumed distributions and analog data. They are a screening envelope, not a field forecast.

---

## Quick start

```bash
git clone https://github.com/rajuraju1406/brownshale_screening.git
cd brownshale_screening
pip install -r requirements.txt        # or: pip install numpy matplotlib
python brownshale_screening.py > run_log.txt
```

The run prints a log of every step, writes `results.json` and four figures to the working directory, and ends with a consistency check against the values reported in the paper. All 14 checks should print `[OK]`.

**Requirements:** Python ≥ 3.9, NumPy, and Matplotlib (Matplotlib is used only for the figures). Tested with Python 3.13, NumPy 2.5, and Matplotlib 3.11.

> **Font note.** The figures use *Liberation Serif*, which is metric-compatible with Times New Roman. If that font is not installed (common on Windows), Matplotlib falls back to its default font and prints a warning. Only the typeface changes; the numbers do not.

---

## Base-case results

Representative depth 2.6 km, *N* = 100,000. The median sampled state is 159 °C, with a reservoir pressure of 26.5 MPa and a fracture pressure of 44.2 MPa.

| Metric (RQ1) | Direct correlations | Anchored to analog MMPs |
|---|---:|---:|
| Reservoir pressure below MMP | 91% | 99% |
| Median pressure window *W* | 3.3 MPa | 9.2 MPa |
| Window closed (MMP ≥ *P*<sub>frac</sub>) | 39% | 7% |

| Finding | Value |
|---|---|
| Anchored MMP (bimodal: analog oil #1 / #2) | 30.3 / 40.6 MPa median; P10–P90 28.5–43.0 MPa |
| Rescaled analog design inside the window (RQ2) | 39–64% (pooled); 41% / 55% (own analog MMP) |
| Rescaled analog design at or above fracture pressure (RQ2) | 27–59% |
| Gradient at which the mean anchored MMP reaches median *P*<sub>frac</sub> (RQ3) | ≈ 55 °C/km (oil #2) and ≈ 77 °C/km (oil #1) |
| CO₂ to retain per incremental barrel (RQ4) | ≈ 10 Mscf (0.43 t combustion + 0.10 t project emissions) |
| Gross injection needed at analog storage fractions (RQ4) | 14.5–22.5 Mscf/bbl |

Whether a window exists is controlled mainly by the **geothermal gradient** and the **reference MMP**. Whether reservoir pressure lies below the MMP is controlled mainly by **pore pressure** and **nanoconfinement**.

---

## Method

**Reservoir state and pressure window.** With surface temperature *T*<sub>s</sub>, geothermal gradient *G*, and pore-pressure and fracture gradients *g*<sub>p</sub> and *g*<sub>f</sub>:

$$T = T_s + Gz, \qquad P_{res} = g_p z, \qquad P_{frac} = g_f z, \qquad W = P_{frac} - \mathrm{MMP}$$

The window is open when *W* > 0 and closed when MMP ≥ *P*<sub>frac</sub>. *W* is an upper bound on the usable margin.

**MMP estimation.** Three empirical correlations are used: Cronquist (1978), Yellig and Metcalfe (1980), and Alston et al. (1985). Each is applied in two ways:
- **Direct:** the correlation as published, evaluated at the sampled temperature and oil properties.
- **Anchored:** the measured analog slim-tube MMP is kept, and the correlation supplies only its temperature dependence:

$$\mathrm{MMP}_{anch}(T) = \mathrm{MMP}_{slim}\,\frac{M(T)}{M(T_{ref})}$$

Nanoconfinement is represented as a scenario reduction, $\mathrm{MMP}_{conf} = (1-r)\,\mathrm{MMP}$.

**Analog injection design (RQ2).** The 60 MPa analog design is rescaled by the analog ratio of injection to reservoir pressure. A margin-preserving alternative is tested as a sensitivity:

$$P_{inj,s} = P_{res}\,\frac{P_{inj,a}}{P_{res,a}}, \qquad P^{(M)}_{inj,s} = \mathrm{MMP}_s\,\frac{P_{inj,a}}{\mathrm{MMP}_a}$$

**Carbon balance (RQ4).** *B* = *E*<sub>c</sub> + *E*<sub>p</sub> is the emission per barrel (combustion plus project). *f* is the retained fraction of injected CO₂, and *c* = 19.0 Mscf/t. The gross injection per incremental barrel needed for combustion-inclusive neutrality is:

$$q^{*} = \frac{cB}{f}, \qquad q^{*}_{e} = \frac{cB}{f-e} \quad (f > e)$$

where *e* is a proportional emission per tonne of CO₂ injected. No injection volume reaches neutrality when *f* ≤ *e*.

---

## Script structure

The script runs the steps below in order. Each step prints a numbered section in the log.

| Step | Calculation | Paper section |
|---|---|---|
| 1 | Benchmark of the three MMP correlations against the two slim-tube MMPs | 4.1 |
| 2 | Sampling of the reservoir state (Eq. 1, Table 1) | 3.2, 4.1 |
| 3, 3A | Direct and anchored MMP and pressure window, pooled and per estimate (Eqs. 2–3) | 4.2, Table 2 |
| 3B–3G | Sensitivity to confinement, pore pressure, depth, geothermal gradient, and oil properties | 4.4 |
| 3H | Diagnostic: Yellig–Metcalfe MMP floored at a Standing bubble point (not applied in the base case) | — |
| 4 | Rescaled analog injection design (Eqs. 5–6), exclusive outcome shares | 4.3, Figure 4 |
| 5, 5A, 5B | Carbon balance (Eqs. 7–8), analog injection intensity, applicability of the analog storage fractions | 4.5, Table 3 |
| 6 | Resource context (one recovery point) | 1, 4.5 |
| 7, 8 | Ranking of unknowns; critical temperature and geothermal gradient | 4.4, Figures 3 and 5 |
| 9 | Convergence over ten further seeds | 3.2 |
| — | Figure 1 (schematic, not computed) and Figures 3–5 (from the same run) | 1, 4 |

Each input in the code is tagged in a comment as **Source** (cited publication), **Analog** (borrowed from another basin), or **Assumption** (no Brown Shale measurement behind it).

---

## Inputs

| Parameter | Base case | Type | Source |
|---|---|---|---|
| Depth | 2.6 km (8,559 ft); scenarios 6,000–10,000 ft | Assumption | Planned total depth of Gulamo-1 (PT Pertamina, 2023) |
| Geothermal gradient | Triangular 40 / 50 / 60 °C/km; surface 27–29 °C; scenarios 30–80 °C/km | Assumption | Reported 30 to >120 °C/km (Siringoringo et al., 2025) |
| Reservoir pressure | Hydrostatic, 0.43–0.46 psi/ft; overpressure ×1.2–1.6 or 31.2–33.7 MPa | Assumption / Source | Normal-pressure class (U.S. EIA, 2015); log-derived (Lukman, 2020) |
| Fracture pressure | 0.70–0.80 psi/ft; coupling *η* = 0.67 in scenarios | Assumption | Typical shale gradient; *η* from Eaton (1969), *ν* = 0.25 |
| Oil properties | API 30–38°; GOR triangular 300 / 600 / 900 scf/STB; C₁ fraction 0.70–0.85 | Source / Assumption | GOR mode from Schenk et al. (2015); C₅₊ MW from API (Cragoe, 1929) |
| Analog slim-tube MMP | 20.00 and 27.50 MPa at 101.6 and 104.1 °C | Analog | Song et al. (2024) |
| Analog injection design | 60 MPa into 35.50 and 37.05 MPa reservoirs | Analog | Song et al. (2024) |
| Confinement reduction *r* | 0.5%, 9%, 16%; upper scenario 32% | Source | Xiong et al. (2026); T. Wan et al. (2024) |
| CO₂ storage fraction *f* | 44.76% and 69.33%; illustrative 14% | Analog / Source | Song et al. (2024); Wang et al. (2025) |
| Emissions | *E*<sub>c</sub> = 0.43 t CO₂/bbl; *E*<sub>p</sub> = 0.10 (0.05–0.15) t/bbl; *e* = 0.05, 0.10 t/t | Source / Assumption | U.S. EPA (2026); *E*<sub>p</sub> and *e* assumed |

The constants are defined at the top of `brownshale_screening.py`. Edit them there to rerun the screening as Brown Shale measurements become available.

---

## Outputs

| File | Content |
|---|---|
| console log | Every step with its intermediate values, the convergence test, and the consistency check |
| `results.json` | The numbers used in Tables 2–3 and Section 4 (state percentiles, window metrics per estimate, RQ2 shares, ranking ranges, critical gradients, carbon-balance thresholds) |
| `Figure1_concept.png` | Conceptual geometry of the two screens (schematic, not computed) |
| `Figure3_mmp_temperature.png` | MMP versus temperature at 2.6 km with sampled pressure and temperature ranges |
| `Figure4_injection_transfer.png` | Outcome shares of the rescaled analog injection design |
| `Figure5_ranking.png` | One-at-a-time ranking of the unknowns |

<p align="center">
  <img src="figures/Figure3_mmp_temperature.png" width="720" alt="MMP versus reservoir temperature at 2.6 km"><br>
  <em>Figure 3 of the paper: bulk MMP versus temperature at 2.6 km, with the gradients at which the anchored MMP reaches reservoir and fracture pressure.</em>
</p>

---

## Reproducibility and verification

- **Fixed seeds.** The base case uses seed 42 with *N* = 100,000 realizations. Scenarios reuse the same random draws, so their differences reflect only the changed assumption.
- **Convergence.** Across ten further seeds, the headline shares vary by 0.03–0.18 percentage points (standard deviation).
- **Consistency check.** The run ends by recomputing 14 headline values and comparing them with those reported in the paper.
- **Single source.** Tables and figures read the same `results.json`, so no value is typed in by hand.

---

## Limitations

- Bulk-fluid MMPs at one representative depth; nanoconfinement is treated only as a scenario reduction.
- Inputs are sampled independently. The fracture gradient is not varied separately from the base distribution.
- The analog slim-tube temperature is assumed equal to the analog reservoir temperature.
- Storage fractions are single-cycle, matrix-scale values from simulation, not field retention factors.
- Project emissions are assumed, and no economic analysis is included.

These limitations affect the precision of the shares more than the direction of the findings. The dominance of temperature and of the reference MMP follows from the window geometry itself.

---

## References

- Alston, R. B., Kokolis, G. P., & James, C. F. (1985). CO₂ minimum miscibility pressure: A correlation for impure CO₂ streams and live oil systems. *SPE Journal, 25*(2), 268–274. https://doi.org/10.2118/11959-PA
- Cragoe, C. S. (1929). *Thermal properties of petroleum products* (Miscellaneous Publication No. 97). U.S. Bureau of Standards.
- Cronquist, C. (1978). Carbon dioxide dynamic miscibility with light reservoir oils. *Proceedings of the Fourth Annual U.S. DOE Symposium*, Tulsa, Oklahoma, USA.
- Eaton, B. A. (1969). Fracture gradient prediction and its application in oilfield operations. *Journal of Petroleum Technology, 21*(10), 1353–1360. https://doi.org/10.2118/2163-PA
- Lukman, A. H. (2020). Integrated of geomechanics wellbore stability & sweet spot zone analysis to unconventional well drilling optimization. *Jurnal Petro, 9*(1), 21–29. https://doi.org/10.25105/petro.v9i1.6115
- PT Pertamina (Persero). (2023, July 27). *Pertamina’s inauguration of the first unconventional oil well in Rokan Block* [Press release].
- Schenk, C. J., et al. (2015). *Assessment of shale-oil resources of the Central Sumatra Basin, Indonesia, 2015* (Fact Sheet 2015–3072). U.S. Geological Survey. https://doi.org/10.3133/fs20153072
- Siringoringo, L. P., Situmeang, Z., & Meka, N. (2025). Central Sumatra Basin: The first sedimentary basin for geothermal energy development in Indonesia? *Rudarsko-geološko-naftni zbornik, 40*(1), 1–12. https://doi.org/10.17794/rgn.2025.1.1
- Song, Y., Song, Z., Chen, Z., Mo, Y., Zhou, Q., & Tian, S. (2024). Simulation of CO₂ enhanced oil recovery and storage in shale oil reservoirs: Unveiling the impacts of nano-confinement and oil composition. *Advances in Geo-Energy Research, 13*(2), 106–118. https://doi.org/10.46690/ager.2024.08.05
- U.S. Energy Information Administration. (2015). *Technically recoverable shale oil and shale gas resources: Indonesia*.
- U.S. Environmental Protection Agency. (2026). *Greenhouse gas equivalencies calculator – calculations and references*.
- Wan, T., Ding, K., Xiong, Q., & Guo, J. (2024). The phase behavior of CO₂ injection in shale reservoirs with nano-pores. *RSC Advances, 14*, 27227–27240. https://doi.org/10.1039/d4ra01239a
- Wang, S., Zhang, M., Zhang, Y., Lei, Z., Feng, Q., Xu, S., & Zhang, J. (2025). Mechanisms of CO₂ huff and puff enhanced oil recovery and storage within shale nanopores. *Chemical Engineering Journal, 506*, 160098. https://doi.org/10.1016/j.cej.2025.160098
- Xiong, Y., Chen, P.-F., Pu, W.-F., Jiang, R., & Pang, Q. (2026). Study on the phase behavior and minimum miscible pressure of CO₂-shale oil in nanopores with confinement effect. *Petroleum Science, 23*, 777–790. https://doi.org/10.1016/j.petsci.2025.11.039
- Yellig, W. F., & Metcalfe, R. S. (1980). Determination and prediction of CO₂ minimum miscibility pressures. *Journal of Petroleum Technology, 32*(1), 160–168. https://doi.org/10.2118/7477-PA

---

## Citation

If you use this code, please cite the paper:

```text
Petrovium (2026). Between Miscibility and Fracture Pressure: A Monte Carlo Screening of CO2
Huff-n-Puff in the Pematang Brown Shale, Rokan Block, Central Sumatra Basin.
GECo UI 2026 Paper & Poster Competition.
```
