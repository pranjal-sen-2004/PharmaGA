# PharmaGA

### A Niching Multi-Objective Genetic Algorithm with Surrogate-Assisted Fitness for Personalized Drug Selection

PharmaGA is a research-oriented computational framework for **patient-specific drug and drug-combination selection**. It formulates treatment selection as a constrained **multi-objective optimization** problem rather than reducing efficacy, toxicity, pharmacokinetic compatibility, and drug-drug interaction risk to a single weighted score.

The system combines four ideas:

1. **Patient-conditional multi-objective optimization** using demographic, biochemical, and pharmacogenomic information.
2. **Mechanism-Complementarity Crossover (MCX)** that performs genetic recombination using a drug-target graph instead of arbitrary chromosome positions.
3. **Surrogate-assisted fitness evaluation** using LightGBM to avoid unnecessary expensive oracle evaluations.
4. **Patient niching and personalization fidelity** to encourage solutions that are meaningfully different for genetically different patients.

> **Important:** PharmaGA is a research/educational prototype and is **not a clinical decision-making system**. Its benchmark uses simulated patients and computational efficacy/ADMET proxies. It must not be used to prescribe or alter medication without qualified clinical review and formal validation.

---

## Why PharmaGA?

Choosing a treatment becomes combinatorially difficult when a patient may receive a combination of drugs and several objectives must be optimized simultaneously. The project models a regimen as a variable-length list of `(drug, dose-bin)` pairs, with a maximum of four drugs.

For a patient profile `p`, the optimizer searches for a Pareto set over:

| Objective | Meaning | Direction |
|---|---|---|
| **Efficacy** `f1` | Predicted probability that a regimen meets the target endpoint | Maximize |
| **Toxicity** `f2` | Aggregate cardio-, hepato-, nephro-, and idiosyncratic-risk burden | Minimize |
| **PK mismatch** `f3` | Penalty when drug metabolism conflicts with patient genotype/clearance | Minimize |
| **DDI risk** `f4` | Pairwise interaction risk from CYP inhibition/induction relationships | Minimize |

Instead of returning one supposedly “best” regimen, PharmaGA returns a **Pareto archive** containing trade-off solutions that can be inspected by objective preference.

---

## Core Architecture

```text
                         ┌───────────────────────┐
                         │      Drug Library     │
                         │ descriptors / ADMET  │
                         │ targets / fingerprints│
                         └──────────┬────────────┘
                                    │
                         ┌──────────▼────────────┐
                         │   Patient Profile     │
                         │ demographics / labs   │
                         │ pharmacogenomics      │
                         └──────────┬────────────┘
                                    │
                         ┌──────────▼────────────┐
                         │ Target / Mechanism    │
                         │ Bipartite Graph       │
                         └──────────┬────────────┘
                                    │
                   ┌────────────────▼────────────────┐
                   │     Stratified Initialization   │
                   │ random + guideline + DPP seeds  │
                   └────────────────┬────────────────┘
                                    │
                   ┌────────────────▼────────────────┐
                   │       Niching Tournament        │
                   └────────────────┬────────────────┘
                                    │
                   ┌────────────────▼────────────────┐
                   │ Mechanism-Complementarity      │
                   │ Crossover (MCX)                 │
                   └────────────────┬────────────────┘
                                    │
                   ┌────────────────▼────────────────┐
                   │       Adaptive Mutation         │
                   │ drug / dose / regimen size      │
                   └────────────────┬────────────────┘
                                    │
                   ┌────────────────▼────────────────┐
                   │    LightGBM Surrogate           │
                   │ prediction + uncertainty        │
                   └───────────────┬─────────────────┘
                                   │
                         uncertainty > threshold?
                              /              \
                            yes              no
                            /                  \
                 ┌──────────▼─────────┐    ┌──────────────┐
                 │ Expensive Oracle  │    │ Accept Sur-  │
                 │ docking + ADMET   │    │ rogate Score │
                 └──────────┬─────────┘    └───────┬──────┘
                            │                       │
                            └──────────┬────────────┘
                                       │
                              ┌────────▼────────┐
                              │ Environmental   │
                              │ Selection/NSGA  │
                              └────────┬────────┘
                                       │
                              ┌────────▼────────┐
                              │ Pareto Archive  │
                              │ Ranked Regimens │
                              └─────────────────┘
```

The implementation is described as a client-server application: a **Python/Flask backend** contains the optimization engine, surrogate, and oracle interface, while a **React frontend** collects patient data and visualizes the Pareto archive.

---

## Algorithm

### 1. Regimen Representation

A chromosome is represented as:

```text
s = [(drug_id, dose_bin), ..., (drug_id, dose_bin)]
```

The regimen has variable length up to `Kmax = 4`, with five discrete dose bins in the benchmark configuration.

### 2. Stratified Initialization

The reference configuration uses a population of `mu = 200` and seeds it using:

- 30% random regimens for exploration.
- 30% guideline-inspired monotherapy regimens for clinically meaningful exploitation.
- 40% DPP-based samples over the drug-target graph to encourage mechanistic diversity.

### 3. MCX — Mechanism-Complementarity Crossover

Traditional one-point or uniform crossover can break biologically meaningful drug combinations because chromosome position does not represent pharmacological structure.

MCX instead:

1. Collects drugs from both parent chromosomes.
2. Builds a drug-target bipartite graph.
3. Groups drugs into mechanistically coherent clusters.
4. Transfers complete clusters to offspring.
5. Enforces the maximum regimen size.

This is the key domain-specific operator in PharmaGA.

### 4. Adaptive Mutation

Three mutations are used:

- **Drug substitution:** replace a drug with a molecular-fingerprint neighbour (Tanimoto threshold >= 0.6 in the benchmark).
- **Dose perturbation:** move a dose bin by `+/-1`.
- **Regimen resize:** add/remove a drug based on diversity.

The mutation rate is adapted using a Rechenberg-style update based on the fraction of successful/non-dominated mutations.

### 5. Surrogate-Assisted Fitness

Every new candidate is first evaluated by a **LightGBM quantile-regression surrogate**. Its prediction uncertainty is used as a gate:

```text
candidate
   │
   ▼
LightGBM prediction + uncertainty
   │
   ├── low uncertainty ──> use surrogate fitness
   │
   └── high uncertainty ─> expensive oracle
                               │
                               ▼
                         update training set
```

The reference setup uses an adaptive threshold targeting roughly one-third oracle usage.

### 6. NSGA-II Environmental Selection + Niching

Parent and offspring populations are merged and ranked through non-dominated sorting. Crowding distance is augmented with a **patient-niching term** so the search favours objective landscapes that remain informative about patient-specific covariates.

An external Pareto archive is maintained for all non-dominated regimens discovered during the run.

---

## Benchmark Configuration

The research benchmark described with this project uses:

| Parameter | Configuration |
|---|---:|
| Drug library | **1,842 compounds** |
| Patient profiles | **500 synthetic profiles** |
| Therapeutic areas | **4** |
| Population size | `200` |
| Offspring size | `200` |
| Generations | `150` |
| Crossover probability | `0.90` |
| Initial mutation rate | `0.15` |
| Maximum regimen size | `4 drugs` |
| Dose bins | `5` |
| Archive capacity | `500` |
| Oracle budget | `5,000` calls/run |
| Random seeds | `30` |
| Surrogate | `LightGBM`, 500 trees, quantile loss |

### Therapeutic Areas

- Hypertension
- Non-small-cell lung cancer (NSCLC)
- Type-2 diabetes mellitus (T2DM)
- Major depressive disorder (MDD)

### Pharmacogenomic Inputs

The benchmark models actionable pharmacogenes including:

`CYP2D6`, `CYP2C19`, `CYP2C9`, `CYP3A5`, `SLCO1B1`, `TPMT`, `DPYD`, `UGT1A1`, `VKORC1`, `HLA-B`, `HLA-A`, and `NAT2`.

Patient profiles combine demographic/biochemical variables such as age, sex, BMI, and eGFR with comorbidity information and pharmacogenomic diplotype/phenotype inputs.

---

## Data and Computational Oracles

The reference benchmark describes a drug library assembled from **DrugBank 5.1** and **ChEMBL 33**, filtered for drug-likeness. Compounds contain identifiers, ATC information, molecular fingerprints, physicochemical descriptors, ADMET predictions, and target annotations.

Patient profiles are synthetic, generated by combining pharmacogenomic distributions from the **1000 Genomes Project** with comorbidity vectors from **Synthea**.

The oracle layer varies by therapeutic area:

- **Hypertension:** pre-computed QSAR efficacy lookups.
- **T2DM:** pre-computed QSAR efficacy lookups.
- **NSCLC:** patient-driver-specific AutoDock Vina docking for EGFR/ALK/KRAS contexts.
- **MDD:** receptor-binding-profile similarity as the efficacy proxy.
- **ADMET / PK:** pre-computed risk tables in the benchmark.

---

## Results

The reported research benchmark compares PharmaGA with Random Search, NSGA-II, SPEA2, MOEA/D, and a policy-gradient RL baseline.

| Method | HV ↑ | IGD ↓ | PF ↑ | HV / 1k Oracle | Runtime (s) |
|---|---:|---:|---:|---:|---:|
| Random | 0.312 | 0.287 | 0.54 | 0.062 | 41 |
| NSGA-II | 0.601 | 0.141 | 0.63 | 0.120 | 188 |
| SPEA2 | 0.589 | 0.148 | 0.61 | 0.118 | 204 |
| MOEA/D | 0.612 | 0.133 | 0.65 | 0.122 | 172 |
| PG-RL | 0.578 | 0.158 | 0.67 | 0.116 | 395 |
| **PharmaGA** | **0.688** | **0.097** | **0.82** | **0.244** | **167** |

Reported headline improvements include:

- **+12.4% hypervolume** over MOEA/D.
- **-27.1% IGD** versus MOEA/D.
- **0.82 personalization fidelity**, compared with `0.65` for MOEA/D.
- Approximately **38x reduction in wall-clock computation for surrogate-assisted oracle evaluation** in the described setup, while retaining about **94% rank correlation** with oracle rankings.

### Per-area Hypervolume

| Therapeutic Area | NSGA-II | MOEA/D | PG-RL | PharmaGA |
|---|---:|---:|---:|---:|
| Hypertension | 0.644 | 0.659 | 0.608 | **0.731** |
| NSCLC | 0.522 | 0.537 | 0.551 | **0.616** |
| T2DM | 0.619 | 0.628 | 0.582 | **0.697** |
| MDD | 0.619 | 0.624 | 0.571 | **0.708** |

---

## Ablation Study

One of the strongest parts of the project is the component ablation analysis.

| Variant | HV ↑ | IGD ↓ | PF ↑ |
|---|---:|---:|---:|
| **Full PharmaGA** | **0.688** | **0.097** | **0.82** |
| - Surrogate | 0.652 | 0.111 | 0.81 |
| - MCX | 0.631 | 0.128 | 0.74 |
| - Patient niching | 0.672 | 0.102 | 0.69 |
| - Pharmacogenomic conditioning | 0.665 | 0.109 | 0.58 |
| - Adaptive mutation | 0.676 | 0.104 | 0.80 |

The largest degradation comes from removing **MCX**, while the personalization fidelity metric is especially sensitive to **pharmacogenomic conditioning and patient niching**.

---

## Personalization Case Study

A highlighted experiment keeps demographics fixed while changing only CYP2D6 phenotype:

| Patient | CYP2D6 phenotype | PharmaGA outcome |
|---|---|---|
| A | Ultra-rapid metabolizer | Dose-escalated venlafaxine-containing regimens |
| B | Poor metabolizer | CYP2D6-independent options centered on sertraline / mirtazapine |

The reported Jaccard overlap between the two PharmaGA Pareto fronts is **0.11**, versus **0.71 for NSGA-II** and **0.64 for MOEA/D**, illustrating that the patient-conditioning mechanism changes the returned solution set instead of simply ranking the same population-wide options.

---

## Parallel Processing

PharmaGA has several natural parallelism axes:

### Population / Oracle Parallelism

Oracle evaluations within a generation are independent and can be executed concurrently.

### Patient-Cohort Parallelism

Each patient is an independent optimization problem, enabling near-embarrassingly-parallel execution across a cohort.

### Surrogate Training / Inference

LightGBM can exploit multi-core execution during model training and inference.

### Non-dominated Sorting

Dominance comparisons can be parallelized, although this contributes less runtime than expensive oracle evaluation.

The project report describes a reduction from approximately **167 seconds on one core to ~32 seconds using 8-core oracle parallelism**, and a **64-core cohort configuration around ~38 seconds per patient** under its reported benchmark assumptions.

---

## Web Application

The documented implementation uses a client-server design.

### Backend

- Python
- Flask REST API
- PharmaGA evolutionary engine
- LightGBM surrogate
- Oracle abstraction
- NetworkX-based drug-target graph logic

Documented API endpoints include:

```http
POST /api/run_pharmaga
GET  /api/drugs
```

### Frontend

- React single-page application
- Patient profile form
- Therapeutic-area / oracle configuration panel
- Pareto result table
- Objective-pair scatter plot
- Expandable regimen details
- Drug detail view with descriptors, ADMET flags, and molecular targets

---

## Recommended Project Structure

A production-oriented layout for this project is:

```text
PharmaGA/
├── backend/
│   ├── app.py
│   ├── api/
│   ├── core/
│   │   ├── ga.py
│   │   ├── chromosome.py
│   │   ├── selection.py
│   │   ├── mutation.py
│   │   ├── mcx.py
│   │   ├── niching.py
│   │   └── archive.py
│   ├── surrogate/
│   │   ├── lightgbm_model.py
│   │   └── uncertainty.py
│   ├── oracle/
│   │   ├── base.py
│   │   ├── docking.py
│   │   ├── admet.py
│   │   └── qsar.py
│   ├── data/
│   └── tests/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── services/
│   │   └── charts/
│   └── package.json
├── configs/
│   ├── default.yaml
│   └── experiments/
├── notebooks/
├── docs/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

# High-Impact Enhancements

The current research design is strong, but the next version can be made significantly more robust, explainable, and portfolio-ready.

## 1. Add an Explainability Layer ⭐⭐⭐⭐⭐

The current output can tell a user which regimen is on the Pareto front, but a strong clinical-research interface should also explain **why** it was selected.

Add an explanation object for every regimen:

```json
{
  "regimen": ["Drug A", "Drug B"],
  "why_selected": [
    "high predicted efficacy",
    "low CYP2D6 mismatch",
    "acceptable DDI risk"
  ],
  "patient_factors": [
    "CYP2D6 poor metabolizer",
    "reduced renal clearance"
  ],
  "tradeoffs": {
    "efficacy_vs_toxicity": "high efficacy / moderate toxicity"
  }
}
```

For the surrogate, add SHAP or permutation-based contribution analysis. For the optimization layer, expose objective values, constraint violations avoided, mechanistic clusters, and sensitivity to patient features.

**Why this matters:** it turns a research optimizer into a system that humans can interrogate rather than a black-box list generator.

---

## 2. Replace Binary Metabolizer Categories with Quantitative PGx Features ⭐⭐⭐⭐⭐

The documented implementation compresses diplotypes into categories such as poor/intermediate/normal/rapid/ultra-rapid metabolizer. This loses information about quantitative enzyme activity and clearance.

A better pipeline is:

```text
VCF / diplotype
      ↓
pharmacogene allele interpretation
      ↓
activity score / phenotype probability
      ↓
quantitative clearance estimate
      ↓
PK-mismatch objective
```

This directly addresses one of the current limitations identified by the project report.

---

## 3. Upgrade Surrogate Uncertainty to Conformal Prediction ⭐⭐⭐⭐⭐

The current LightGBM quantile model provides an uncertainty proxy, but the project already identifies the need for **conformal-prediction-calibrated uncertainty**.

Upgrade to:

```text
Base model
  ↓
Calibration set
  ↓
Conformal interval
  ↓
Coverage-aware uncertainty
  ↓
Oracle routing decision
```

Then benchmark:

- coverage probability
- interval width
- oracle calls saved
- hypervolume at fixed budget
- ranking quality

This would create a much stronger research contribution than simply swapping one regression model for another.

---

## 4. Introduce Multi-Fidelity Evaluation ⭐⭐⭐⭐⭐

Instead of a binary `surrogate OR expensive oracle` decision, use several fidelity levels:

```text
Level 0 → cheap rules / cached scores
Level 1 → LightGBM surrogate
Level 2 → QSAR / ADMET models
Level 3 → molecular docking
Level 4 → high-fidelity simulation / experimental evidence
```

Candidates move upward only when their expected value justifies the cost.

This naturally extends the current uncertainty-gated oracle idea and can substantially improve oracle efficiency.

---

## 5. Make DDI/Safety Constraints a First-Class Guardrail ⭐⭐⭐⭐⭐

Keep hard safety rules outside the optimizer so a prediction model cannot trade safety away for efficacy.

Example:

```text
Patient input
   ↓
Safety / contraindication engine
   ↓
Allowed candidate pool
   ↓
PharmaGA optimization
   ↓
Pareto ranking
   ↓
Final safety re-check
```

Add rule categories for:

- absolute contraindications
- gene-drug warnings
- renal/hepatic dose restrictions
- duplicate therapy
- severe CYP conflicts
- known high-risk combinations
- maximum regimen size

This should be a **deterministic guardrail layer**, not something learned by the surrogate.

---

## 6. Add a Temporal Treatment Planner ⭐⭐⭐⭐

The current model is essentially a static “choose a regimen now” problem.

A stronger formulation is:

```text
Patient state t0
     ↓
Induction regimen
     ↓
Patient response / adverse events
     ↓
Updated state t1
     ↓
Consolidation regimen
     ↓
Updated state t2
     ↓
Maintenance regimen
```

This turns PharmaGA into a sequential decision problem and creates a natural bridge to reinforcement learning, model predictive control, or planning algorithms.

---

## 7. Add a 4D Pareto Visualization ⭐⭐⭐⭐

The frontend should go beyond a single 2D scatter plot.

Add:

- parallel-coordinates plot across all four objectives
- Pareto knee-point highlighting
- dominated/non-dominated toggle
- regimen comparison mode
- patient-vs-patient comparison
- uncertainty overlay
- objective-weight sliders that **filter** the Pareto set rather than collapsing it into a single hidden score

A particularly strong demo feature would be:

> **Change only CYP2D6 phenotype → rerun → visualize how the Pareto front moves.**

---

## 8. Add Oracle Result Caching ⭐⭐⭐⭐

Docking and other oracle calls are expensive. Cache by a deterministic regimen signature:

```text
hash(
    disease,
    patient-driver-context,
    drug_ids,
    dose_bins,
    oracle_version
)
```

Then:

```text
candidate → cache lookup
              ├── hit  → return result
              └── miss → run oracle → cache → return
```

This prevents repeated evaluation of identical regimens across generations and makes experiments more reproducible.

---

## 9. Parallelize the Application Properly ⭐⭐⭐⭐

For a portfolio-grade implementation, move long-running optimization jobs out of the Flask request lifecycle.

Recommended architecture:

```text
React
  ↓
Flask/FastAPI API
  ↓
Redis / message broker
  ↓
Worker pool
 ┌────────┬────────┬────────┐
 │Worker 1│Worker 2│Worker N│
 └────────┴────────┴────────┘
  ↓
Result store
  ↓
Frontend polling / WebSocket
```

Good choices include Celery/RQ plus Redis for a simple deployment, or a containerized worker service for larger experiments.

Benefits:

- non-blocking UI
- multiple patients in parallel
- resilient long-running jobs
- progress tracking
- retries for failed docking jobs
- easier cloud deployment

---

## 10. Add Real Experiment Tracking ⭐⭐⭐⭐

Store every experiment as a versioned run:

```yaml
seed: 17
population: 200
generations: 150
oracle_budget: 5000
surrogate: lightgbm_quantile
mcx: true
patient_niching: true
pgx_conditioning: true
```

Track:

- seed
- configuration
- git commit
- dataset version
- objective distributions
- oracle count
- surrogate accuracy
- runtime
- HV / IGD / PF

MLflow, Weights & Biases, or a lightweight local JSON/Parquet run store would work.

---

## 11. Add a Proper Reproducibility / MLOps Layer ⭐⭐⭐⭐

The next release should include:

- fixed data snapshots
- dataset checksums
- configuration files
- deterministic seeds
- environment lockfile
- Docker image
- automated test suite
- benchmark command
- CI workflow

A user should be able to run one command such as:

```bash
make benchmark
```

and reproduce the main table from the paper.

---

## 12. Improve Clinical Validation ⭐⭐⭐⭐⭐

This is the most important research-level enhancement.

The current evaluation relies on **synthetic patient profiles**, which is appropriate for a prototype but insufficient for clinical claims.

A stronger validation ladder is:

```text
Synthetic benchmark
      ↓
Retrospective de-identified EHR validation
      ↓
External cohort validation
      ↓
Calibration / subgroup analysis
      ↓
Clinician review
      ↓
Prospective study
```

Measure not only optimization metrics, but also:

- agreement with physician decisions
- contraindication recall
- calibration
- subgroup performance
- sensitivity to missing patient data
- stability under small input perturbations
- false-safe / false-dangerous recommendation rate

---

# Recommended Roadmap

## Phase 1 — Portfolio Upgrade

Implement these first because they provide the biggest visible improvement for relatively little engineering effort:

1. Explainable regimen cards.
2. Parallel-coordinates + Pareto knee visualization.
3. Patient A vs Patient B comparison.
4. Oracle-result cache.
5. Config-based reproducible experiments.
6. Dockerized backend/frontend.
7. Strong README + architecture diagram + benchmark command.

## Phase 2 — Research Upgrade

1. Conformal uncertainty.
2. Quantitative PGx features.
3. Multi-fidelity oracle routing.
4. Expanded DDI/safety engine.
5. More optimization baselines.
6. Statistical significance testing with confidence intervals.

## Phase 3 — Advanced Research

1. Temporal treatment planning.
2. Real-world EHR/PGx validation.
3. External-cohort validation.
4. Human/clinician-in-the-loop ranking.
5. Prospective evaluation.

---

# Suggested New Features for a Strong Demo

A polished demo can expose three workflows:

### Patient-Specific Optimization

```text
Enter patient data
       ↓
Select disease
       ↓
Run PharmaGA
       ↓
View Pareto front
       ↓
Inspect regimen explanations
```

### Genomic Sensitivity

```text
Patient A: CYP2D6 ultra-rapid
        ↓
     PharmaGA
        ↓
 Pareto Front A

Patient B: CYP2D6 poor
        ↓
     PharmaGA
        ↓
 Pareto Front B

Compare fronts → personalization signal
```

### Drug Repurposing

Keep the patient profile fixed and change the disease/target context to search the existing compound library for alternative indications. This uses the same optimization framework without requiring de novo molecule generation.

---

# Testing Strategy

At minimum, add:

```text
Unit tests
├── chromosome encoding
├── feasibility constraints
├── MCX invariants
├── mutation operators
├── Pareto dominance
├── archive pruning
├── surrogate uncertainty
└── DDI calculations

Integration tests
├── POST /api/run_pharmaga
├── GET /api/drugs
└── end-to-end optimization

Benchmark tests
├── HV
├── IGD
├── PF
├── oracle efficiency
└── runtime / scaling
```

Important invariants for MCX and mutation should include:

- no duplicate drugs
- regimen size <= `Kmax`
- all drugs belong to the library
- dose bins remain in range
- hard contraindications are never returned
- archived solutions are non-dominated

---

# Limitations

The current research package explicitly identifies several limitations:

- **Synthetic patients:** the benchmark uses synthetic profiles rather than real EHR-linked pharmacogenomic cohorts.
- **Incomplete drug-target graph:** missing off-target annotations can hide clinically important interactions.
- **MDD efficacy proxy:** receptor-binding similarity is a comparatively weak approximation of clinical antidepressant response.
- **Coarse genotype representation:** phenotype categories lose quantitative information about allele activity and clearance.
- **Static treatment representation:** the current chromosome does not represent treatment over multiple time phases.

These limitations should be visible in the README because they demonstrate that the project understands the difference between computational benchmark performance and clinical readiness.

---

# Ethics & Safety

PharmaGA is a computational research prototype. It should not be represented as a prescription engine or clinical decision-support system without the appropriate validation, governance, medical oversight, regulatory review, and safety testing.

Patient data should be minimized, de-identified where possible, encrypted in transit and at rest, and handled according to applicable privacy requirements.

---

# Research Assets

This project is accompanied by:

- the PharmaGA research paper
- a parallel-processing / architecture report
- a project presentation
- benchmark methodology and ablation results

These materials describe the formulation, algorithmic components, benchmark, parallel execution model, results, limitations, and future research directions.

---

# Citation

```bibtex
@misc{pharmaga2026,
  title        = {PharmaGA: A Niching Multi-Objective Genetic Algorithm with Surrogate-Assisted Fitness for Personalized Drug Selection},
  year         = {2026},
  note         = {Research project and source-code repository}
}
```

---

# Authors

- Sanket Kulkarani
- Rahul Hongekar
- Prem Rathod
- Pranjal Sen
- Pradyot Bilurkar

Department of Machine Learning, B.M.S. College of Engineering, Bangalore, India.

---

## Project Positioning

PharmaGA sits at the intersection of:

`Genetic Algorithms` · `Multi-Objective Optimization` · `Pharmacogenomics` · `Drug Discovery` · `Surrogate Modeling` · `Molecular Docking` · `Parallel Computing` · `Full-Stack ML Systems`

Its strongest differentiator is not simply “using a genetic algorithm for drugs.” The project combines a **domain-aware genetic operator (MCX)**, **patient-conditioned objectives**, **uncertainty-gated surrogate evaluation**, and **parallel execution** into a single end-to-end system.
