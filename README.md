
# P04: Phase-Aware HW/SW Co-Optimisation for Concurrent Multi-Model ML Workloads

## Project Overview

This project investigates resource-aware scheduling of concurrent machine-learning workloads on heterogeneous chiplet accelerators.

Different neural-network operations can have different compute, local-memory, and DRAM bandwidth requirements. The project evaluates whether explicit runtime resource-phase information provides useful scheduling information beyond static and reactive scheduling policies.

The framework uses analytical simulation and Timeloop-derived ResNet-50 layer profiles. The results are simulator outputs, not measurements from physical chiplet hardware.

## Architecture

The model contains four heterogeneous chiplets connected through a 2 x 2 Network-on-Package (NoP) mesh.

| Chiplet | Type | Compute | Memory Capacity | Memory Bandwidth |
|---|---|---:|---:|---:|
| C1 | Compute-oriented | 1.50x | 1.00x | 0.75x |
| C2 | Compute-oriented | 1.50x | 1.00x | 0.75x |
| C3 | Memory-oriented | 0.75x | 1.00x | 1.50x |
| C4 | Balanced | 1.00x | 1.25x | 1.00x |

ResNet-50 profiles cover 53 layers. The estimated execution time of a group is:

`T_group = max(T_compute, T_local, T_DRAM)`

Resource phases are classified using normalised resource demand, with a 10% tie threshold.

## Scheduling Policies

- **Static:** fixed chiplet placement.
- **Static Round-Robin:** cyclic distribution across eligible chiplets.
- **Reactive:** placement based on estimated execution duration and resource availability.
- **Causal Phase-Aware:** resource-phase affinity followed by duration and simplified remapping cost.
- **Oracle Phase-Aware:** bounded-lookahead phase-aware comparison.
- **Locality-Aware:** locality used as a tie-breaker within a tolerance window.

## Repository Structure

```text
P04/
├── README.md
├── requirements.txt
├── .gitignore
├── extract_sram_reference.py
├── configs/
├── experiments/
├── figures/
├── models/
├── results/
├── scripts/
├── simulator/
├── tests/
└── timeloop/
```

The directories retain experiment configurations, implementation modules, tests, saved predictor models, Timeloop files, figures, and recorded CSV/JSON results.

## Software Environment

The project was developed using Python 3.10.12.

| Dependency | Version |
|---|---|
| NumPy | 2.1.0 |
| Pandas | 2.2.3 |
| Matplotlib | 3.10.1 |
| PyYAML | 5.3 |
| Scikit-learn | 1.5.2 |
| Joblib | 1.4.2 |
| PyTorch | 2.4.0 |
| Torchvision | 0.19.0 |
| Pytest | 9.2.0 |

Install Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

A configured Timeloop/Accelergy environment is also required. The Python requirements file does not install the complete Timeloop/Accelergy toolchain.

## Experiments

Run experiment commands from the repository root.

| Experiment | Script |
|---|---|
| Scheduler comparison | `experiments/run_causal_scheduler_comparison.py` |
| E2: Calibration ablation | `experiments/run_calibration_ablation_e2.py` |
| E3: Causal phase-aware | `experiments/run_causal_phase_aware_e3.py` |
| E4: Locality-aware | `experiments/run_locality_aware_e4.py` |
| E5: Mixed CNN-GEMM | `experiments/run_mixed_cnn_gemm_e5.py` |
| E6: Phase-varying mixed workload | `experiments/run_phase_varying_mixed_e6.py` |
| E7: Controlled phase experiment | `experiments/run_controlled_phase_e7.py` |

For example:

```bash
python3 experiments/run_controlled_phase_e7.py
```

**Important:** E2 and E3 may overwrite files in existing output directories. E4-E7 refuse to run when their respective output directories already exist. Preserve existing results before rerunning experiments.

The recorded experiment outputs are included in `results/`.

## Results Summary

### E1: Compact-Group Validation

| Policy | Maximum Makespan (cycles) |
|---|---:|
| Static | 200,704 |
| Static Round-Robin | 65,856 |
| Reactive | 65,856 |
| Causal Phase-Aware | 65,856 |

Round-robin reduces maximum makespan by approximately 67.2% relative to static placement.

### E2: Calibration Sensitivity

| Profile Source | Scale Source | Makespan |
|---|---|---:|
| Offline Timeloop | Offline v2 | 3,653,248 |
| Offline Timeloop | Causal v2 | 16,314,971.376 |
| Metadata-causal | Offline v2 | 2,197,312 |
| Metadata-causal | Causal v2 | 4,566,016 |

Calibration choices materially affect the estimated makespan.

### E3-E4: Scheduler Ablations

Causal phase-aware scheduling provides approximately 0.0094% improvement over reactive scheduling in one evaluated condition and ties it in the other three conditions.

Locality-aware scheduling matches reactive makespan in all four evaluated conditions.

### E5-E6: Mixed Workloads

| Experiment | Policy | Makespan (cycles) |
|---|---|---:|
| E5 | Static | 4,281,901.333 |
| E5 | Static Round-Robin | 1,405,108.375 |
| E5 | Reactive | 1,405,108.375 |
| E6 | Static | 13,909,793.889 |
| E6 | Static Round-Robin | 4,616,296.625 |
| E6 | Reactive | 3,973,234.750 |
| E6 | Causal Phase-Aware | 3,973,234.750 |

Reactive scheduling improves over round-robin by approximately 13.93% in E6. Causal phase-aware scheduling ties reactive scheduling.

### E7: Controlled Synthetic Phase Experiment

| Policy | Makespan (cycles) |
|---|---:|
| Static Round-Robin | 2,666.667 |
| Reactive | 2,300 |
| Causal Phase-Aware | 2,133.333 |

Causal phase-aware scheduling improves makespan by 7.25% over reactive scheduling in this controlled synthetic experiment.

The results suggest that explicit phase information can be useful when resource characteristics are clearly differentiated, but its benefit is not consistent across all evaluated workloads.

## Tests

Run the test suite:

```bash
python3 -m pytest tests/ -v
```

Previously recorded validation included 182 passing tests and 39 passing subtests. Re-run the suite in the target environment to verify the current checkout.

## Figures

The `figures/` directory contains PDF and PNG versions of the system architecture, resource characteristics, methodology, policy comparisons, calibration sensitivity, mixed-workload results, scheduler ablations, and chiplet placement plots.

## Modelling Assumptions and Limitations

This project uses a compact analytical simulation model rather than physical chiplet hardware.

1. Physical NoP contention is not explicitly included in the compact scheduler objective.
2. Complete SRAM-capacity feasibility is not enforced by the compact scheduler.
3. Remapping cost is represented by a simplified 5% reference-cycle penalty.
4. The compact scheduler does not receive the full dependency graph of GEMM groups.
5. The oracle phase-aware scheduler uses bounded lookahead and is not globally optimal.
6. Calibration and resource-estimation choices can materially affect makespan.

The reported results should be interpreted as analytical simulator outcomes, not physical hardware speedups.

## Research Scope

This project evaluates whether explicit resource-phase information adds useful scheduling information beyond static and reactive policies.

It does not claim to outperform SCAR, CLEAR, or PhaseWeave, since those systems have not been faithfully reproduced in this implementation.
