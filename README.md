# Cardiac Digital Twin: Knowledge Graph + GNN for Ventricular Remodeling

> **Project Status: Early design / prototype phase**  
> This project is actively being developed. Improvements, additional features, and clinical validations will be added incrementally.

## Overview

This repository implements a **Digital Twin of the left ventricle (LV)** using a hybrid approach that combines:

- **Domain knowledge** (cardiac anatomy and physiology) represented as a **heterogeneous knowledge graph**.
- **Graph Neural Networks (GNNs)** to learn patient-specific remodeling responses.
- **Neuro‑symbolic loss functions** that embed the law of Laplace, mass conservation, and clinical classification rules (HCM vs. DCM).

The model is trained exclusively on **healthy subjects (NOR)** and then used to simulate the progression of:

- **Hypertrophic Cardiomyopathy (HCM)**: triggered by increased afterload (hypertension).
- **Dilated Cardiomyopathy (DCM)**: triggered by volume overload (preload increase).

A temporal simulator integrates the GNN‑predicted rates of change (`dr/dt`, `dm/dt`) over time (months to years), capturing geometric and metabolic remodeling, including a **metabolic burnout mechanism**.

---

## Key Features

- **Knowledge‑driven graph** with node types: `chamber`, `state` (ED/ES), `tissue`, `vessel`, `system`, and edges for anatomical flow, boundary relations, and temporal transitions.
- **HeteroGNN** built with `torch_geometric` that processes the graph and outputs `Δh` (wall thickness change), `Δm` (relative mass change), and `Δr` (cavity radius change).
- **Neuro‑symbolic loss** (`NeuroSymbolicLoss`) with terms for:
  - Physics (Laplace’s law)
  - Mutual exclusion (HCM/DCM)
  - Data fidelity (target geometry)
  - Mass conservation
  - Sign constraints (direction of remodeling)
  - L2 regularization
- **Temporal simulation** using Euler integration, with adaptive activation based on wall stress.
- **Phenotype classification** (HCM / DCM / NOR) based on post‑simulation `h/r` ratio.
- **Fully reproducible pipeline**: from ACDC dataset preprocessing, graph construction, training, evaluation, to time‑series visualisation.

---

## Dataset

- **Source**: ACDC (Automated Cardiac Diagnosis Challenge) dataset, preprocessed into `.h5` files containing labelled segmentation masks (LV = 3, myocardium = 2).
- **Preprocessing**:  
  - Pixel counts per slice → 2D areas → radius (`r`) and thickness (`h`) assuming circular cross‑section.
  - End‑diastolic (ED) and end‑systolic (ES) frames identified by LV volume.
- **Patient graphs**: one graph per patient containing all derived metrics (volumes, radii, thickness, EF, h/r ratio).
- **Labels**: symbolic classification (NOR / HCM / DCM / LETHAL_STATE) based on hard thresholds.

---

## Project Structure (Notebook Cells)

| Cell | Content |
|------|---------|
| 1    | Data loading, feature aggregation, graph creation (NetworkX) |
| 2    | Visualisation of a sample patient graph |
| 3    | Pickle serialisation of NetworkX graphs |
| 4    | Loading saved graphs |
| 5    | Hard‑constraint application (phenotype labelling) |
| 6    | Installation of `torch_geometric` |
| 7    | Conversion of NetworkX graphs to `HeteroData` (PyG) |
| 8    | Definition of `DigitalTwinGNN` (heterogeneous GNN) |
| 9    | Initialisation helper (`init_weights`) |
| 10   | `NeuroSymbolicLoss` implementation |
| 11   | Model training (only on NOR patients, using HCM trigger) |
| 12   | Train/test split and re‑training |
| 13   | Evaluation: HCM simulation on test set |
| 14   | Evaluation: DCM simulation on test set |
| 15   | Violin plot of h/r ratios |
| 16   | Temporal simulation for DCM (one patient) |
| 17   | Temporal simulation for HCM (one patient) |
| 18–24| Systematic screening for severe HCM, best‑case plotting |

---

## Current Limitations & Planned Improvements

This is an **ongoing project**. The following limitations have been identified and will be addressed in future iterations:

| Limitation | Planned Improvement |
|------------|----------------------|
| **Arbitrary voxel volume** (volumes in pseudo‑units) | Extract real `pixel_spacing` and `slice_thickness` from `.h5` metadata → true volumes (mL). |
| **Limited training data** (63 healthy patients) | Implement graph‑based data augmentation (scaling of r/h, gaussian noise, etc.). |
| **Only two cardiac states (ED, ES)** | Add intermediate states (e.g., LV_MID) and finer temporal edges. |
| **Concentric hypertrophy only** (no length change) | Introduce `length` feature and `Δlength` output, with appropriate loss terms. |
| **Training only on HCM triggers** | Alternate HCM and DCM triggers during training to learn both remodeling directions. |
| **No validation on real HCM/DCM patients** | Test the model directly on patients with known HCM/DCM labels from the dataset. |
| **Missing dynamic ODE for cycle** | Replace simple `state→state` edge with a continuous‑time representation (optional). |
| **2D circular geometry assumption** | Future: 3D ellipsoidal or mesh‑based shape representation. |

Contributions and suggestions are welcome.

---

## Getting Started

### Prerequisites

- Python 3.12+
- PyTorch
- torch_geometric
- NetworkX
- h5py
- numpy, matplotlib, pandas, seaborn

Install the required packages:

```bash
pip install torch torch_geometric networkx h5py numpy matplotlib pandas seaborn