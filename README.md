markdown
# Two Voids Model — Simulation

## About

This repository contains the simulation code for the **Two Voids Model**: a growing causal graph that reproduces 4-dimensional spacetime and the mass spectrum of elementary particles from a single ontological principle.

The model is based on two fundamental principles: rupture (active) and contraction (passive). Their interaction generates a directed acyclic graph whose vertices are elementary acts, and whose edges are causal connections.

## Key Results

- **4-dimensional spacetime** emerges at ~15,000 events (topological phase transition).
- **Mass spectrum** of 14 Standard Model particles reproduced with <15% deviation.
- **Fundamental constant α = 1.0663** derived from the geometry of the graph.

## Quick Start (Google Colab)

Click the button below to open and run the simulation directly in your browser — no installation required.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/karat225/two-voids-simulation/blob/main/your-notebook.ipynb)

## Requirements (for local run)

- Python 3.8+
- NumPy
pip install numpy

text

## Files

- `your-notebook.ipynb` — Main Colab notebook with the full simulation, analysis, and visualization.
- `results/` — Output data from the 5M event run.

## How to Run Locally

1. Clone the repository:
git clone https://github.com/your-username/two-voids-simulation.git
cd two-voids-simulation

text

2. Install dependencies:
pip install numpy

text

3. Run the notebook:
jupyter notebook your-notebook.ipynb

text

## Parameters

| Parameter | Value | Description |
|---|---|---|
| N | 5,000,000 | Number of events |
| R | 2.0 | Rupture/contraction ratio |
| σ_phase | 1.0 | Phase noise |
| σ_dir | 0.25 | Direction noise |
| sigma_base | 5×10⁻⁵ | Baseline phase noise |

## Citation

If you use this code in your research, please cite:

- [The Two Voids Model: From Fundamental Ontology to the Mass Spectrum of Elementary Particles](https://doi.org/10.5281/zenodo.21373978)
- [The Two Voids Model (original work)](https://doi.org/10.5281/zenodo.20627691)

## License

This project is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) License.
