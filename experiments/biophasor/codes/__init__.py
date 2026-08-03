"""
biophasor.experiments.codes
============================
Publication-grade experiment scripts for the BioPhasor manuscript.

Each script runs the *unmodified* BioPhasor package against a real public
dataset and regenerates the corresponding manuscript figure(s) + a results
JSON. Data are read from ``experiments/data/raw`` (downloaded from NCBI GEO
on first run).

Scripts
-------
exp01_cellcycle_assignment  -- Cell-cycle phase assignment (GSE293316 scRNA-seq)
exp02_circadian_rhythm      -- Circadian rhythmicity (GSE171432 mouse liver)

Planned (next iteration, see feasibility-and-plan/plan-II.md): encoding
comparison, Kuramoto GRN synchrony, multi-omics fusion, phasor classification,
Cell State Tensor dynamics, attractor landscape.
"""
