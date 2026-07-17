# BioPhasor `data/` — cached source datasets

Persistent store of the public source datasets, so the project can be restarted
without re-downloading. Everything here was retrieved from public archives.
Computed outputs do **not** live here — result files are in `../results/`,
figures in `../figures/`.

## Layout

```
data/
└── raw/            # public source datasets, exactly as downloaded (do not edit)
    ├── GSE293316_reh.h5          # cell-cycle (scenarios 1,3,4,6,8)
    ├── GSE171432_fpkm.tsv.gz     # circadian (scenarios 2,6)
    └── cptac_ucec/               # matched RNA+protein cache (scenarios 5,7,9)
```

## `raw/` — source datasets

| File | Source | Description | sha256 (prefix) |
|---|---|---|---|
| `GSE293316_reh.h5` | NCBI GEO GSE293316 | REH human B-ALL scRNA-seq, 10x filtered matrix (7,433 cells × 36,601 genes). | `d583410e…` |
| `GSE171432_fpkm.tsv.gz` | NCBI GEO GSE171432 | WT + Bmal1-KO mouse-liver circadian RNA-seq, FPKM table (36 cols = 18 WT + 18 KO; only the 18 `WT_` columns are used). | `57438e05…` |
| `cptac_ucec/` | `cptac` package (CPTAC UCEC) | Matched RNA+protein cache, 109 samples × 9,200 shared genes; see [`raw/cptac_ucec/README.md`](raw/cptac_ucec/README.md). | (regeneratable) |

The two GEO files were retrieved from the NCBI GEO FTP mirror. To re-fetch from
scratch, run the loader with an empty output dir (it skips download when the
files already exist here):
`python ../feasibility-and-plan/biophasor_realdata_loader.py --outdir <dir>`

The `GSE293316_reh.h5` and `cptac_ucec/` cache are gitignored (large /
regeneratable); the small circadian FPKM table is tracked.

## Where the computed outputs are

| Output | Location |
|---|---|
| Result files (per scenario) | `../results/*_results.json` |
| Publication figures | `../figures/*.png` |
| Honest verdicts + scorecard | `../reports/plan2_verdicts.md` |
| Scenario → script → result → figure map | `../README.md` |

Current headline verdicts (post-fix): cell-cycle **reproduces** (acc 0.69, G1
recall 0.98), circadian **partial** (peak-ZT MAE 1.4h; recall 0.43
sampling-limited). Full 9-scenario scorecard in `../reports/FINDINGS.md`.

## Regeneration

- **Data:** GEO accessions are stable; re-fetched by the loader above. The CPTAC
  cache rebuilds from the `cptac` package (see its README).
- **Results/figures:** re-run the scenario scripts under `../codes/` (seeded →
  byte-identical result files). To restyle a figure without recomputing, edit
  only the `_plot` function in the relevant script.
- **Verify integrity:** `shasum -a 256 raw/GSE293316_reh.h5 raw/GSE171432_fpkm.tsv.gz`
  against the prefixes above.

See [`../feasibility-and-plan/plan-II.md`](../feasibility-and-plan/plan-II.md)
for how these datasets scale up in the next (GPU/cloud) iteration.
