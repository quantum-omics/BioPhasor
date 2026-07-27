#!/usr/bin/env bash
# ==========================================================================
# BioPhasor platform — unified, per-manuscript reproduction.
#
#   bash reproduce.sh              # run every manuscript's experiments
#   bash reproduce.sh --check      # verify env + shared data cache only
#   bash reproduce.sh biophasor    # run one manuscript's experiments
#   bash reproduce.sh phnn spectral# run a subset
#
# Reusable science is imported from the installed `biophasor` package; every
# manuscript's scripts read shared public datasets ONCE from
# experiments/_shared/data/raw/<accession> via experiments/_shared/common.py.
# ==========================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-$HERE/.venv/bin/python}"
EXP="$HERE/experiments"
CACHE="$EXP/_shared/data/raw"

echo "== BioPhasor platform reproduce =="
"$PY" -c "import biophasor,sys; print('biophasor', biophasor.__version__, '| py', sys.version.split()[0])"

# --- shared data cache presence check ---
MISS=0
for f in GSE293316_reh.h5 GSE171432_fpkm.tsv.gz cptac_ucec/rna.pkl.gz cptac_ucec/protein.pkl.gz; do
  [[ -e "$CACHE/$f" ]] || { echo "MISSING SHARED DATA: $CACHE/$f"; MISS=1; }
done
if [[ $MISS -ne 0 ]]; then
  echo "  scRNA-seq GSE293316 | circadian GSE171432 | multi-omics CPTAC UCEC (cptac pkg)"
  echo "  GEO series (GSE10072/GSE11923) are fetched on demand by _shared/common.py"
  exit 2
fi
echo "shared data cache: OK ($CACHE)"
[[ "${1:-}" == "--check" ]] && { echo "check OK"; exit 0; }

# --- per-manuscript experiment runners ---
run_dir () {  # run_dir <manuscript> <script1> <script2> ...
  local m="$1"; shift
  local d="$EXP/$m/codes"
  echo ""; echo "########## manuscript: $m ##########"
  for s in "$@"; do
    echo ""; echo "---- $m/$s ----"
    "$PY" "$d/$s"
  done
}

MANUSCRIPTS=("$@")
[[ ${#MANUSCRIPTS[@]} -eq 0 ]] && MANUSCRIPTS=(biophasor phnn cell-atlas ehr spectral)

for m in "${MANUSCRIPTS[@]}"; do
  case "$m" in
    biophasor)
      run_dir biophasor \
        exp03_encoding_coherence.py exp01_cellcycle_assignment.py \
        exp03_kuramoto_synchrony.py exp02_circadian_rhythm.py \
        exp03_multiomics_fusion.py exp04_ml_classification.py \
        exp04_manifold_geometry.py exp06_cst_temporal.py \
        exp07_cst_tensornetwork.py exp08_cst_pathway.py \
        exp09_cst_omics_pac.py exp10_cst_quantum_bridge.py \
        exp11_vpc_vqc_complexity.py exp04_attractor_floquet.py \
        exp05_cst_knockout.py \
        exp09b_omics_pac_hardened_stats.py exp12_multiomics_benchmark.py \
        exp12b_benchmark_competitors_fix.py exp13_hardtask_clinical.py ;;
    phnn)        run_dir phnn        fig5_data_overview.py fig9_cascade_mechanism.py ;;
    cell-atlas)  run_dir cell-atlas  run_atlas.py ;;
    ehr)         run_dir ehr         run_ehr.py ;;
    spectral)    run_dir spectral    exp_circadian.py exp_cancer.py ;;
    generative-omics)
      echo ""; echo "########## manuscript: generative-omics ##########"
      echo "  (training-driven; see experiments/generative-omics/README + hf_bundle)" ;;
    *) echo "unknown manuscript: $m" ;;
  esac
done
echo ""; echo "== reproduce complete; outputs in experiments/<manuscript>/{results,figures} =="
