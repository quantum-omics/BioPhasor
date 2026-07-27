"""biophasor.phnn.utils — pHNN domain utilities.

Public API
----------
CascadePredictor       : Falsifiable G→P phase-cascade test
verify_passivity       : Passivity invariant check (Ḣ|_{u=0} ≤ 0)
evaluate_edge_recovery : Held-out edge AUROC above chance
evaluate_held_out_perturbation : Perturbation forecasting RMSE
print_validation_report: Formatted validation summary
"""

from biophasor.phnn.utils.cascade_predictor import CascadePredictor
from biophasor.phnn.utils.validation import (
    verify_passivity,
    evaluate_edge_recovery,
    evaluate_held_out_perturbation,
    print_validation_report,
)

__all__ = [
    "CascadePredictor",
    "verify_passivity",
    "evaluate_edge_recovery",
    "evaluate_held_out_perturbation",
    "print_validation_report",
]
