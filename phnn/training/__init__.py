"""biophasor.phnn.training — pHNN training pipeline.

Physics losses live in biophasor.core.losses (shared platform-wide);
this subpackage holds the training loop and CLI entry point.

Public API
----------
train_omics_surrogate : full biologically-grounded GNN-pHNN training pipeline
"""

from biophasor.phnn.training.train_surrogate import train_omics_surrogate

__all__ = ["train_omics_surrogate"]
