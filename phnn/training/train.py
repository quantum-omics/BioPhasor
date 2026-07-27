"""
train.py  —  CLI entry point

Command-line interface for the biologically grounded GNN-pHNN training pipeline.
Delegates to train_surrogate.train_omics_surrogate() for all logic.

Usage:
    python train.py
    python train.py --epochs 500 --batch_size 48 --seed 0

Note:
    Use train_surrogate.py directly for the full pipeline including the
    biological graph, two-layer state, and validation.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))


def main():
    parser = argparse.ArgumentParser(
        description="GNN-Surrogate Port-Hamiltonian Neural Network Training"
    )
    parser.add_argument("--epochs",     type=int,   default=500,
                        help="Number of training epochs (default: 500)")
    parser.add_argument("--batch_size", type=int,   default=48,
                        help="Training batch size (default: 48)")
    parser.add_argument("--seed",       type=int,   default=0,
                        help="Random seed for reproducibility (default: 0)")
    args = parser.parse_args()

    from biophasor.phnn.training.train_surrogate import train_omics_surrogate
    train_omics_surrogate(
        epochs     = args.epochs,
        batch_size = args.batch_size,
        seed       = args.seed,
    )


if __name__ == "__main__":
    main()
