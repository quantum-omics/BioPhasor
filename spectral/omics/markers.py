"""
omics.markers — Curated marker genes for the five CCM compartments (theory.md §5.1).

The five compartments are Clock, Redox, Energy, Signalling, and Biosynthesis.
Symbols are given in both
mouse (title-case) and human (upper-case) forms; matching is case-insensitive so
either organism resolves. This is a compact, hand-curated seed list — not an
exhaustive pathway database — sufficient to anchor the CCM compartment axes; the
remaining features are assigned by their dominant omics harmonic.

SPDX-License-Identifier: Apache-2.0
Copyright 2024-2026 Quantum Omics Foundation
"""

from __future__ import annotations

from typing import Dict, List, Sequence

# Marker symbols per compartment (canonical, organism-agnostic on case).
COMPARTMENT_MARKERS: Dict[str, List[str]] = {
    # Core circadian clock (TTFL loop)
    "Clock": [
        "Arntl", "Bmal1", "Clock", "Npas2", "Per1", "Per2", "Per3",
        "Cry1", "Cry2", "Nr1d1", "Nr1d2", "Rora", "Rorb", "Rorc",
        "Dbp", "Nfil3", "Tef", "Hlf", "Ciart", "Bhlhe40", "Bhlhe41",
    ],
    # Redox / transcription-independent rhythms
    "Redox": [
        "Prdx1", "Prdx2", "Prdx3", "Prdx4", "Prdx5", "Prdx6",
        "Gpx1", "Gpx4", "Cat", "Sod1", "Sod2", "Txn1", "Txn2",
        "Nqo1", "Gclc", "Gclm", "Nfe2l2", "Hmox1", "Gsr",
    ],
    # Energy / adenylate & nucleotide metabolism
    "Energy": [
        "Atp5a1", "Atp5b", "Ndufa1", "Sdha", "Cox4i1", "Cox5a",
        "Pdha1", "Ldha", "Ldhb", "Pfkl", "Pfkm", "Gapdh",
        "Prkaa1", "Prkaa2", "Ppargc1a", "Slc2a1", "Slc2a4", "Hk2", "Cs",
    ],
    # Signalling (kinases, receptors, immediate-early)
    "Signalling": [
        "Mapk1", "Mapk3", "Akt1", "Mtor", "Gsk3b", "Pik3ca",
        "Egfr", "Insr", "Igf1r", "Jak2", "Stat3", "Creb1",
        "Fos", "Jun", "Egr1", "Nr4a1", "Dusp1", "Rps6kb1",
    ],
    # Biosynthesis (amino acid, lipid, protein synthesis)
    "Biosynthesis": [
        "Srebf1", "Srebf2", "Fasn", "Acaca", "Scd1", "Hmgcr",
        "Elovl6", "Mvd", "Idi1", "Sqle", "Eif4e", "Eif4ebp1",
        "Rpl3", "Rps6", "Mthfd1", "Shmt1", "Phgdh", "Asns", "Gpt",
    ],
}


def marker_to_compartment() -> Dict[str, str]:
    """Return {UPPER_symbol: compartment} for fast lookup."""
    m: Dict[str, str] = {}
    for comp, syms in COMPARTMENT_MARKERS.items():
        for s in syms:
            m[s.strip().upper()] = comp
    return m


def build_membership(
    feature_symbols: Sequence[str],
    compartments: Sequence[str] = tuple(COMPARTMENT_MARKERS.keys()),
) -> Dict[str, List[int]]:
    """Map a list of feature gene symbols to compartment index lists.

    Parameters
    ----------
    feature_symbols : sequence of str, length N
        Gene symbol for each feature (row order of the omics matrix). Empty /
        None entries are skipped.
    compartments : sequence of str
        Compartment label order (defaults to the five canonical compartments).

    Returns
    -------
    dict {compartment: [feature indices]} — features whose symbol matched a
    marker. Unmatched features are omitted (assigned later by harmonic loading).
    """
    lookup = marker_to_compartment()
    membership: Dict[str, List[int]] = {c: [] for c in compartments}
    for i, sym in enumerate(feature_symbols):
        if not sym:
            continue
        comp = lookup.get(str(sym).strip().upper())
        if comp is not None and comp in membership:
            membership[comp].append(i)
    return membership
