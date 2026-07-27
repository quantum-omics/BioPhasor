# Licensing

Recorded so the reasoning does not have to be reconstructed later.

## Current state

| Project | Licence | Copyright holder | Where |
|---|---|---|---|
| **BioPhasor** (this repo) | Apache-2.0 | Quantum Omics Foundation (nonprofit) | `LICENSE`, `NOTICE` |
| **phasorflow** (optional dependency) | MIT | Mindverse Computing LLC | separate repo; PyPI |

BioPhasor does not vendor phasorflow. It is an optional extra
(`pip install -e ".[vpc]"`); the VPC-backed code paths fall back to logistic
regression when it is absent, and experiments record `phasorflow_available` in
their results.

## Why Apache-2.0 for BioPhasor

Previously CC BY-NC 4.0 with a patent-and-trademark reservation clause. That was
changed because CC BY-NC is not an open-source licence (Creative Commons advise
against using it for software), "non-commercial" has no settled legal meaning,
and the publishability review flagged it as conflicting with journal
open-source and data-availability policies.

Apache-2.0 was chosen over MIT for three reasons specific to this project:

1. **Patents (§ 3).** The Foundation holds no patents today and may file later.
   Apache-2.0's grant covers only patents that read on *released code*, so
   future inventions not embodied in what is published stay unaffected. What it
   forecloses is publishing code and later asserting a patent against its users
   — a right a nonprofit with an open-research mission should not be quietly
   reserving. MIT is silent on patents, which leaves adopters with copyright
   permission and no patent assurance; that silence is conspicuous here because
   the work's lineage traces to an LLC whose previous licence opened with a
   patent reservation.
2. **Trademarks (§ 6).** Explicitly withheld, so the BioPhasor name and the
   Foundation's marks stay protected while the code stays open — the same
   concern the old licence expressed, handled by a standard clause.
3. **Contributions (§ 5).** Inbound-equals-outbound, so a small nonprofit needs
   only a DCO sign-off rather than CLA machinery.

**Known cost:** Apache-2.0 cannot be combined into GPLv2-*only* projects
(GPLv3 is fine). Accepted deliberately.

## Why MIT stays right for phasorflow

Maximum permissiveness for a general-purpose library heading to PyPI, minimal
friction for adopters, and it remains commercially owned by Mindverse Computing
LLC while free for everyone to use. MIT is one-way compatible into Apache-2.0,
so an Apache-2.0 BioPhasor depending on MIT phasorflow is unproblematic.

## Dependency licence compatibility

All runtime dependencies are BSD-3-family or equivalent (numpy, scipy, torch,
scikit-learn, anndata, scanpy, pandas, networkx, h5py, matplotlib, seaborn).
No copyleft anywhere in the runtime path.

## Open items — not resolved by changing the licence text

These are legal questions, not code questions. Recorded so they are not lost.

1. **Ownership.** The copyright line on every source file now reads
   `Quantum Omics Foundation`. That is a factual assertion. If the code was
   authored by Mindverse Computing LLC personnel, a written assignment or
   licence from the LLC to the Foundation is what makes it true. Paper this
   before the repository is public.
2. **Patent non-assertion from the LLC.** A licence binds only the licensor. If
   Mindverse Computing LLC holds or later files patents covering methods
   BioPhasor implements, the Foundation's Apache-2.0 grant does not cover them —
   users could have full copyright permission and still be exposed. A short
   non-assertion covenant from the LLC covering BioPhasor and its users closes
   this, and costs the LLC nothing given phasorflow is already free for everyone.
3. **Publication forfeits patent rights.** Public disclosure — code, preprint,
   or manuscript — is prior art against a later application by the same party.
   The US allows a 12-month grace period for the inventor's own disclosure;
   Europe, China and most other jurisdictions apply absolute novelty and the
   application is barred outright. `manuscript/main.tex` states the code is
   publicly available at a GitHub URL, and commits have been pushed to that
   remote. **If the Foundation may want to patent any method embodied here, file
   a provisional before the repository is public** — or accept deliberately that
   it goes into the commons.
4. **Manuscript metadata still says Mindverse Computing LLC.** Author
   affiliations and the data-availability URL in `manuscript/main.tex` were left
   unchanged; they are statements about the paper, not the software licence, and
   are the authors' call.

None of the above is legal advice. Items 1-3 have real consequences and are
cheap to fix now; a short review by counsel before the repository goes public is
worth the cost.
