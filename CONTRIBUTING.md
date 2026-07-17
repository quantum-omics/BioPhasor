Contributing to BioPhasor
========================

Thank you for your interest in contributing to BioPhasor! This document provides
guidelines to help you get started.

---

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/your-username/biophasor.git
   cd biophasor/biophasor
   ```
3. **Create a virtual environment** and install in editable mode:
   ```bash
   conda create -n biophasor-dev python=3.11
   conda activate biophasor-dev
   pip install -e ".[dev]"
   ```
4. **Create a branch** for your feature or bugfix:
   ```bash
   git checkout -b feat/my-new-feature
   ```

---

## Development Workflow

### Running Tests

```bash
pytest tests/ -v
# With coverage
pytest tests/ --cov=biophasor --cov-report=html
```

All 62 tests must pass before submitting a PR. New code should include tests.

### Code Style

We use **black** for formatting and **ruff** for linting:

```bash
black biophasor/ tests/         # auto-format
ruff check biophasor/ tests/    # lint
mypy biophasor/                 # type checking (optional)
```

Line length: **100 characters**.

### Adding a New Feature

1. Add your code to the appropriate sub-package (`core/`, `dynamics/`, etc.)
2. Write unit tests in `tests/test_<subpackage>.py`
3. Update the relevant documentation chapter in `docs/`
4. Add an entry to `CHANGELOG.md` under `[Unreleased]`
5. Update `biophasor/__init__.py` if you are exporting new public symbols

---

## Pull Request Guidelines

- Keep PRs **focused** — one feature or bugfix per PR
- Write a **clear PR title** following conventional commits:
  - `feat: add PhasorVAE encoder`
  - `fix: circular mean at ±π boundary`
  - `docs: add metabolomics chapter`
  - `test: add PLV matrix tests`
- Fill in the **PR description** with motivation, approach, and how to test
- Ensure all CI checks pass

---

## Reporting Issues

Please use GitHub Issues with:
- A **minimal reproducible example**
- BioPhasor version (`import biophasor; print(biophasor.__version__)`)
- Python version and OS
- Full traceback

---

## Code of Conduct

We follow the [Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).
Please be respectful and constructive in all interactions.

---

## License

By contributing, you agree that your contributions will be licensed under the
same terms as the project (CC BY-NC 4.0).
