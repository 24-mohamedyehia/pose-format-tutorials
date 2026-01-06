# Contributing to pose-format-tutorials

Thank you for your interest in contributing. This document explains how to set up a development environment, the repository conventions we follow, and what we expect from contributions. All content in this repository must be written in English.

## Table of contents
- Purpose
- Getting started
- Notebook style guidelines
- Code & docs workflow
- Quality checks & pre-commit
- Reporting issues
- Maintainers & contact

## Purpose
This repository contains tutorial notebooks and small helper scripts that demonstrate usage patterns for the `pose-format` library. Contributions should improve clarity, correctness, or add practical examples.

## Getting started
1. Clone the repository and create a Python virtual environment:

```bash
git clone https://github.com/your-org/pose-format-tutorials.git
cd pose-format-tutorials
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\activate     # Windows PowerShell
pip install -r requirements.txt
```

2. Optional: install `pre-commit` for automatic formatting:

```bash
pip install pre-commit
pre-commit install
```

## Notebook style guidelines
- Language: English only in text and code comments.
- Keep output cleared for PRs: use `nbstripout` or run `pre-commit` to remove large outputs.
- Cell structure: small, focused cells. Add short Markdown headings and explanatory text.
- Metadata: keep kernel and language metadata present; avoid committing environment-specific paths.
- File names: keep the existing numbering prefix (e.g., `01_...ipynb`) to preserve order.

## Code & docs workflow
- Branching: create descriptive branches: `feat/<short-description>`, `fix/<short-description>`, or `docs/<short-desc>`.
- Commits: write clear, atomic commit messages. Use present-tense imperative (e.g., "Add video extraction example").
- Pull requests: include a short description, list of changes, and mention any notebooks that require manual review.

### PR checklist
- [ ] The change is small, focused, and tested locally.
- [ ] Notebooks have outputs stripped for the diff or include lightweight outputs only.
- [ ] New dependencies are added to `requirements.txt` with a justification in the PR description.
- [ ] README and table of notebooks updated if notebooks are added or renamed.

## Quality checks & pre-commit
We recommend these checks in `.pre-commit-config.yaml` (examples):
- `black` for Python formatting
- `isort` for imports
- `nbstripout` to strip notebook outputs on commit
- `flake8` for linting (optional)

If CI is present, it should run:
- `pip install -r requirements.txt`
- `black --check` and `flake8` (if configured)
- Notebook execution or fast smoke tests for example notebooks (optional)

## Reporting issues
- Use GitHub Issues for bug reports or feature requests.
- A good issue should include: steps to reproduce, expected result, actual result, and sample files if applicable.

## Maintainers & contact
If you need help or want to propose substantial changes, open an issue or tag maintainers in a PR. Maintainers will triage contributions and review PRs.

---
Thank you for helping improve these tutorials! Contributions make these resources more useful for the community.
