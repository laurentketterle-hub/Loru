# Contributing to Loru

Thank you for your interest in contributing to Loru! This guide will help you get started.

## Getting Started

### Prerequisites
- Python 3.10+
- Git

### Setup
```bash
git clone https://github.com/mergeos-bounties/Loru.git
cd Loru
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

### Running Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ -v --cov=src --cov-report=term
```

### Code Style
We follow PEP 8 with 100-char line length. Format with:
```bash
pip install ruff
ruff check . --select=E,F,W --ignore=E501
ruff format .
```

## Development Workflow

1. **Find an issue** — Browse [open issues](https://github.com/mergeos-bounties/Loru/issues) for `good first issue` or `bounty` labels
2. **Claim a bounty** — Comment `I claim this bounty` on the issue
3. **Fork & branch** — Fork the repo, create `feat/<description>` branch
4. **Implement** — Write code + tests
5. **Test locally** — Run `python -m pytest tests/ -v`
6. **Open a PR** — Use `Fixes #<issue-number>` in the description

## PR Checklist
- [ ] Tests pass: `python -m pytest tests/ -v`
- [ ] New features include tests
- [ ] Code is formatted with ruff
- [ ] PR description references the issue with `Fixes #N`
- [ ] No unrelated changes

## Bounty Claims (MergeOS)

1. Star the [Loru repo](https://github.com/mergeos-bounties/Loru) and [MergeOS](https://github.com/mergeos-bounties/mergeos)
2. Comment `I claim this bounty` on the issue
3. Comment on [MergeOS Claim #1](https://github.com/mergeos-bounties/mergeos/issues/1) with issue link
4. Open PR with `Fixes #N`
5. After merge, MRG credit is distributed

See [BOUNTY.md](docs/BOUNTY.md) for full policy.

## Project Structure
```
Loru/
├── data/           # Sign-pack data files
├── docs/           # Documentation
├── src/            # Source code
├── tests/          # Test suite
├── pyproject.toml  # Project config
└── ai_solution.py  # AI solution module
```

## Questions?
Open a [discussion](https://github.com/mergeos-bounties/Loru/discussions) or comment on your issue.

Happy contributing! 🎯
