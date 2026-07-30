# Contributing to Loru

Thank you for your interest in contributing to Loru! This guide will help you get started.

## Getting Started

1. **Fork the repository** and clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Loru.git
   cd Loru
   ```

2. **Set up the environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Create a branch** for your changes:
   ```bash
   git checkout -b feat/your-feature-name
   ```

## Development Workflow

- Keep changes focused and minimal
- Write clear commit messages
- Add tests for new features
- Run tests before submitting: `python -m pytest`

## Good First Issues

Look for issues labeled `good first issue` — these are great for new contributors:

- [Add a sign pack](https://github.com/mergeos-bounties/Loru/labels/good%20first%20issue)
- [Fix documentation](https://github.com/mergeos-bounties/Loru/labels/documentation)

## Submitting a Pull Request

1. Push your branch to your fork
2. Open a PR against the `main` branch
3. Describe what your PR does and link to the issue
4. Wait for review

## Community

- Be respectful and inclusive
- Follow the [Code of Conduct](CODE_OF_CONDUCT.md)
