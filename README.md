# Semantic Envelope Extraction

Siehe `docs/superpowers/specs/2026-04-17-semantic-envelope-extraction-design.md` für das Design.

## Installation

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows / Git Bash
pip install -r requirements.txt
pip install "git+https://github.com/facebookresearch/sam2.git"
pip install -e .
```

## Smoke-Test

```bash
python extract.py --input data/stray_scanner/228fb53d88.zip --output outputs/228fb53d88/
```
