#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m streamlit run app/perturbation_explorer.py
