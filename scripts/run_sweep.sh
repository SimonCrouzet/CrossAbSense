#!/bin/bash
# Generic hyperparameter sweep runner for all tuning phases
#
# Usage:
#   ./scripts/run_sweep.sh --phase 4.2 <property>
#   ./scripts/run_sweep.sh --phase 4.2 --all
#
# Examples:
#   ./scripts/run_sweep.sh --phase 4.2 HIC
#   ./scripts/run_sweep.sh --phase 4.1 Titer
#   ./scripts/run_sweep.sh --phase 4.2 --all

set -e

# Parse arguments - forward all to Python script
exec python3 scripts/run_sweep.py "$@"
