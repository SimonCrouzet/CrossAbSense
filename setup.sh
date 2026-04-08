#!/bin/bash
# CrossAbSense setup script
# Installs dependencies, downloads the GDPa1 dataset, and precomputes ESM-C embeddings.
#
# Usage:
#   bash setup.sh           # install + download + precompute
#   bash setup.sh --test    # also run the full test battery

set -e

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
RESET='\033[0m'

ok()   { echo -e "${GREEN}✓${RESET} $*"; }
fail() { echo -e "${RED}✗${RESET} $*"; }
info() { echo -e "${BOLD}$*${RESET}"; }
warn() { echo -e "${YELLOW}⚠${RESET}  $*"; }

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
RUN_TESTS=false
for arg in "$@"; do
    case $arg in
        --test) RUN_TESTS=true ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}╔══════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║       CrossAbSense Setup             ║${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════╝${RESET}"
echo ""

# ---------------------------------------------------------------------------
# Step 1 — Dependencies
# ---------------------------------------------------------------------------
info "[1/3] Installing Python dependencies..."
if pip install -r requirements.txt -q; then
    ok "Dependencies installed"
else
    fail "pip install failed"; exit 1
fi
echo ""

# ---------------------------------------------------------------------------
# Step 2 — Dataset
# ---------------------------------------------------------------------------
info "[2/3] Downloading GDPa1 dataset..."
if python scripts/download_dataset.py; then
    ok "Dataset ready"
else
    fail "Dataset download failed"; exit 1
fi
echo ""

# ---------------------------------------------------------------------------
# Step 3 — Precompute embeddings
# ---------------------------------------------------------------------------
info "[3/3] Precomputing ESM-C embeddings..."
warn "This may take 10–30 minutes depending on GPU speed."
echo ""

run_precompute() {
    local label="$1"; shift
    echo -e "  ${BOLD}→ $label${RESET}"
    if python src/utils/precompute_embeddings.py "$@"; then
        ok "$label"
    else
        fail "$label — aborting"; exit 1
    fi
    echo ""
}

run_precompute "ESM-C 300M (VH/VL)" \
    --input inputs/GDPa1_complete.csv --encoder esmc_300m --device cuda

run_precompute "ESM-C 300M full-chain (HC/LC)" \
    --input inputs/GDPa1_complete.csv --encoder esmc_300m --full-chain --device cuda

run_precompute "ESM-C 600M (VH/VL)" \
    --input inputs/GDPa1_complete.csv --encoder esmc_600m --device cuda

run_precompute "ESM-C 600M full-chain (HC/LC)" \
    --input inputs/GDPa1_complete.csv --encoder esmc_600m --full-chain --device cuda

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo -e "${GREEN}${BOLD}Setup complete!${RESET}"
echo ""
echo "To train a model:"
echo "  python src/train.py --property HIC --config src/config/oracle_efficient_config.yaml"
echo ""

# ---------------------------------------------------------------------------
# Test battery (optional)
# ---------------------------------------------------------------------------
if [ "$RUN_TESTS" != true ]; then
    read -r -p "Run test battery? [y/N] " answer
    case "$answer" in
        [yY]*) RUN_TESTS=true ;;
    esac
fi

if [ "$RUN_TESTS" = true ]; then
    echo ""
    info "=== Test battery ==="
    echo ""

    # Generate smoke config (oracle_efficient_config with max_epochs=2)
    python scripts/make_smoke_config.py

    python tests/test_setup_checks.py
    test_exit=$?

    rm -f src/config/smoke_config.yaml

    if [ $test_exit -eq 0 ]; then
        echo ""
        ok "${BOLD}All tests passed${RESET}"
    else
        echo ""
        fail "${BOLD}Some tests failed — see output above${RESET}"
        exit 1
    fi
fi
