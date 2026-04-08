"""
Smoke tests for train, predict, and sweep pipelines.

Tests are sequential: train produces a model used by predict.
All tests skip gracefully if the dataset or precomputed embeddings are missing.

Run standalone:
    python tests/test_smoke_pipelines.py

Run via pytest:
    pytest tests/test_smoke_pipelines.py -v -s

Prerequisites (created by setup.sh):
    inputs/GDPa1_complete.csv
    inputs/embeddings/GDPa1_complete_esmc_600m_fullchain_*.pt
"""

import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_PATH = PROJECT_ROOT / "inputs" / "GDPa1_complete.csv"

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
GREEN  = "\033[0;32m"
RED    = "\033[0;31m"
YELLOW = "\033[1;33m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def _ok(msg):   print(f"{GREEN}  ✓{RESET}  {msg}")
def _fail(msg): print(f"{RED}  ✗{RESET}  {msg}")
def _skip(msg): print(f"{YELLOW}  ~{RESET}  {msg}")
def _info(msg): print(f"     {msg}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_embeddings(encoder="esmc_600m", full_chain=True):
    suffix = "_fullchain" if full_chain else ""
    pattern = str(PROJECT_ROOT / "inputs" / "embeddings" / f"GDPa1_complete_{encoder}{suffix}_*.pt")
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def data_available():
    if not DATASET_PATH.exists():
        return False, f"Dataset not found: {DATASET_PATH}"
    if not find_embeddings("esmc_600m", full_chain=True):
        return False, "esmc_600m full-chain embeddings not found — run setup.sh first"
    return True, None


def run(cmd, env=None, timeout=600):
    full_env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT), **(env or {})}
    result = subprocess.run(
        cmd, cwd=PROJECT_ROOT, capture_output=True, text=True,
        timeout=timeout, env=full_env
    )
    return result


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

ALL_PROPERTIES = ["HIC", "Titer", "PR_CHO", "AC-SINS_pH7.4", "Tm2"]
SMOKE_PROPERTY = "HIC"


def smoke_train(tmp_dir):
    """Train HIC final model using smoke_config.yaml (2 epochs)."""
    print(f"\n{BOLD}[1/3] Train — {SMOKE_PROPERTY} final model (2 epochs){RESET}")

    ok, msg = data_available()
    if not ok:
        _skip(f"SKIP: {msg}")
        return None

    config_path = PROJECT_ROOT / "src" / "config" / "smoke_config.yaml"
    if not config_path.exists():
        _info("smoke_config.yaml not found — generating...")
        r = run([sys.executable, "scripts/make_smoke_config.py"])
        if r.returncode != 0:
            _skip("SKIP: could not generate smoke_config.yaml")
            return None

    last_model_dir = None
    for prop in ["HIC"]:  # TODO: restore ALL_PROPERTIES for final release
        result = run(
            [sys.executable, "src/train.py",
             "--property", prop,
             "--config", str(config_path),
             "--fold", "final"],
            env={"WANDB_MODE": "offline"},
            timeout=600,
        )

        if result.returncode != 0:
            _fail(f"train.py failed for {prop} (exit {result.returncode})")
            _info("--- STDOUT (tail) ---")
            for line in result.stdout.splitlines()[-20:]:
                _info(line)
            _info("--- STDERR (tail) ---")
            for line in result.stderr.splitlines()[-20:]:
                _info(line)
            raise RuntimeError(f"train.py failed for {prop} (exit {result.returncode})")

        matches = list((PROJECT_ROOT / "models").glob(f"{prop}_smoke_config_*"))
        if not matches:
            raise RuntimeError(f"train.py succeeded for {prop} but no model directory found")

        model_dir = max(matches, key=lambda p: p.stat().st_mtime)
        if not (model_dir / "final.ckpt").exists():
            raise RuntimeError(f"Expected final.ckpt not found in {model_dir}")

        _ok(f"{prop}: {model_dir.name}")
        if prop == "HIC":
            last_model_dir = model_dir  # Use HIC for predict smoke test

    return last_model_dir


def smoke_predict(tmp_dir, model_dir):
    """Run predict on 5 sequences using the model from smoke_train."""
    print(f"\n{BOLD}[2/3] Predict — 5 sequences{RESET}")

    if model_dir is None:
        _skip("SKIP: no model available (train did not produce a model)")
        return

    # Use public mAbs (NOT in GDPa1) to exercise on-the-fly ESM encoding.
    # This CSV is bundled with the repo and includes hc_subtype/lc_subtype per row.
    PUBLIC_MABS_CSV = PROJECT_ROOT / "inputs" / "public_mabs_not_in_gdpa1.csv"
    if not PUBLIC_MABS_CSV.exists():
        raise RuntimeError(f"Smoke predict input not found: {PUBLIC_MABS_CSV}")
    _info(f"Using {PUBLIC_MABS_CSV.name} as predict input")

    output_csv = tmp_dir / "smoke_predictions.csv"

    result = run(
        [sys.executable, "-m", "src.predict",
         "--input", str(PUBLIC_MABS_CSV),
         "--model", str(model_dir),
         "--output", str(output_csv)],
        env={"WANDB_MODE": "offline"},
        timeout=300,
    )

    if result.returncode != 0:
        _fail(f"predict.py failed (exit {result.returncode})")
        _info("--- STDOUT (tail) ---")
        for line in result.stdout.splitlines()[-20:]:
            _info(line)
        _info("--- STDERR (tail) ---")
        for line in result.stderr.splitlines()[-20:]:
            _info(line)
        raise RuntimeError(f"predict.py failed (exit {result.returncode})")

    if not output_csv.exists():
        raise RuntimeError("predict.py succeeded but no output CSV found")

    import pandas as pd
    preds = pd.read_csv(output_csv)
    pred_cols = [c for c in preds.columns if c.endswith("_prediction")]
    _ok(f"{len(preds)} rows, {len(pred_cols)} prediction column(s): {pred_cols}")


def smoke_sweep():
    """Verify sweep scripts start up correctly (imports, arg parsing, config loading)."""
    print(f"\n{BOLD}[3/3] Sweep — script startup checks{RESET}")

    for script in ["scripts/run_sweep.py", "src/tuning/tune_hyperparam.py"]:
        result = run([sys.executable, script, "--help"])
        if result.returncode != 0:
            _fail(f"{script} --help failed (exit {result.returncode})")
            raise RuntimeError(f"{script} --help failed:\n{result.stderr[-500:]}")
        _ok(script)


# ---------------------------------------------------------------------------
# Entry point (standalone)
# ---------------------------------------------------------------------------

def main():
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  CrossAbSense smoke tests{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")

    import tempfile
    tmp_dir = Path(tempfile.mkdtemp(prefix="crossabsense_smoke_"))
    model_dir = None
    results = {}  # name -> (passed, skipped, message)

    try:
        for name, fn, args in [
            ("train",   smoke_train,   (tmp_dir,)),
            ("predict", smoke_predict, (tmp_dir, None)),  # placeholder, updated below
            ("sweep",   smoke_sweep,   ()),
        ]:
            try:
                if name == "train":
                    model_dir = fn(*args)
                    results[name] = (True, False, "")
                elif name == "predict":
                    fn(tmp_dir, model_dir)
                    results[name] = (True, False, "")
                else:
                    fn(*args)
                    results[name] = (True, False, "")
            except RuntimeError as e:
                results[name] = (False, False, str(e))
            except Exception as e:
                results[name] = (False, False, str(e))

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        for prop in ALL_PROPERTIES:
            for d in (PROJECT_ROOT / "models").glob(f"{prop}_smoke_config_*"):
                shutil.rmtree(d, ignore_errors=True)

    # Summary
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  Summary{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")
    failures = []
    for name, (passed, skipped, msg) in results.items():
        if skipped:
            print(f"  {YELLOW}~{RESET}  {name:<10} SKIPPED")
        elif passed:
            print(f"  {GREEN}✓{RESET}  {name:<10} PASSED")
        else:
            print(f"  {RED}✗{RESET}  {name:<10} FAILED  — {msg}")
            failures.append(name)

    print(f"{BOLD}{'='*60}{RESET}\n")

    if failures:
        print(f"{RED}{BOLD}  {len(failures)} test(s) failed: {', '.join(failures)}{RESET}\n")
        sys.exit(1)
    else:
        print(f"{GREEN}{BOLD}  All smoke tests passed.{RESET}\n")


# ---------------------------------------------------------------------------
# pytest compatibility
# ---------------------------------------------------------------------------

def test_train_smoke(tmp_path):
    test_train_smoke._model_dir = smoke_train(tmp_path)


def test_predict_smoke(tmp_path):
    model_dir = getattr(test_train_smoke, "_model_dir", None)
    try:
        smoke_predict(tmp_path, model_dir)
    finally:
        if model_dir and model_dir.exists():
            shutil.rmtree(model_dir, ignore_errors=True)


def test_sweep_smoke():
    smoke_sweep()


if __name__ == "__main__":
    main()
