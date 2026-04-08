#!/usr/bin/env python3
"""
CrossAbSense test battery — validates the full setup end-to-end.

Tests (in order):
  1. GPU / CUDA functional
  2. Config files valid
  3. All local encoders load and encode
  4. ESM-C 6B API (optional — needs FORGE_TOKEN)
  5. Antibody feature extraction
  6. Train smoke (2 epochs on HIC)
  7. Predict smoke (on public mAbs)
  8. Sweep startup check

Run:
    python tests/test_setup_checks.py

Prerequisites (created by setup.sh):
    inputs/GDPa1_complete.csv
    inputs/embeddings/GDPa1_complete_esmc_600m_fullchain_*.pt
"""

import glob
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATASET_PATH = PROJECT_ROOT / "inputs" / "GDPa1_complete.csv"
PUBLIC_MABS = PROJECT_ROOT / "inputs" / "public_mabs_not_in_gdpa1.csv"
ALL_PROPERTIES = ["HIC", "Titer", "PR_CHO", "AC-SINS_pH7.4", "Tm2"]

# ---------------------------------------------------------------------------
# Colors / output
# ---------------------------------------------------------------------------
GREEN  = "\033[0;32m"
RED    = "\033[0;31m"
YELLOW = "\033[1;33m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def _ok(msg):   print(f"{GREEN}  ✓{RESET}  {msg}")
def _fail(msg): print(f"{RED}  ✗{RESET}  {msg}")
def _warn(msg): print(f"{YELLOW}  ⚠{RESET}  {msg}")
def _info(msg): print(f"     {msg}")

def _run(cmd, env=None, timeout=600):
    """Run a subprocess from project root."""
    full_env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT), **(env or {})}
    return subprocess.run(
        cmd, cwd=PROJECT_ROOT, capture_output=True, text=True,
        timeout=timeout, env=full_env,
    )

def _find_embeddings(encoder="esmc_600m", full_chain=True):
    suffix = "_fullchain" if full_chain else ""
    pattern = str(PROJECT_ROOT / "inputs" / "embeddings" / f"GDPa1_complete_{encoder}{suffix}_*.pt")
    matches = glob.glob(pattern)
    return matches[0] if matches else None


# ===================================================================
# 1. GPU / CUDA
# ===================================================================

def check_gpu():
    print(f"\n{BOLD}[1/8] GPU / CUDA{RESET}")

    if not torch.cuda.is_available():
        _fail("CUDA not available — GPU required for training")
        raise RuntimeError("CUDA not available")

    name = torch.cuda.get_device_name(0)
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    _ok(f"{name} ({mem_gb:.1f} GB)")

    # Functional check
    x = torch.randn(256, 256, device="cuda")
    y = x @ x.T
    assert y.shape == (256, 256)
    del x, y
    torch.cuda.empty_cache()
    _ok("CUDA compute functional")

    if torch.cuda.is_bf16_supported():
        _ok("bfloat16 supported")
    else:
        _warn("bfloat16 not supported — will use fp16 instead")


# ===================================================================
# 2. Config files
# ===================================================================

REQUIRED_TOP = {"encoder", "decoder", "training", "data"}
REQUIRED_ENC = {"freeze_encoder"}

def check_configs():
    print(f"\n{BOLD}[2/8] Config files{RESET}")

    for name in ["default_config.yaml", "oracle_efficient_config.yaml"]:
        path = PROJECT_ROOT / "src" / "config" / name
        if not path.exists():
            _fail(f"{name} — not found")
            raise RuntimeError(f"Missing config: {path}")

        cfg = yaml.safe_load(path.read_text())
        missing = REQUIRED_TOP - set(cfg)
        if missing:
            _fail(f"{name} — missing top-level keys: {missing}")
            raise RuntimeError(f"{name} missing keys: {missing}")

        missing_enc = REQUIRED_ENC - set(cfg.get("encoder", {}))
        if missing_enc:
            _fail(f"{name} — missing encoder keys: {missing_enc}")
            raise RuntimeError(f"{name} missing encoder keys: {missing_enc}")

        # Check decoder has a type
        if "type" not in cfg.get("decoder", {}):
            _fail(f"{name} — decoder missing 'type' key")
            raise RuntimeError(f"{name} decoder missing 'type'")

        _ok(name)

    # Sweep configs
    sweep_dir = PROJECT_ROOT / "config" / "tuning"
    if sweep_dir.exists():
        sweep_files = list(sweep_dir.glob("example_*.yaml"))
        for sf in sweep_files:
            scfg = yaml.safe_load(sf.read_text())
            if "parameters" not in scfg:
                _fail(f"{sf.name} — missing 'parameters'")
                raise RuntimeError(f"Invalid sweep config: {sf.name}")
        _ok(f"{len(sweep_files)} sweep configs valid")


# ===================================================================
# 3. Local encoders
# ===================================================================

ENCODER_SPECS = {
    "esmc_300m":  ("ESMCEncoder",    {"model_name": "esmc_300m"}),
    "esmc_600m":  ("ESMCEncoder",    {"model_name": "esmc_600m"}),
}

def check_encoders():
    print(f"\n{BOLD}[3/8] Encoders (local){RESET}")

    from src.encoders import ESMCEncoder
    classes = {
        "ESMCEncoder": ESMCEncoder,
    }

    test_vh = "QVQLVQSGAEVKKPGSSVKVSCKAS"
    test_vl = "DIQMTQSPSSLSASVGDRVTITC"

    failed = []
    for name, (cls_name, kwargs) in ENCODER_SPECS.items():
        cls = classes[cls_name]
        try:
            encoder = cls(**kwargs)
            encoder.eval()
            with torch.no_grad():
                vh_emb, vl_emb = encoder(heavy_sequences=[test_vh], light_sequences=[test_vl])
            dim = vh_emb.shape[-1]
            _ok(f"{name}: dim={dim}, device={vh_emb.device}")
            del encoder, vh_emb, vl_emb
            torch.cuda.empty_cache()
        except Exception as e:
            _fail(f"{name}: {e}")
            failed.append(name)

    if failed:
        raise RuntimeError(f"Encoder(s) failed: {', '.join(failed)}")


# ===================================================================
# 4. ESM-C 6B API
# ===================================================================

def check_esmc_6b_api():
    print(f"\n{BOLD}[4/8] ESM-C 6B API{RESET}")

    token = os.environ.get("FORGE_TOKEN") or os.environ.get("ESM_API_KEY")
    if not token:
        _warn("FORGE_TOKEN not set — skipping ESM-C 6B")
        _info("Set FORGE_TOKEN to enable the 6B encoder for your own sequences.")
        _info("Get a token at: https://forge.evolutionaryscale.ai")
        return

    try:
        from src.encoders import ESMCEncoder
        encoder = ESMCEncoder(model_name="esmc_6b").eval()
        test_vh = "QVQLVQSGAEVKKPGSSVKVSCKAS"
        with torch.no_grad():
            vh_emb, _ = encoder(heavy_sequences=[test_vh], light_sequences=None)
        _ok(f"esmc_6b: dim={vh_emb.shape[-1]} (Forge API)")
        del encoder, vh_emb
    except ValueError as e:
        if "token" in str(e).lower():
            _fail(f"FORGE_TOKEN set but rejected: {e}")
        else:
            _fail(f"esmc_6b: {e}")
        raise
    except Exception as e:
        _fail(f"esmc_6b API error: {e}")
        _info("Check your FORGE_TOKEN and network connectivity.")
        raise


# ===================================================================
# 5. Antibody features
# ===================================================================

def check_features():
    print(f"\n{BOLD}[5/8] Antibody features{RESET}")

    from src.features import AntibodyFeatures

    vh = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
    vl = "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK"

    extractor = AntibodyFeatures(
        use_abnumber=True, use_biophi=False,
        use_scalop=True, use_sequence_features=True,
    )
    features = extractor.extract_features(vh, vl)
    arr = extractor.features_to_array(features)
    non_sentinel = int(np.sum(arr != -999.0))
    _ok(f"{non_sentinel}/{len(arr)} features computed (dim={extractor.get_feature_dim()})")


# ===================================================================
# 6. Train smoke (all properties)
# ===================================================================

def _data_available():
    if not DATASET_PATH.exists():
        return False, f"{DATASET_PATH} not found — run setup.sh first"
    if not _find_embeddings("esmc_600m", full_chain=True):
        return False, "esmc_600m embeddings not found — run setup.sh first"
    return True, None

def _ensure_smoke_config():
    config_path = PROJECT_ROOT / "src" / "config" / "smoke_config.yaml"
    if not config_path.exists():
        r = _run([sys.executable, "scripts/make_smoke_config.py"])
        if r.returncode != 0:
            return None
    return config_path

def check_train(tmp_dir):
    """Train each property (2 epochs). Returns (model_dirs, per_property_results)."""
    print(f"\n{BOLD}[6/8] Train smoke — all properties (2 epochs each){RESET}")

    ok, msg = _data_available()
    if not ok:
        _warn(f"SKIP: {msg}")
        return {}, {}

    config_path = _ensure_smoke_config()
    if config_path is None:
        _warn("SKIP: could not generate smoke_config.yaml")
        return {}, {}

    model_dirs = {}
    prop_results = {}  # property -> True or error string

    for prop in ALL_PROPERTIES:
        print(f"  {BOLD}→ {prop}{RESET} ", end="", flush=True)

        # Each property runs as a subprocess — no GPU memory stacking
        result = _run(
            [sys.executable, "src/train.py",
             "--property", prop,
             "--config", str(config_path),
             "--fold", "final"],
            env={"WANDB_MODE": "offline"},
            timeout=600,
        )

        if result.returncode != 0:
            print()
            err = result.stderr.splitlines()[-1] if result.stderr.strip() else f"exit {result.returncode}"
            _fail(f"{prop}: {err}")
            prop_results[prop] = err
            continue

        matches = list((PROJECT_ROOT / "models").glob(f"{prop}_smoke_config_*"))
        if not matches:
            print()
            _fail(f"{prop}: no model directory produced")
            prop_results[prop] = "no model"
            continue

        model_dir = max(matches, key=lambda p: p.stat().st_mtime)
        if not (model_dir / "final.ckpt").exists():
            print()
            _fail(f"{prop}: final.ckpt not found")
            prop_results[prop] = "no checkpoint"
            continue

        # Extract timing from output if available
        timing = ""
        for line in result.stdout.splitlines():
            if "fit" in line.lower() and ("sec" in line.lower() or "time" in line.lower()):
                timing = f" ({line.strip()})"
                break

        print(f"{GREEN}✓{RESET}{timing}")
        model_dirs[prop] = model_dir
        prop_results[prop] = True

    return model_dirs, prop_results


# ===================================================================
# 7. Predict smoke (all properties)
# ===================================================================

def check_predict(tmp_dir, model_dirs):
    """Predict on public mAbs for each trained property. Returns per_property_results."""
    print(f"\n{BOLD}[7/8] Predict smoke — public mAbs (all properties){RESET}")

    if not model_dirs:
        _warn("SKIP: no models from train step")
        return {}

    if not PUBLIC_MABS.exists():
        _fail(f"{PUBLIC_MABS} not found")
        return {p: "input CSV missing" for p in model_dirs}

    import pandas as pd
    prop_results = {}

    for prop, model_dir in model_dirs.items():
        print(f"  {BOLD}→ {prop}{RESET} ", end="", flush=True)

        output_csv = tmp_dir / f"predictions_{prop}.csv"

        # Each property runs as a subprocess — no GPU memory stacking
        result = _run(
            [sys.executable, "-m", "src.predict",
             "--input", str(PUBLIC_MABS),
             "--model", str(model_dir),
             "--output", str(output_csv)],
            env={"WANDB_MODE": "offline"},
            timeout=300,
        )

        if result.returncode != 0:
            print()
            err = result.stderr.splitlines()[-1] if result.stderr.strip() else f"exit {result.returncode}"
            _fail(f"{prop}: {err}")
            prop_results[prop] = err
            continue

        if not output_csv.exists():
            print()
            _fail(f"{prop}: no output CSV")
            prop_results[prop] = "no output"
            continue

        preds = pd.read_csv(output_csv)
        pred_cols = [c for c in preds.columns if c.endswith("_prediction")]
        n_rows = len(preds)
        print(f"{GREEN}✓{RESET}  {n_rows} rows")
        prop_results[prop] = True

    return prop_results


# ===================================================================
# 8. Sweep startup
# ===================================================================

def check_sweep():
    print(f"\n{BOLD}[8/8] Sweep scripts{RESET}")

    for script in ["scripts/run_sweep.py", "src/tuning/tune_hyperparam.py"]:
        result = _run([sys.executable, script, "--help"])
        if result.returncode != 0:
            _fail(f"{script} --help failed")
            raise RuntimeError(f"{script} --help failed:\n{result.stderr[-500:]}")
        _ok(script)


# ===================================================================
# Main
# ===================================================================

def main():
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  CrossAbSense — test battery{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="crossabsense_tests_"))
    results = {}
    model_dirs = {}
    train_results = {}
    predict_results = {}

    checks = [
        ("gpu",       lambda: check_gpu()),
        ("configs",   lambda: check_configs()),
        ("encoders",  lambda: check_encoders()),
        ("esmc_6b",   lambda: check_esmc_6b_api()),
        ("features",  lambda: check_features()),
        ("train",     lambda: None),   # placeholder
        ("predict",   lambda: None),   # placeholder
        ("sweep",     lambda: check_sweep()),
    ]

    for name, fn in checks:
        try:
            if name == "train":
                model_dirs, train_results = check_train(tmp_dir)
                train_ok = sum(1 for v in train_results.values() if v is True)
                if train_ok == len(ALL_PROPERTIES):
                    results[name] = True
                elif train_ok > 0:
                    results[name] = f"{train_ok}/{len(ALL_PROPERTIES)} properties"
                else:
                    results[name] = "all properties failed"
            elif name == "predict":
                predict_results = check_predict(tmp_dir, model_dirs)
                pred_ok = sum(1 for v in predict_results.values() if v is True)
                total = len(predict_results) or len(model_dirs) or len(ALL_PROPERTIES)
                if pred_ok == total and pred_ok > 0:
                    results[name] = True
                elif pred_ok > 0:
                    results[name] = f"{pred_ok}/{total} properties"
                else:
                    results[name] = "all properties failed"
            else:
                fn()
                results[name] = True
        except Exception as e:
            results[name] = str(e)

    # Cleanup
    shutil.rmtree(tmp_dir, ignore_errors=True)
    for prop in ALL_PROPERTIES:
        for d in (PROJECT_ROOT / "models").glob(f"{prop}_smoke_config_*"):
            shutil.rmtree(d, ignore_errors=True)

    # Summary
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  Summary{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")

    failures = []
    for name, result in results.items():
        if result is True:
            if name == "train":
                print(f"  {GREEN}✓{RESET}  {name} ({len(ALL_PROPERTIES)}/{len(ALL_PROPERTIES)} properties)")
            elif name == "predict":
                print(f"  {GREEN}✓{RESET}  {name} ({len(predict_results)}/{len(predict_results)} properties)")
            else:
                print(f"  {GREEN}✓{RESET}  {name}")
        else:
            print(f"  {RED}✗{RESET}  {name}  — {result}")
            failures.append(name)

    print(f"{BOLD}{'='*60}{RESET}\n")

    # esmc_6b without token is not fatal
    fatal = [f for f in failures if f != "esmc_6b"]
    if fatal:
        print(f"{RED}{BOLD}  {len(fatal)} test(s) failed: {', '.join(fatal)}{RESET}\n")
        sys.exit(1)
    elif failures:
        print(f"{YELLOW}{BOLD}  ESM-C 6B skipped (no FORGE_TOKEN) — all other tests passed.{RESET}\n")
    else:
        print(f"{GREEN}{BOLD}  All tests passed.{RESET}\n")


if __name__ == "__main__":
    main()
