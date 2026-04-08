#!/usr/bin/env python3
"""
Test runner for CrossAbSense project.

Usage:
    python tests/run_tests.py                    # Run all tests
    python tests/run_tests.py model              # Run model tests only
    python tests/run_tests.py training           # Run training tests only
    python tests/run_tests.py utils              # Run utility tests only
    python tests/run_tests.py integration        # Run integration tests only
    python tests/run_tests.py <test_file>        # Run specific test file
"""

import sys
import subprocess
from pathlib import Path
from typing import List, Optional


def get_test_files(category: Optional[str] = None) -> List[Path]:
    """Get test files based on category."""
    tests_dir = Path(__file__).parent
    
    if category is None:
        # All test files
        pattern = "test_*.py"
    elif category in ["model", "training", "utils", "integration"]:
        # Specific category
        pattern = f"test_{category}_*.py"
    else:
        # Specific file
        test_file = tests_dir / category
        if test_file.exists():
            return [test_file]
        test_file = tests_dir / f"{category}.py"
        if test_file.exists():
            return [test_file]
        print(f"Error: Test file not found: {category}")
        return []
    
    files = sorted(tests_dir.glob(pattern))
    # Exclude this runner script
    files = [f for f in files if f.name != "run_tests.py"]
    return files


def run_test(test_file: Path) -> bool:
    """Run a single test file."""
    print(f"\n{'=' * 80}")
    print(f"Running: {test_file.name}")
    print('=' * 80)
    
    try:
        result = subprocess.run(
            [sys.executable, str(test_file)],
            cwd=test_file.parent.parent,
            capture_output=False,
            text=True
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Error running test: {e}")
        return False


def main():
    """Run tests based on command line arguments."""
    category = sys.argv[1] if len(sys.argv) > 1 else None
    
    test_files = get_test_files(category)
    
    if not test_files:
        print("No test files found!")
        print(__doc__)
        sys.exit(1)
    
    print(f"\n{'#' * 80}")
    print(f"# CrossAbSense Test Suite")
    print(f"# Running {len(test_files)} test file(s)")
    print(f"{'#' * 80}\n")
    
    results = {}
    for test_file in test_files:
        success = run_test(test_file)
        results[test_file.name] = success
    
    # Summary
    print(f"\n{'=' * 80}")
    print("TEST SUMMARY")
    print('=' * 80)
    
    passed = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)
    
    for test_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {test_name:50} {status}")
    
    print()
    print(f"Total: {passed} passed, {failed} failed out of {len(results)} tests")
    print('=' * 80)
    print()
    
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
