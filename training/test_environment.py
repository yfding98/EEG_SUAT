#!/usr/bin/env python3
"""
Test script to verify the environment is set up correctly.
"""

import sys
import os
from pathlib import Path

def test_encoding():
    """Test encoding settings."""
    print("=" * 60)
    print("Testing Encoding Settings")
    print("=" * 60)
    
    import locale
    
    tests_passed = []
    
    # Test 1: Check PYTHONIOENCODING
    print("\n1. Environment Variable:")
    pythonioencoding = os.environ.get('PYTHONIOENCODING', 'NOT SET')
    print(f"   PYTHONIOENCODING = {pythonioencoding}")
    if pythonioencoding.lower() == 'utf-8':
        print("   ✓ PASS")
        tests_passed.append(True)
    else:
        print("   ⚠ WARNING: Not set to UTF-8")
        print("   Set it with: $env:PYTHONIOENCODING='utf-8' (PowerShell)")
        print("              or: set PYTHONIOENCODING=utf-8 (CMD)")
        tests_passed.append(False)
    
    # Test 2: Check stdout encoding
    print("\n2. Standard Output Encoding:")
    stdout_encoding = sys.stdout.encoding
    print(f"   sys.stdout.encoding = {stdout_encoding}")
    if stdout_encoding and 'utf' in stdout_encoding.lower():
        print("   ✓ PASS")
        tests_passed.append(True)
    else:
        print("   ⚠ WARNING: Not UTF-8")
        tests_passed.append(False)
    
    # Test 3: Check default encoding
    print("\n3. Default Encoding:")
    default_encoding = sys.getdefaultencoding()
    print(f"   sys.getdefaultencoding() = {default_encoding}")
    if default_encoding == 'utf-8':
        print("   ✓ PASS")
        tests_passed.append(True)
    else:
        print("   ℹ INFO: Not UTF-8 (this is usually OK)")
        tests_passed.append(True)
    
    # Test 4: Check filesystem encoding
    print("\n4. Filesystem Encoding:")
    fs_encoding = sys.getfilesystemencoding()
    print(f"   sys.getfilesystemencoding() = {fs_encoding}")
    print("   ℹ INFO: This depends on your OS")
    tests_passed.append(True)
    
    # Test 5: Check locale
    print("\n5. Locale Encoding:")
    locale_encoding = locale.getpreferredencoding()
    print(f"   locale.getpreferredencoding() = {locale_encoding}")
    print("   ℹ INFO: This depends on your system settings")
    tests_passed.append(True)
    
    return all(tests_passed)


def test_imports():
    """Test that all required modules can be imported."""
    print("\n" + "=" * 60)
    print("Testing Required Modules")
    print("=" * 60)
    
    required_modules = [
        'torch',
        'numpy',
        'pandas',
        'tqdm',
        'sklearn',
    ]
    
    tests_passed = []
    
    for i, module in enumerate(required_modules, 1):
        print(f"\n{i}. {module}:")
        try:
            __import__(module)
            print(f"   ✓ PASS - {module} is installed")
            tests_passed.append(True)
        except ImportError as e:
            print(f"   ✗ FAIL - {module} not found: {e}")
            print(f"   Install with: pip install {module}")
            tests_passed.append(False)
    
    return all(tests_passed)


def test_paths():
    """Test that required paths exist."""
    print("\n" + "=" * 60)
    print("Testing Data Paths")
    print("=" * 60)
    
    paths_to_check = [
        ('Features Root', r'E:\output\connectivity_features'),
        ('Labels CSV', r'E:\output\connectivity_features\labels.csv'),
    ]
    
    tests_passed = []
    
    for i, (name, path) in enumerate(paths_to_check, 1):
        print(f"\n{i}. {name}:")
        print(f"   Path: {path}")
        if os.path.exists(path):
            print(f"   ✓ PASS - Path exists")
            if path.endswith('.csv'):
                # Try to read CSV
                try:
                    import pandas as pd
                    df = pd.read_csv(path, encoding='utf-8')
                    print(f"   ✓ CSV readable, {len(df)} rows")
                    tests_passed.append(True)
                except Exception as e:
                    print(f"   ⚠ WARNING - CSV exists but can't be read: {e}")
                    tests_passed.append(False)
            else:
                # Check if directory has files
                try:
                    files = list(Path(path).rglob('*.npz'))
                    print(f"   ✓ Found {len(files)} NPZ files")
                    tests_passed.append(True)
                except Exception as e:
                    print(f"   ⚠ WARNING - Directory exists but can't list files: {e}")
                    tests_passed.append(False)
        else:
            print(f"   ✗ FAIL - Path does not exist")
            tests_passed.append(False)
    
    return all(tests_passed)


def test_training_scripts():
    """Test that training scripts exist and are importable."""
    print("\n" + "=" * 60)
    print("Testing Training Scripts")
    print("=" * 60)
    
    scripts = [
        'training.contrastive_pretrain',
        'training.finetune',
        'training.datasets',
        'training.models',
        'training.attention_fusion',
    ]
    
    tests_passed = []
    
    for i, script in enumerate(scripts, 1):
        print(f"\n{i}. {script}:")
        try:
            __import__(script)
            print(f"   ✓ PASS - {script} is importable")
            tests_passed.append(True)
        except ImportError as e:
            print(f"   ✗ FAIL - {script} cannot be imported: {e}")
            tests_passed.append(False)
    
    return all(tests_passed)


def main():
    """Run all tests."""
    print("\n" + "🔍 Environment Test for Multi-Matrix EEG Training".center(60))
    print()
    
    results = []
    
    # Run tests
    results.append(("Encoding", test_encoding()))
    results.append(("Imports", test_imports()))
    results.append(("Paths", test_paths()))
    results.append(("Scripts", test_training_scripts()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:20s}: {status}")
    
    print("\n" + "=" * 60)
    
    if all(p for _, p in results):
        print("✓ ALL TESTS PASSED!")
        print("\nYou can now run training:")
        print("  - Double-click: training\\run_pretrain.bat")
        print("  - Or run: .\\training\\run_pretrain.ps1")
        return 0
    else:
        print("✗ SOME TESTS FAILED!")
        print("\nPlease fix the issues above before running training.")
        print("\nFor encoding issues:")
        print("  - PowerShell: $env:PYTHONIOENCODING='utf-8'")
        print("  - CMD: set PYTHONIOENCODING=utf-8")
        return 1


if __name__ == "__main__":
    sys.exit(main())
