#!/usr/bin/env python
"""
Data setup validation and initialization script.

This script validates that all training data is properly set up
and provides diagnostics for data structure issues.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List

from config import (
    DATA_DIR, REAL_DATA_DIR, SYNTH_DATA_DIR, ORIGINAL_IMG_DIR,
    REAL_COMPONENT_JSON, REAL_TRAIN_SPLIT, REAL_TEST_SPLIT,
    SYNTH_COMPONENT_JSON, SYNTH_TRAIN_SPLIT, SYNTH_TEST_SPLIT,
    REAL_IMG_DIR, SYNTH_IMG_DIR, validate_data_structure
)


class DataValidator:
    """Validates and reports on data structure and content."""
    
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.stats = {}
    
    def check_files_exist(self) -> bool:
        """Check if all required files exist."""
        print("\n" + "=" * 60)
        print("CHECKING FILE STRUCTURE")
        print("=" * 60)
        
        required = {
            'Real Data': [
                (REAL_COMPONENT_JSON, 'component.json'),
                (REAL_TRAIN_SPLIT, 'train_split.json'),
                (REAL_TEST_SPLIT, 'test_split.json'),
            ],
            'Synthetic Data': [
                (SYNTH_COMPONENT_JSON, 'component.json'),
                (SYNTH_TRAIN_SPLIT, 'train_split.json'),
                (SYNTH_TEST_SPLIT, 'test_split.json'),
            ],
            'Directories': [
                (REAL_IMG_DIR, 'img/'),
                (SYNTH_IMG_DIR, 'img/'),
                (ORIGINAL_IMG_DIR, 'original_img/'),
            ]
        }
        
        all_exist = True
        for category, items in required.items():
            print(f"\n{category}:")
            for path, label in items:
                exists = path.exists()
                status = "✅" if exists else "❌"
                print(f"  {status} {label}: {path.relative_to(Path.cwd()) if path.exists() else path}")
                if not exists:
                    all_exist = False
                    self.issues.append(f"Missing: {path}")
        
        return all_exist
    
    def check_data_integrity(self) -> bool:
        """Check the integrity of data files."""
        print("\n" + "=" * 60)
        print("CHECKING DATA INTEGRITY")
        print("=" * 60)
        
        all_valid = True
        
        # Check JSON files
        json_files = {
            'Real Data (component.json)': REAL_COMPONENT_JSON,
            'Real Train Split': REAL_TRAIN_SPLIT,
            'Real Test Split': REAL_TEST_SPLIT,
            'Synth Data (component.json)': SYNTH_COMPONENT_JSON,
            'Synth Train Split': SYNTH_TRAIN_SPLIT,
            'Synth Test Split': SYNTH_TEST_SPLIT,
        }
        
        for label, path in json_files.items():
            if not path.exists():
                print(f"\n❌ {label}: File not found")
                all_valid = False
                continue
            
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                
                if isinstance(data, dict):
                    num_items = len(data)
                    print(f"✅ {label}: Valid JSON with {num_items} entries")
                    self.stats[label] = num_items
                elif isinstance(data, list):
                    num_items = len(data)
                    print(f"✅ {label}: Valid JSON with {num_items} entries")
                    self.stats[label] = num_items
                else:
                    print(f"⚠️  {label}: Unknown JSON structure")
                    self.warnings.append(f"Unknown structure in {label}")
            
            except json.JSONDecodeError as e:
                print(f"❌ {label}: Invalid JSON - {e}")
                all_valid = False
                self.issues.append(f"Invalid JSON in {label}")
            except Exception as e:
                print(f"❌ {label}: Error reading file - {e}")
                all_valid = False
                self.issues.append(f"Error reading {label}: {e}")
        
        return all_valid
    
    def check_image_files(self) -> bool:
        """Check if image files exist in expected directories."""
        print("\n" + "=" * 60)
        print("CHECKING IMAGE FILES")
        print("=" * 60)
        
        all_valid = True
        
        # Check original page images
        if ORIGINAL_IMG_DIR.exists():
            img_files = list(ORIGINAL_IMG_DIR.glob('*.png'))
            print(f"\n📁 original_img/: {len(img_files)} PNG files")
            if len(img_files) == 0:
                self.warnings.append("No PNG files in original_img/")
                all_valid = False
            else:
                print(f"  ✅ Found {len(img_files)} page images")
        else:
            print(f"\n❌ original_img/ directory not found")
            all_valid = False
        
        # Check real data images
        if REAL_IMG_DIR.exists():
            img_files = list(REAL_IMG_DIR.glob('*.*'))
            print(f"\n📁 real/img/: {len(img_files)} files")
        else:
            print(f"\n⚠️  real/img/ directory not found")
        
        # Check synth data images
        if SYNTH_IMG_DIR.exists():
            img_files = list(SYNTH_IMG_DIR.glob('*.*'))
            print(f"\n📁 synth/img/: {len(img_files)} files")
        else:
            print(f"\n⚠️  synth/img/ directory not found")
        
        return all_valid
    
    def print_summary(self):
        """Print validation summary."""
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        
        if self.issues:
            print(f"\n❌ ISSUES ({len(self.issues)}):")
            for issue in self.issues:
                print(f"   - {issue}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   - {warning}")
        
        if self.stats:
            print(f"\n📊 DATA STATISTICS:")
            for label, count in self.stats.items():
                print(f"   {label}: {count:,} items")
        
        if not self.issues and not self.warnings:
            print("\n✅ All checks passed!")
            return True
        elif not self.issues:
            print(f"\n⚠️  {len(self.warnings)} warning(s), but data structure is valid")
            return True
        else:
            print(f"\n❌ {len(self.issues)} critical issue(s) found")
            return False


def main():
    """Run all validation checks."""
    print("\n" + "=" * 60)
    print("STRIKEOUT DETECTOR - DATA VALIDATION")
    print("=" * 60)
    print(f"Working directory: {Path.cwd()}")
    print(f"Data directory: {DATA_DIR}")
    
    validator = DataValidator()
    
    # Run checks
    files_ok = validator.check_files_exist()
    integrity_ok = validator.check_data_integrity()
    images_ok = validator.check_image_files()
    
    # Print summary
    validator.print_summary()
    
    # Exit with appropriate code
    if validator.issues:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
