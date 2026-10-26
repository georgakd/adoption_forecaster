# Output Validation Script

A simple tool to check if your code changes produce the same output files as the main branch.

## What it does

1. **Switches to main branch** → Runs predictions → Saves output files
2. **Switches to your branch** → Runs predictions → Saves output files  
3. **Compares all output files** → Shows if they're identical or different

## Quick Start

```bash
# Compare your current branch with main
python testing/validation_local.py --base-branch main --compare-branch your-branch-name --pred-year 2026 --last-year 2024

# See detailed differences if files don't match
python testing/validation_local.py --base-branch main --compare-branch your-branch-name --pred-year 2026 --last-year 2024 --show-diffs
```

## What files it checks

- `Education/output/2026_Education_16-64_predictions`
- `Education/output/2026_Education_16-74_predictions` 
- `Age_Gender/output/2026_Age_Gender_16-74_predictions`
- All their `_with_past_data` versions

## Example output

**✅ Success:**
```
🎉 SUCCESS: All output files are identical!
```

**❌ Failure:**
```
❌ Education/output/2026_Education_16-64_predictions - DIFFERENT!
⚠️  WARNING: Some output files are different!
```

## When to use

- **Before merging** your feature branch
- **After refactoring** to make sure nothing broke
- **During code review** to verify changes don't affect results

## Requirements

- Python 3.7+
- Git repository
- `pandas` and `numpy` installed
- Input data files available

## Notes

- The script automatically stashes your uncommitted changes
- It runs the full prediction pipeline on both branches
- Temporary files are cleaned up automatically
- Exit code 0 = success, 1 = failure
