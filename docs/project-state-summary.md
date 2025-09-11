# DecodingDD Project State Summary

## Project Overview
This is a sophisticated fMRI analysis project implementing decision-making research with delay discounting paradigms. The project has achieved production-grade reliability through extensive debugging and refinement.

## Key Architecture Decisions

### Multi-Model GLM Approach
- **Problem**: Severe multicollinearity between behavioral regressors (r=0.99+ correlations)
- **Solution**: Separate GLM models for each regressor to avoid VIF issues
- **Implementation**: `model_specifications` in config defines distinct models for:
  - `choice`: Choice regressor only
  - `value_chosen`: SVchosen regressor only  
  - `value_unchosen`: SVunchosen regressor only
  - `value_difference`: SVdiff regressor only
  - `large_amount`: Large amount regressor only

### SLURM Infrastructure Design
- **Dynamic Array Jobs**: Auto-calculate subject counts, no hard-coded bounds
- **Environment Independence**: All scripts use PROJECT_ROOT, work from any location
- **Comprehensive Validation**: Every environment variable and path checked
- **Bulletproof Error Handling**: set -e, ${VAR:?} checks, directory validation

## Critical Implementation Details

### GLM Script (scripts/modeling/run_standard_glm.py)
- **Confound Harmonization**: Zero-fills missing confounds for design matrix consistency
- **VIF Calculation**: Per-run VIF calculation using sklearn LinearRegression
- **Onset Validation**: Multi-criteria detection for run-relative vs session-relative timing
- **Choice Recoding**: Robust handling of both string labels and numeric values
- **Error Recovery**: Graceful handling of edge cases (single data points, row mismatches)

### Configuration Structure (config/project_config.yaml)
```yaml
glm:
  model_specifications:
    choice: ['choice']
    value_chosen: ['SVchosen']
    value_unchosen: ['SVunchosen'] 
    value_difference: ['SVdiff']
    large_amount: ['large_amount']
```

## File Organization
```
scripts/modeling/run_standard_glm.py    # Main GLM script
config/project_config.yaml             # Central configuration
tests/test_modeling.py                  # Integration tests
slurm/submit_*_dynamic.sh              # Dynamic job submission
slurm/prepare_slurm_environment.sh     # Environment setup
```

## Quality Assurance
- **Testing**: Comprehensive integration tests with synthetic data
- **VIF Monitoring**: Automatic multicollinearity detection
- **Validation**: Strict parameter and file validation throughout
- **Documentation**: Detailed error messages and usage guidance

## Current Status: PRODUCTION READY
All major debugging completed through 6 comprehensive sessions:
1. Initial GLM robustness (confounds, onsets, modulators, VIF)
2. Advanced reliability (missing files, VIF scope, onset alignment)
3. Multi-model architecture implementation
4. Edge case refinement (choice recoding, onset detection, confound handling)
5. Final GLM perfection (session timing, row count validation)
6. Ultimate SLURM perfection (log dirs, dynamic bounds, unused vars)

## Future Continuity Notes
- **Dependencies**: User does not use requirements.txt, manages dependencies differently
- **Testing**: Always run pytest after major changes
- **SLURM**: Use dynamic submission scripts, avoid manual array bounds
- **Config**: Central configuration in project_config.yaml drives all analysis
