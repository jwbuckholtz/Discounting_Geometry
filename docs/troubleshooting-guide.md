# DecodingDD Troubleshooting Guide

## Common Issues and Solutions

### GLM Script Issues

#### 1. VIF Warnings
**Symptom**: "High VIF detected" warnings
**Cause**: Multicollinearity between regressors
**Solution**: This is expected and handled by multi-model approach - each model runs separately

#### 2. Onset Timing Errors
**Symptom**: "CRITICAL: Run X has session-relative timing"
**Cause**: Event onsets are session-relative, not run-relative
**Solution**: Add explicit run_start_times to config:
```yaml
analysis_params:
  run_start_times:
    '1': 0      # If already run-relative
    '2': 300    # Actual run start time in seconds
```

#### 3. Missing Confound Files
**Symptom**: "Run X has no confounds - created zero-filled confound matrix"
**Cause**: Confound file missing for a run
**Solution**: This is automatically handled - zero-filled matrix ensures consistency

#### 4. Choice Recoding Failures
**Symptom**: "Unable to convert choice values"
**Cause**: Unexpected choice column format
**Solution**: Check choice column contains 'smaller_sooner'/'larger_later' or 0/1 values

### SLURM Issues

#### 1. Array Job Bounds Errors
**Symptom**: "Task ID X is out of bounds"
**Cause**: Using old hard-coded array bounds
**Solution**: Use dynamic submission:
```bash
./slurm/submit_glm_array_dynamic.sh
./slurm/submit_lss_array_dynamic.sh
```

#### 2. Missing Log Directories
**Symptom**: sbatch fails to create log files
**Cause**: logs/ directory doesn't exist
**Solution**: Run environment preparation:
```bash
./slurm/prepare_slurm_environment.sh
```

#### 3. Environment Variable Errors
**Symptom**: "ERROR: PROJECT_ROOT not set"
**Cause**: Required environment variables not exported
**Solution**: Set before submission:
```bash
export PROJECT_ROOT=/path/to/project
export BEHAVIORAL_DIR=/path/to/behavioral/data
```

#### 4. Wrong Working Directory
**Symptom**: Scripts can't find files
**Cause**: Submitting from wrong directory
**Solution**: All scripts now handle this automatically via PROJECT_ROOT

### Testing Issues

#### 1. Test Failures After Changes
**Symptom**: pytest fails on integration tests
**Cause**: Code changes broke existing functionality
**Solution**: Check test output, likely need to update test expectations

#### 2. Synthetic Data Issues
**Symptom**: "All GLM models failed" in tests
**Cause**: Synthetic data generation problem
**Solution**: Check synthetic_glm_dataset fixture, ensure variance in regressors

## Quick Diagnostic Commands

### Check Project Health
```bash
# Test suite
python -m pytest tests/ -v

# SLURM environment
./slurm/prepare_slurm_environment.sh

# GLM script validation (dry run)
python scripts/modeling/run_standard_glm.py config/project_config.yaml local sub-test --dry-run
```

### Debug Specific Issues
```bash
# Check subject count
find derivatives/behavioral -name "sub-*" -type d | wc -l

# Validate config
python -c "import yaml; yaml.safe_load(open('config/project_config.yaml'))"

# Check environment
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "BEHAVIORAL_DIR: $BEHAVIORAL_DIR"
```

## Emergency Recovery

### If Everything Breaks
1. **Check git status**: `git status` - see what changed
2. **Run tests**: `python -m pytest tests/` - identify failures
3. **Check environment**: `./slurm/prepare_slurm_environment.sh`
4. **Validate config**: Load config/project_config.yaml in Python
5. **Reset to known good state**: `git reset --hard HEAD~1` (if needed)

### If SLURM Jobs Fail
1. **Check logs**: `ls logs/` and examine error files
2. **Validate environment**: Ensure PROJECT_ROOT and BEHAVIORAL_DIR set
3. **Use dynamic submission**: Avoid manual sbatch with static bounds
4. **Check subject count**: Ensure behavioral data directory accessible

## Key Files to Check When Debugging
- `config/project_config.yaml` - Central configuration
- `logs/` - SLURM job logs and errors  
- `tests/test_modeling.py` - Integration test failures
- `scripts/modeling/run_standard_glm.py` - Main GLM logic
- `slurm/submit_*_dynamic.sh` - Dynamic job submission
