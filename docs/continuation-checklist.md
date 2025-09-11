# Continuation Checklist for Future Sessions

## When Starting a New Session

### 1. Project State Verification
- [ ] Read `docs/project-state-summary.md` for current architecture
- [ ] Check git log for recent changes: `git log --oneline -10`
- [ ] Verify working directory: `/Users/joshuabuckholtz/Code/DecodingDD`
- [ ] Check environment: User does not use requirements.txt [[memory:8254756]]

### 2. Core Architecture Understanding
- [ ] **Multi-Model GLM**: Separate models for each regressor (choice, SVchosen, SVunchosen, SVdiff, large_amount)
- [ ] **SLURM Infrastructure**: Dynamic submission via wrapper scripts, no hard-coded bounds
- [ ] **Configuration**: Central config in `config/project_config.yaml` with `model_specifications`
- [ ] **Testing**: Integration tests in `tests/test_modeling.py` with synthetic data

### 3. Key Implementation Details
- [ ] GLM script handles multicollinearity via separate models
- [ ] VIF calculation per-run using sklearn LinearRegression  
- [ ] Onset validation with multi-criteria detection (run vs session relative)
- [ ] Confound harmonization with zero-filling for missing files
- [ ] Robust choice recoding for both string and numeric values

### 4. SLURM Best Practices
- [ ] Use dynamic submission: `./slurm/submit_glm_array_dynamic.sh`
- [ ] Never hard-code array bounds in batch scripts
- [ ] Always set PROJECT_ROOT and BEHAVIORAL_DIR environment variables
- [ ] Run `./slurm/prepare_slurm_environment.sh` for setup
- [ ] Check logs/ directory exists before submission

### 5. Quality Assurance Workflow
- [ ] Run tests after any changes: `python -m pytest tests/ -v`
- [ ] Check for linting errors on modified files
- [ ] Validate configuration: Load `config/project_config.yaml` in Python
- [ ] Use git for version control and change tracking

## Common Tasks and Patterns

### Adding New Regressors
1. Update `config/project_config.yaml` model_specifications
2. Add to parametric_modulators list if needed
3. Update test synthetic data generation
4. Update test assertions for new model output
5. Run integration tests to verify

### SLURM Script Modifications
1. Always add `set -e` for fail-fast behavior
2. Use `${VAR:?}` for required environment variables
3. Include directory existence checks: `[[ -d "$DIR" ]]`
4. Add `mkdir -p logs` before any sbatch commands
5. Use PROJECT_ROOT instead of submission directory

### Debugging Workflow
1. Check `docs/troubleshooting-guide.md` for common issues
2. Examine logs/ directory for SLURM job errors
3. Run diagnostic commands from troubleshooting guide
4. Use git to track changes and identify regression points
5. Test fixes with integration test suite

## Critical Files to Monitor
- `scripts/modeling/run_standard_glm.py` - Main GLM logic
- `config/project_config.yaml` - Central configuration  
- `tests/test_modeling.py` - Integration tests
- `slurm/submit_*_dynamic.sh` - Dynamic job submission
- `slurm/prepare_slurm_environment.sh` - Environment setup

## Red Flags to Watch For
- ❌ Hard-coded array bounds in SLURM scripts
- ❌ Using SLURM_SUBMIT_DIR instead of PROJECT_ROOT  
- ❌ Missing environment variable validation
- ❌ Static submission without dynamic calculation
- ❌ Test failures after changes
- ❌ VIF values > 10 in single-model approach (use multi-model)
- ❌ Session-relative onsets without explicit run_start_times

## Success Indicators
- ✅ All tests pass: `python -m pytest tests/`
- ✅ SLURM jobs submit without bounds errors
- ✅ GLM produces model-specific output directories
- ✅ VIF warnings present but models run successfully
- ✅ Logs/ directory created automatically
- ✅ Environment preparation script runs cleanly

## Emergency Recovery Commands
```bash
# Check project health
python -m pytest tests/ -v
./slurm/prepare_slurm_environment.sh

# Reset to known good state (if needed)
git status
git log --oneline -5
git reset --hard HEAD~1  # CAREFUL: Only if absolutely necessary

# Validate core functionality
python -c "import yaml; print('Config valid:', bool(yaml.safe_load(open('config/project_config.yaml'))))"
```

This checklist ensures smooth transitions between sessions and maintains the hard-earned production quality of the DecodingDD project.
