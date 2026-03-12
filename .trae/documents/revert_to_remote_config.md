# Plan: Revert to Remote-Optimized Configuration

The user prefers a strictly remote-optimized script without automatic environment detection logic. This plan details the steps to revert `scripts/run_backward_test.sh` to its explicit, remote-specific configuration while retaining the critical fixes for SFT arguments, RL config compatibility, and eval memory management.

## 1. Revert Shell Script Logic (`scripts/run_backward_test.sh`)
- **Remove:** The `if [ -d "/export/home/asifali/Noise_math_data" ]; then ... else ... fi` logic block.
- **Restore:** Hardcoded remote paths and settings:
  - `BASE_DIR="/export/home/asifali/Noise_math_data"`
  - `MODEL_PATH="/export/home/asifali/HF_cache/Qwen2.5-1.5B-Instruct"`
  - `NUM_GPUS=4`
  - Hyperparameters:
    - `SFT_BATCH_SIZE=8`
    - `RL_BATCH_SIZE=16`
    - `PPO_MINI_BATCH=8`
    - `ROLLOUT_GPU_UTIL=0.8`
    - `EVAL_GPU_UTIL=0.5`
- **Retain Fixes:** Ensure the script still uses the corrected argument names (e.g., `--max_seq_length`) and configuration syntax (e.g., `++reward...`).

## 2. Review & Verify
- Check `scripts/run_backward_test.sh` to confirm no local fallback logic remains.
- Verify `scripts/prepare_backward_data.py` retains the relative path logic (as it is robust and harmless) or revert if requested (plan assumes retaining robustness unless specified, but user asked for "original remote adaptation", so we will revert to absolute paths for consistency if that's what "original" implies. *Correction*: User said "keep original remote adapted code", so I will revert `prepare_backward_data.py` to use the absolute path `/export/home/...` as well to be safe and consistent).

## 3. Push to Remote
- Commit the changes with a message indicating a revert to explicit remote configuration.
- Push to `origin master:main`.
