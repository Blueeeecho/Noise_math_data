# Plan: Local Testing and Remote Synchronization

This plan ensures the training pipeline is fully functional locally using a smaller model (`Qwen2.5-0.5B-Instruct`) before pushing a robust, environment-aware version to the remote server.

## 1. Environment Verification
- Check local installed packages (`trl`, `vllm`, `verl`) to ensure they match `requirements.txt`.
- Install or update dependencies if necessary (specifically `vllm==0.6.3.post1` and `trl==0.9.6`).

## 2. Script Optimization (Environment Awareness)
- Modify `scripts/run_backward_test.sh` to automatically detect the running environment (Local vs. Remote).
- **Remote Mode (Priority):**
  - Detect existence of `/export/home/asifali/Noise_math_data`.
  - Settings: `NUM_GPUS=4`, `MODEL_PATH=/export/home/asifali/HF_cache/Qwen2.5-1.5B-Instruct`.
- **Local Mode (Fallback):**
  - Settings: `NUM_GPUS=1` (or auto-detect), `MODEL_PATH=/home/wwq416/snap/wwq/model/Qwen/Qwen2.5-0.5B-Instruct`.
  - Adjust batch sizes if necessary for the smaller local GPU memory.

## 3. Local Execution & Validation
- Run `scripts/run_backward_test.sh` locally.
- Monitor each stage:
  1.  **Data Preparation:** Ensure `prepare_backward_data.py` runs without path errors.
  2.  **SFT:** Ensure `sft_train.py` runs with the 0.5B model and correct arguments (`max_seq_length`).
  3.  **RL:** Ensure `train_verl.py` initializes vLLM correctly (checking the `LoRALRUCache` fix) and runs PPO/GRPO.
  4.  **Eval:** Ensure `eval_model.py` accepts the new memory argument and produces results.

## 4. Final Polish & Push
- If local tests pass, the script is proven to be logically correct and environment-robust.
- Commit the changes (including the "Environment Awareness" logic).
- Push to `origin master:main`.
