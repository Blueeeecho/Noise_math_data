import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf

# Monkey Patch for Verl
# We need to inject our custom reward function and data loader if necessary
# But Verl is designed to be configurable via Hydra. 
# We just need to make sure our custom code is importable.

# Add the current directory to sys.path so we can import reward_fn
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch

# Patch for local environment compatibility (torch 2.4 vs verl)
try:
    import torch.distributed.tensor
    if not hasattr(torch.distributed.tensor, 'DTensor'):
        import torch.distributed._tensor
        torch.distributed.tensor.DTensor = torch.distributed._tensor.DTensor
except ImportError:
    pass

from verl.trainer.main_ppo import main as verl_main_ppo
from reward_fn import compute_reward

# Inject custom reward function
# Verl allows specifying a custom reward function path in config, 
# but injecting it here ensures it's available.
# Actually, Verl's `reward_score.py` loads functions dynamically. 
# We just need to point to it in the config: `reward.custom_reward_function.path`

# However, we can also override the default compute_score if we want strict control
# or if we are using an older version of Verl that doesn't support dynamic loading fully.
# Let's try to be non-intrusive first and rely on config.
# But for safety in this refactor, we can register it.

def main():
    # Set environment variables if needed
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # We will just call the standard Verl entry point, but wrapped 
    # to ensure our environment is set up.
    # Note: Verl uses Hydra, so we need to pass arguments.
    
    # If we want to intercept the config, we can use the @hydra.main decorator ourselves
    # and then call the internal verl function.
    
    # For now, let's just let Hydra handle it, but we provide this script 
    # as the entry point for `python train_verl.py ...`
    
    # We need to make sure `compute_reward` is available to Verl if it tries to import it.
    # It is available in `reward_fn.py` next to this script.
    
    verl_main_ppo()

if __name__ == "__main__":
    main()
