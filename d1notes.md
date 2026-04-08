D1

Q1 : Repo structure 
d1/
├── SFT/
│   ├── sft_train.py       ← SFT training entry
│   └── sft_trainer.py     ← 3 classes：dLLMTrainer, dLLMDataCollator, dLLMSFTDataset
│
├── diffu-grpo/
│   ├── diffu_grpo_trainer.py  ← RL core
│   └── run.sh                 ← running script


Q2: New Estimator + Small Patch 
# diffu_grpo_trainer.py 
from trl.trainer.grpo_trainer import GRPOTrainer
class DiffuGRPOTrainer(GRPOTrainer):  


`DiffuGRPOTrainer` directly inherits TRL's `GRPOTrainer`. The only overridden methods are:

| `_get_per_token_logps()` | Core estimator , main contribution |
| `forward_process()` | Random prompt masking for diffusion |
| `generate()` | **REPLACED** | LLaDA-style iterative denoising |
| `compute_loss()` | **PATCHED** | Plug new log-probs into GRPO loss |
| `_generate_and_score_completions()` | **PATCHED** | Connect diffusion generation to RL loop |
| `_prepare_inputs()` | **PATCHED** | Control outer/inner loop logic |

Everything else — advantage computation, clipping, KL penalty, optimizer, gradient accumulation, checkpointing, logging — is **directly inherited from TRL's AR GRPO trainer**.

Q3: The trainer is LLaDA-specific 
**1. LLaDA's mask token ID (126336)
**2. Fixed-length generation** 
**3. Semi-autoregressive block decoding 

