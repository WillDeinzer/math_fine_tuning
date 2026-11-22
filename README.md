# math_fine_tuning

For SFT Colab training, this should be run at the start of any session:


# 1. Mount Drive
from google.colab import drive
drive.mount("/content/drive")

# 2. Clone/update repository
import os

REPO = "/content/repo"
URL = "https://github.com/WillDeinzer/math_fine_tuning.git"

if not os.path.exists(REPO):
    !git clone {URL} {REPO}

%cd {REPO}
!git pull

# 3. Install dependencies BEFORE importing them
!pip install -q -r requirements_colab.txt

# 4. Verify installed versions
!pip show wandb trl transformers peft

# 5. Only NOW import W&B
import wandb
wandb.login()

# 6. Other imports
import torch
import transformers
import trl
import peft

print("W&B:", wandb.__version__)
print("TRL:", trl.__version__)
print("Transformers:", transformers.__version__)
print("PEFT:", peft.__version__)

!nvidia-smi

Then, can run sft as follows:

!python train_sft.py --output_dir /content/drive/MyDrive/runs/sft_v1

## Environments

SFT and GRPO use separate environments, because GRPO needs a newer trl (plus vLLM) than SFT:

| Environment | Requirements file | Key versions |
|---|---|---|
| SFT | `requirements.txt` | trl 0.19.1, transformers 5.23.0, torch 2.15.0 |
| GRPO | `requirements_grpo.txt` | trl 1.14.2, vllm 0.30.0, transformers 5.24.0, torch 2.13.0 |

Both use Python 3.12. To create them with conda:

```bash
conda create -y -p <env_dir>/math_sft python=3.12
<env_dir>/math_sft/bin/pip install -r requirements.txt

conda create -y -p <env_dir>/math_grpo python=3.12
<env_dir>/math_grpo/bin/pip install -r requirements_grpo.txt
```

The torch builds in both target CUDA 13, so they need a GPU node with a CUDA 13 driver and bf16 support (for example A100, L40S, H100).

## Math evaluation

`eval_math.py` scores a model on GSM8K test and MATH-500 (greedy decoding, last `\boxed{}` answer checked with
math-verify). Needs `math-verify` and `vllm` (or add `--backend hf`); on a T4 add `--dtype float16`.

```
!python eval_math.py --output_dir /content/drive/MyDrive/results/math/base_zero_shot
!python eval_math.py --shots 4 --output_dir /content/drive/MyDrive/results/math/base_few_shot
!python eval_math.py --checkpoint <run_dir>/checkpoint-N --output_dir /content/drive/MyDrive/results/math/<name>
```

Writes `generations_<dataset>.jsonl` (one row per problem) and `metrics.json` (accuracy, lenient accuracy,
no-`\boxed{}` rate, truncation rate, and MATH-500 accuracy by level and subject). Use `--limit 20` for a quick check.
