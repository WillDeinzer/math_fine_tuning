# math_fine_tuning

For SFT Colab training, this should be run at the start of any session:

from google.colab import drive
drive.mount("/content/drive")

import os
REPO = "/content/repo"
URL = "https://github.com/WillDeinzer/math_fine_tuning.git"

if not os.path.exists(REPO):
    !git clone {URL} {REPO}
%cd {REPO}
!git pull
!pip install -q -r requirements_colab.txt
!nvidia-smi
!pip list 2>/dev/null | grep -i -E "^(torch|transformers|trl|peft|datasets|pandas|numpy|accelerate) "

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