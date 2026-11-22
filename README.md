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