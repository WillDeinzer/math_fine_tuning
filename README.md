# math_fine_tuning

For SFT Colab training, this should be run at the start of any session:

from google.colab import drive
drive.mount("/content/drive")

import os
REPO = "/content/repo"
URL = "https://github.com/<you>/<repo>.git"

if not os.path.exists(REPO):
    !git clone {URL} {REPO}
%cd {REPO}
!git pull
!pip install -q -r requirements-colab.txt
!nvidia-smi
!pip list 2>/dev/null | grep -i -E "^(torch|transformers|trl|peft|datasets|pandas|numpy|accelerate) "

Then, can run sft as follows:

!python train_sft.py --output_dir /content/drive/MyDrive/runs/sft_v1