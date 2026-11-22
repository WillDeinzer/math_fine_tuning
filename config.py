import os
from typing import ClassVar, Optional
from pydantic import BaseModel

os.environ.setdefault("WANDB_PROJECT", "cs4650-math-finetuning")

MODEL_NAME = "Qwen/Qwen2.5-1.5B"

SEED = 42

TRAIN_PATH = "datasets/train.jsonl"
VAL_PATH = "datasets/val.jsonl"
TEST_PATHS = {
    "gsm8k": "datasets/gsm8k_test.jsonl",
    "math500": "datasets/math_500.jsonl",
}

SFT_EPOCH_ADAPTER_DIR = "/content/drive/MyDrive/math_finetuning/epoch_adapters"


def check_filled(config, allowed_none=()):
    missing = [k for k, v in config.model_dump().items() if v is None and k not in allowed_none]
    if missing:
        raise ValueError(f"{type(config).__name__}: fill in these values in config.py: {missing}")
    return config.model_dump()

class LoraHParams(BaseModel):
    r: Optional[int] = None
    lora_alpha: Optional[int] = None
    lora_dropout: Optional[float] = None
    target_modules: list[str] = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    def peft_kwargs(self):
        return check_filled(self)

class SFTLoraConfig(LoraHParams):
    r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.1


class GRPOLoraConfig(LoraHParams):
    r: Optional[int] = None  # FILL IN
    lora_alpha: Optional[int] = None  # FILL IN
    lora_dropout: Optional[float] = None  # FILL IN


class TrainingArgumentsConfig(BaseModel):
    num_train_epochs: Optional[int] = None
    learning_rate: Optional[float] = None
    optim: Optional[str] = None
    weight_decay: Optional[float] = None
    max_grad_norm: Optional[float] = None
    lr_scheduler_type: Optional[str] = None
    warmup_steps: Optional[float] = None  # a value < 1 is a fraction of total steps
    per_device_train_batch_size: Optional[int] = None
    per_device_eval_batch_size: Optional[int] = None
    gradient_accumulation_steps: Optional[int] = None
    gradient_checkpointing: Optional[bool] = None
    fp16: Optional[bool] = None
    bf16: Optional[bool] = None
    logging_steps: Optional[int] = None

    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"
    eval_on_start: bool = True
    save_only_model: bool = True
    gradient_checkpointing_kwargs: dict = {"use_reentrant": False}
    report_to: str = "wandb"
    seed: int = SEED

    ALLOWED_NONE: ClassVar[tuple] = ()

    def trainer_kwargs(self):
        return check_filled(self, self.ALLOWED_NONE)


class SFTTrainingConfig(TrainingArgumentsConfig):
    num_train_epochs: int = 3
    learning_rate: float = 1e-4
    optim: str = 'adamw_torch_fused'
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    warmup_steps: float = 0.03
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    gradient_checkpointing: bool = True
    fp16: bool = True
    bf16: bool = False
    logging_steps: int = 25
    run_name: str = "sft_lora_r64_v1"

    completion_only_loss: bool = True
    max_seq_length: int = 1024
    train_sampling_strategy: str = "group_by_length"
    packing: bool = False


class GRPOTrainingConfig(TrainingArgumentsConfig):
    fp16: bool = False
    bf16: bool = True

    # GRPO-only (all FILL IN)
    num_generations: Optional[int] = None  # completions sampled per prompt (the "group")
    max_completion_length: Optional[int] = None  # max tokens per generated completion
    temperature: Optional[float] = None  # sampling temperature for generation
    beta: Optional[float] = None  # KL penalty toward the starting model (0 = no KL term)
    loss_type: Optional[str] = None  # which GRPO loss variant, e.g. "grpo", "dapo", "dr_grpo"
    epsilon: Optional[float] = None  # clipping range for the policy ratio
    num_iterations: Optional[int] = None  # optimizer passes per batch of generations
    scale_rewards: Optional[str] = None  # how advantages are normalized: "group", "batch", or "none"
    reward_weights: Optional[list[float]] = None  # one weight per reward function; None = all equal
    use_vllm: Optional[bool] = None  # generate with vLLM instead of HF generate
    vllm_mode: Optional[str] = None  # "colocate" = vLLM shares the training GPU
    vllm_gpu_memory_utilization: Optional[float] = None  # fraction of GPU memory vLLM may take
    log_completions: Optional[bool] = None  # print sample completions during training

    ALLOWED_NONE: ClassVar[tuple] = ("reward_weights",)
