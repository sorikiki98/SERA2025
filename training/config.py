from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Union

from constants import VALIDATION_PROMPTS
from utils.types import PESigmas


@dataclass
class LogConfig:
    """ Parameters for logging and saving """
    # Name of experiment. This will be the name of the output folder
    exp_name: str
    # The output directory where the model predictions and checkpoints will be written
    exp_dir: Path = Path("./outputs")
    # Save interval
    save_steps: int = 250
    # [TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to
    # `output_dir/runs/**CURRENT_DATETIME_HOSTNAME`
    logging_dir: Path = Path("logs")
    # The integration to report the results to. Supported platforms are "tensorboard" '
    # (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
    report_to: str = "tensorboard"
    # Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator`
    checkpoints_total_limit: Optional[int] = None


@dataclass
class DataConfig:
    """ Parameters for data """
    dataset: str
    # A folder containing the training data
    train_data_dir: str
    # A token to use as a placeholder for the concept
    placeholder_img_token: str = "<img_unknown>"
    # Super category token to use for normalizing the mapper output
    super_category_img_token: Optional[str] = "photo"
    # Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process
    dataloader_num_workers: int = 8
    # How many times to repeat the training data
    repeats: int = 1
    # The resolution for input images, all the images in the train/validation dataset will be resized to this resolution
    resolution: int = 224
    # Whether to center crop images before resizing to resolution
    center_crop: bool = False


@dataclass
class ModelConfig:
    """ Parameters for defining all models """
    # Path to pretrained model or model identifier from huggingface.co/models
    pretrained_diffusion_model_name_or_path: str = "CompVis/stable-diffusion-v1-4"
    pretrained_image_model_name_or_path: str = "openai/clip-vit-large-patch14"
    # dimension of word embedding. Is 768 if sd1 and 1024 if sd2
    word_embedding_dim: int = 768
    # dimension of hidden layers in the MLP
    arch_mlp_hidden_dims: int = 128
    # Whether to use our Nested Dropout technique
    use_nested_dropout: bool = True
    # Probability to apply nested dropout during training
    nested_dropout_prob: float = 0.5
    # Whether to normalize the norm of the mapper's output vector
    normalize_img_mapper_output: bool = True
    # Target norm for the mapper's output vector
    target_norm_img: float = None
    # Pos encoding for (t,l) conditioning. 0 - scale to [-1,1], 1 - pos encoding
    # proposed by Neti. 2 - standard Fourier feature pos encoding.
    use_positional_encoding_img: int = 0
    # Sigmas used for computing positional encoding
    pe_sigmas: Dict[str, float] = field(default_factory=lambda: {'sigma_t': 0.03, 'sigma_l': 2.0})
    # Number of time anchors for computing our positional encodings
    num_pe_time_anchors: int = 10
    # Whether to output the textual bypass vector
    output_bypass_img: bool = True
    # Revision of pretrained model identifier from huggingface.co/models
    revision: Optional[str] = None
    # Whether training should be resumed from a previous checkpoint.
    mapper_checkpoint_path: Optional[Path] = None
    # configuration for view neti-mappers. Ignored by object neti-mappers
    arch_img_disable_tl: bool = True
    # Run as original-TI. A single vector is learned per placeholder token.
    original_ti: bool = False
    # free movement in the bypass space
    bypass_unconstrained_img: bool = False
    # size of alpha hyperparameter for the output bypass
    output_bypass_alpha_img: float = 0.2

    def __post_init__(self):
        if self.pe_sigmas is not None:
            assert len(self.pe_sigmas) == 2, "Should provide exactly two sigma values: one for two and one for layers!"
            self.pe_sigmas = PESigmas(sigma_t=self.pe_sigmas['sigma_t'], sigma_l=self.pe_sigmas['sigma_l'])


@dataclass
class EvalConfig:
    """ Parameters for validation """
    # Prompts for validation (only for learnable_mode=0)
    validation_prompts: List[str] = field(
        default_factory=lambda: VALIDATION_PROMPTS)
    # Number of images that should be generated during validation with `validation_prompt`
    num_validation_images: int = 3
    # Seeds to use for generating the validation images
    validation_seeds: Optional[List[int]] = field(
        default_factory=lambda: [0, 1, 2])
    # Run validation every X steps.
    validation_steps: int = 10
    # Number of denoising steps
    num_denoising_steps: int = 50
    # for learnable_mode==3, which of the `placholder_object_tokens` to include in validation
    eval_placeholder_img_tokens: List[str] = None
    image_paths: List[str] = None

    def __post_init__(self):
        if self.validation_seeds is None:
            self.validation_seeds = list(range(self.num_validation_images))
        assert len(self.validation_seeds) == self.num_validation_images, \
            "Length of validation_seeds should equal num_validation_images"


@dataclass
class OptimConfig:
    """ Parameters for the optimization process """
    # Total number of training steps to perform.
    max_train_steps: Optional[int] = 1000
    # Learning rate
    learning_rate: float = 1e-3
    # Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size
    scale_lr: bool = True
    # Batch size (per device) for the training dataloader
    train_batch_size: int = 2
    # Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass
    gradient_checkpointing: bool = False
    # Number of updates steps to accumulate before performing a backward/update pass
    gradient_accumulation_steps: int = 4
    # A seed for reproducible training
    seed: Optional[int] = None
    # The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",
    # "constant", "constant_with_warmup"]
    lr_scheduler: str = "constant"
    # Number of steps for the warmup in the lr scheduler
    lr_warmup_steps: int = 0
    # The beta1 parameter for the Adam optimizer
    adam_beta1: float = 0.9
    # The beta2 parameter for the Adam optimizer
    adam_beta2: float = 0.999
    # Weight decay to use
    adam_weight_decay: float = 1e-2
    # Epsilon value for the Adam optimizer
    adam_epsilon: float = 1e-08
    # Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10.
    # and an Nvidia Ampere GPU.
    mixed_precision: str = "no"
    # Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see
    # https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    allow_tf32: bool = False


@dataclass
class RunConfig:
    """ The main configuration for the coach trainer """
    log: LogConfig = field(default_factory=LogConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
