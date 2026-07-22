# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import OBS_IMAGES

from ..rtc.configuration_rtc import RTCConfig


@PreTrainedConfig.register_subclass("smolvla")
@dataclass
class SmolVLAConfig(PreTrainedConfig):
    """SmolVLA 策略配置。

    这里同时包含数据输入输出、图像/文本预处理、动作生成、微调范围、优化器、模型结构和
    真机推理优化等配置。通过 ``--policy.path=lerobot/smolvla_base`` 微调官方预训练模型时，
    建议只调整训练范围、学习率、动作步数等不改变网络形状的参数；层数、专家宽度和注意力结构
    必须与 checkpoint 保持一致，否则预训练权重可能因名称或张量形状不匹配而无法加载。
    """

    # ===== 输入与动作序列 =====
    # 每次策略调用使用的观测时间步数。SmolVLA 当前只使用最新一帧观测。
    n_obs_steps: int = 1

    # 模型每次预测的完整动作块长度，也是训练样本中需要读取的未来动作数。
    chunk_size: int = 50

    # 推理时从动作块中连续执行/缓存的动作数；必须小于或等于 chunk_size。
    # 数值越小，策略重新观察并规划得越频繁，但推理开销也越高。
    n_action_steps: int = 50

    # 不同模态的归一化方式：图像由视觉处理器负责，因此保持原值；状态和动作使用数据集均值/方差。
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # ===== 状态与动作维度 =====
    # 状态/动作向量会在进入网络前补零到固定维度。真实维度不能超过这里的上限。
    # 修改这两个值会改变投影层形状，加载 smolvla_base 时应保持默认值 32。
    max_state_dim: int = 32
    max_action_dim: int = 32

    # ===== 图像输入 =====
    # 将每路图像按比例缩放并补边到目标高宽，不会直接拉伸破坏宽高比。
    # 相机数量由 input_features 决定，不受这个字段限制；四相机数据也可以使用。
    resize_imgs_with_padding: tuple[int, int] = (512, 512)

    # 当实际数据集相机少于策略配置预期时，额外创建多少路全黑占位图像。
    # 它用于“缺少相机”的兼容场景，不是用来启用数据集中已有的额外相机。
    empty_cameras: int = 0

    # 将标准 ALOHA 的关节/夹爪表示转换到 PI 内部运行时使用的表示。
    # 只有数据明确采用标准 ALOHA 顺序和单位时才能开启；AlohaMini 数据不要仅凭名称贸然启用。
    adapt_to_pi_aloha: bool = False

    # 将关节动作转换为相对当前状态的增量，夹爪仍使用绝对值。
    # 当前 LeRobot 的 SmolVLA 尚未移植该功能，设为 True 会在 __post_init__ 中直接报错。
    use_delta_joint_actions_aloha: bool = False

    # ===== 语言输入 =====
    # 任务文本分词后的最大 token 数；过长文本会被截断。
    tokenizer_max_length: int = 48

    # ===== Flow Matching 动作生成 =====
    # 从噪声迭代生成动作所使用的去噪步数。减少可加快推理但可能降低动作质量。
    num_steps: int = 10

    # 推理时是否缓存 VLM 前缀的注意力 Key/Value，通常应保持开启以减少重复计算。
    use_cache: bool = True

    # ===== 微调范围 =====
    # 冻结视觉编码器可显著减少显存和训练量，适合单张 24 GB GPU 的首轮微调。
    freeze_vision_encoder: bool = True

    # True：冻结整个 VLM，只训练动作专家和策略投影层；False：同时微调可训练的 VLM 参数。
    train_expert_only: bool = True

    # 是否训练机器人状态到 VLM 隐空间的投影层。机器人状态定义发生变化时通常应保持 True。
    train_state_proj: bool = True

    # ===== AdamW 优化器预设 =====
    # use_policy_training_preset=True（训练命令默认值）时，这些字段会生成实际 optimizer 配置。
    # 因此覆盖学习率应使用 --policy.optimizer_lr，而不是只写 --optimizer.lr。
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10

    # ===== 余弦学习率调度 =====
    # 前 warmup_steps 步线性升温，随后在 decay_steps 内从 optimizer_lr 衰减到 decay_lr。
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    # ===== VLM 骨干与初始化 =====
    # 提供图像和语言表征的视觉语言骨干。更换模型通常意味着不能直接复用 smolvla_base 权重。
    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

    # 是否先加载上述 VLM 的预训练权重。从头创建 SmolVLA 时默认 False；官方 smolvla_base
    # checkpoint 自带的 config 会将其设为 True。正常微调官方模型时不要手动改回 False。
    load_vlm_weights: bool = False

    # 是否在图像特征周围加入专用边界 token。它会改变输入序列格式，应与预训练配置保持一致。
    add_image_special_tokens: bool = False

    # ===== VLM 与动作专家的注意力结构 =====
    # cross_attn 让动作专家通过交叉注意力读取 VLM 前缀。改变模式会改变部分投影层结构。
    attention_mode: str = "cross_attn"

    # 将图像、状态和语言组成的前缀补齐到固定长度；负数表示不强制，0 表示不额外补齐。
    prefix_length: int = -1

    # longest：补齐到当前 batch 最长文本；max_length：统一补齐到 tokenizer_max_length。
    pad_language_to: str = "longest"

    # 动作专家层数。小于等于 0 表示与 VLM 层数相同；若显式指定，必须能整除 num_vlm_layers。
    # smolvla_base 实际为 16 层，微调 checkpoint 时改变此值会造成权重不匹配。
    num_expert_layers: int = -1

    # 使用 VLM 的前多少个文本 Transformer 层。smolvla_base 使用前 16 层。
    # 这是结构裁剪参数，不是普通的“冻结多少层”参数；预训练微调时应保持 16。
    num_vlm_layers: int = 16

    # cross-attention 模式下，每隔多少个专家层保留一次专家内部 self-attention。
    # smolvla_base 使用 2，改变它可能改变注意力投影形状和 checkpoint 兼容性。
    self_attn_every_n_layers: int = 2

    # 动作专家隐藏维度相对于 VLM 隐藏维度的倍率。改变它会直接改变专家权重张量形状。
    expert_width_multiplier: float = 0.75

    # Flow Matching 时间步正弦/余弦编码所覆盖的周期范围，控制时间条件对不同频率的敏感度。
    min_period: float = 4e-3
    max_period: float = 4.0

    # ===== 真机推理优化 =====
    # Real-Time Chunking 配置。None 表示禁用；启用后可约束新旧动作块衔接，减少慢模型造成的顿挫。
    rtc_config: RTCConfig | None = None

    # 是否使用 torch.compile 编译训练和采样函数。首次运行编译较慢，且会增加调试难度。
    compile_model: bool = False

    # 传给 torch.compile 的模式，例如 default、reduce-overhead 或 max-autotune。
    compile_mode: str = "max-autotune"

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.use_delta_joint_actions_aloha:
            raise NotImplementedError(
                "`use_delta_joint_actions_aloha` is used by smolvla for aloha real models. It is not ported yet in LeRobot."
            )

    def validate_features(self) -> None:
        # 为空相机生成规范的视觉特征键；真正的相机特征仍来自 policy.input_features。
        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            )
            self.input_features[key] = empty_camera

    def get_optimizer_preset(self) -> AdamWConfig:
        # 将 policy.* 优化器字段转换为训练器最终使用的 AdamWConfig。
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        # 学习率从 optimizer_lr 预热，再按余弦曲线衰减到 scheduler_decay_lr。
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list:
        # 只读取当前观测（相对索引 0），不读取历史帧。
        return [0]

    @property
    def action_delta_indices(self) -> list:
        # 数据集需要提供从当前时刻开始的完整未来动作块。
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
