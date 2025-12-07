
# routing_config.py

from dataclasses import dataclass
from typing import Literal, List, Tuple
import os

import torch

from squad_dataset import SQuADDataset
from llm_wrapper import HFLLMWrapper
from qa_span_wrapper import SquadQAModelWrapper
from qa_env import QARoutingEnv

# qa: span extractor (SquadQAModelWrapper)
# lm: generate (HFLLMWrapper)
ModelType = Literal["qa", "lm"]

# 🔧 여기 이름/타입만 바꿔가면서 실험하면 됨
#CHEAP_MODEL_NAME = "deepset/tinyroberta-squad2"
CHEAP_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
CHEAP_MODEL_TYPE: ModelType = "lm"

STRONG_MODEL_NAME = "deepset/roberta-large-squad2"
STRONG_MODEL_TYPE: ModelType = "qa"

# 실험 폴더 이름: qa+qa, lm+qa, lm+lm 등 자동 구성
DIR = f"{CHEAP_MODEL_TYPE}+{STRONG_MODEL_TYPE}"
CHECKPOINT_DIR = f"{DIR}/checkpoint"
RESULT_DIR = f"{DIR}/results"


# ---- 개별 모델 설정 ----
@dataclass
class ModelRoutingConfig:
    model_name: str          # HF model name
    model_type: ModelType    # "qa" or "lm"
    device: str | None = None
    max_new_tokens: int = 64     # lm일 때만 사용
    temperature: float = 0.0     # lm일 때만 사용


# ---- 전체 라우팅 + 로깅/체크포인트 설정 ----
@dataclass
class RoutingConfig:
    cheap: ModelRoutingConfig
    strong: ModelRoutingConfig

    # Env reward 관련
    max_retry: int = 2
    token_budget: int = 512
    w_token: float = 0.0    # QA-only면 0, LM 쓸 때 >0로 설정
    w_retry: float = 0.2
    w_strong: float = 0.7

    # 체크포인트 / 결과 저장 관련
    checkpoint_dir: str = "trained_model"
    result_dir: str = "results"
    checkpoint_episodes: List[int] | None = None


def default_routing_config() -> RoutingConfig:
    """
    👉 여기만 수정하면 됨.
    cheap / strong / reward weight / 폴더 이름까지 전부 한 군데에서.
    """
    default_device = "cuda" if torch.cuda.is_available() else "cpu"

    # ===== cheap 모델 설정 =====
    if CHEAP_MODEL_TYPE == "qa":
        cheap_cfg = ModelRoutingConfig(
            model_name=CHEAP_MODEL_NAME,
            model_type=CHEAP_MODEL_TYPE,
            device=default_device,
        )
    else:  # "lm"
        cheap_cfg = ModelRoutingConfig(
            model_name=CHEAP_MODEL_NAME,
            model_type=CHEAP_MODEL_TYPE,
            device=default_device,
            max_new_tokens=64,
            temperature=0.0,
        )

    # ===== strong 모델 설정 =====
    if STRONG_MODEL_TYPE == "qa":
        strong_cfg = ModelRoutingConfig(
            model_name=STRONG_MODEL_NAME,
            model_type=STRONG_MODEL_TYPE,
            device=default_device,
        )
    else:  # "lm"
        strong_cfg = ModelRoutingConfig(
            model_name=STRONG_MODEL_NAME,
            model_type=STRONG_MODEL_TYPE,
            device=default_device,
            max_new_tokens=64,
            temperature=0.0,
        )

    # ===== env + 로깅/체크포인트 설정 =====
    return RoutingConfig(
        cheap=cheap_cfg,
        strong=strong_cfg,
        max_retry=2,
        token_budget=512,
        w_token=0.01,            # QA-only면 0
        w_retry=0.2,
        w_strong=0.7,
        checkpoint_dir=CHECKPOINT_DIR,
        result_dir=RESULT_DIR,  # 결과 저장 폴더 이름
        checkpoint_episodes=None,
    )


# ---- 내부: 모델 config -> 실제 wrapper 생성 ----
def _build_model_from_cfg(cfg: ModelRoutingConfig):
    """
    cfg.model_type에 따라 HFLLMWrapper 또는 SquadQAModelWrapper 생성.
    반환: (wrapper 객체, kind 문자열, 실제 device)
    """
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.model_type == "lm":
        model = HFLLMWrapper(
            model_name=cfg.model_name,
            device=device,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
        )
    elif cfg.model_type == "qa":
        model = SquadQAModelWrapper(
            model_name=cfg.model_name,
            device=device,
        )
    else:
        raise ValueError(f"Unknown model_type: {cfg.model_type} (expected 'qa' or 'lm')")

    return model, cfg.model_type, device


# ---- 외부에서 쓰는 엔트리포인트 ----
def build_env_and_paths(
    split: str = "train",
    config: RoutingConfig | None = None,
) -> Tuple[QARoutingEnv, str, str, List[int] | None]:
    """
    - ✅ 항상 SQuAD(= QA dataset)만 사용
    - cheap / strong wrapper 생성 (qa 또는 lm)
    - QARoutingEnv 생성
    - checkpoint_dir / result_dir / checkpoint_episodes 함께 리턴
    """
    if config is None:
        config = default_routing_config()

    # 1) Dataset: ✅ LM/QA 상관없이 무조건 SQuAD만 사용
    dataset = SQuADDataset(split=split)
    print(f"[Dataset] QA (SQuAD) for split={split}: {len(dataset)} samples")

    # 2) cheap/strong 모델 생성
    cheap_model, cheap_kind, cheap_device = _build_model_from_cfg(config.cheap)
    strong_model, strong_kind, strong_device = _build_model_from_cfg(config.strong)

    print(f"Cheap model  : {config.cheap.model_name} (type={cheap_kind}, device={cheap_device})")
    print(f"Strong model : {config.strong.model_name} (type={strong_kind}, device={strong_device})")

    # 3) Env 생성
    env = QARoutingEnv(
        dataset=dataset,
        cheap_model=cheap_model,
        cheap_kind=cheap_kind,         # "qa" or "lm"
        strong_model=strong_model,
        strong_kind=strong_kind,       # "qa" or "lm"
        max_retry=config.max_retry,
        token_budget=config.token_budget,
        w_token=config.w_token,
        w_retry=config.w_retry,
        w_strong=config.w_strong,
    )

    # 4) 폴더 생성
    os.makedirs(DIR,exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.result_dir, exist_ok=True)

    return env, config.checkpoint_dir, config.result_dir, config.checkpoint_episodes

