# squad_dataset.py

from typing import Dict, Any, List
import random
from datasets import load_dataset


class SQuADDataset:
    """
    HuggingFace 'squad' 데이터셋 래퍼.

    각 샘플을 다음 형식으로 평탄화:
      { "id": str, "question": str, "context": str, "answers": [ "정답1", ... ] }
    """

    def __init__(self, split: str = "train"):
        ds = load_dataset("squad", split=split)  # train / validation

        samples: List[Dict[str, Any]] = []
        for ex in ds:
            qid = ex["id"]
            question = ex["question"]
            context = ex["context"]
            answers = ex["answers"]["text"]  # 리스트
            samples.append(
                {
                    "id": qid,
                    "question": question,
                    "context": context,
                    "answers": answers,
                }
            )

        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def sample(self) -> Dict[str, Any]:
        return random.choice(self.samples)
