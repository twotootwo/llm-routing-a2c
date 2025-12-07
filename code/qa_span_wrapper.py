# qa_span_wrapper.py

from typing import Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


class SquadQAModelWrapper:
    """
    SQuAD용 span QA 모델 래퍼.
    - 예: deepset/roberta-large-squad2
    - predict(question, context) -> {answer, score, num_tokens}
      * num_tokens: 입력 question+context 토큰 개수 (비용 계산용)
    """

    def __init__(
        self,
        model_name: str = "deepset/roberta-large-squad2",
        device: str = "cuda",
        max_length: int = 512,
        max_answer_len: int = 30,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)

        self.device = device
        self.max_length = max_length
        self.max_answer_len = max_answer_len

    @torch.no_grad()
    def predict(self, question: str, context: str) -> Dict[str, Any]:
        # (Q, C)를 하나의 입력으로 인코딩
        inputs = self.tokenizer(
            question,
            context,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        num_tokens = int(inputs["input_ids"].shape[1])

        outputs = self.model(**inputs)
        start_logits = outputs.start_logits[0]  # (seq_len,)
        end_logits = outputs.end_logits[0]      # (seq_len,)

        # 기본적인 argmax span 선택
        start_idx = int(torch.argmax(start_logits).item())
        end_idx = int(torch.argmax(end_logits).item())

        # end < start인 경우 보정
        if end_idx < start_idx:
            end_idx = start_idx

        # 너무 긴 span은 잘라줌
        if end_idx - start_idx + 1 > self.max_answer_len:
            end_idx = start_idx + self.max_answer_len - 1

        answer_ids = inputs["input_ids"][0][start_idx : end_idx + 1]
        answer = self.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

        # 대충 score도 하나 만들어두자 (start/end logit 평균)
        score = (
            float(torch.max(start_logits).item())
            + float(torch.max(end_logits).item())
        ) / 2.0

        return {
            "answer": answer,
            "score": score,
            "num_tokens": num_tokens,
        }
