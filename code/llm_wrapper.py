# llm_wrapper.py

from typing import Dict, Any, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class HFLLMWrapper:
    """
    Hugging Face causal LM 래퍼.
    - generate(prompt)로 텍스트 생성
    - token-level logprob과 토큰 수를 함께 리턴
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        max_new_tokens: int = 128,
        temperature: float = 0.2,
        top_p: float = 0.9,
        top_k: int = 40,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if "cuda" in device else torch.float32,
        ).to(device)

        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

        # pad_token이 없으면 eos_token으로 설정
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.no_grad()
    def generate(self, prompt: str) -> Dict[str, Any]:
        """
        prompt를 입력으로 답변을 생성.
        return:
            {
              "answer": str,
              "token_logprobs": List[float],  # 생성된 토큰들의 log p
              "num_prompt_tokens": int,
              "num_answer_tokens": int,
              "total_tokens": int,
            }
        """
        # 1) 프롬프트 토크나이즈
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        prompt_len = inputs["input_ids"].shape[1]

        # 2) 생성 (scores 반환하도록 설정)
        output = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True if self.temperature > 0 else False,
            temperature=self.temperature if self.temperature > 0 else None,
            top_p=self.top_p if self.temperature > 0 else None,
            top_k=self.top_k if self.temperature > 0 else None,
            pad_token_id=self.tokenizer.pad_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )

        sequences = output.sequences  # (1, prompt_len + gen_len)
        scores: List[torch.Tensor] = output.scores  # 길이 = gen_len

        # 3) 생성된 부분만 잘라서 answer 복원
        generated_ids = sequences[0, prompt_len:]
        answer = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        ).strip()

        # 4) token-level log-prob 계산
        token_logprobs: List[float] = []
        # scores[i]: (batch=1, vocab_size) -> 해당 스텝에서 생성된 token의 log prob 추출
        for step_logits, token_id in zip(scores, generated_ids):
            log_probs = torch.log_softmax(step_logits[0], dim=-1)
            token_logprobs.append(log_probs[token_id].item())

        num_prompt_tokens = int(prompt_len)
        num_answer_tokens = int(generated_ids.shape[0])
        total_tokens = num_prompt_tokens + num_answer_tokens

        return {
            "answer": answer,
            "token_logprobs": token_logprobs,
            "num_prompt_tokens": num_prompt_tokens,
            "num_answer_tokens": num_answer_tokens,
            "total_tokens": total_tokens,
        }
