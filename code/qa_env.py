# qa_env.py

from typing import Dict, Any, Tuple
import numpy as np

from evaluate_v1 import f1_score, metric_max_over_ground_truths
from squad_dataset import SQuADDataset
from state_features import build_state_features

# 액션 정의
ACTION_ACCEPT = 0
ACTION_RETRY = 1
ACTION_ESCALATE = 2


class QARoutingEnv:
    """
    Q&A + LLM 라우팅 RL 환경.

    - cheap_model: 타입이 "lm" 또는 "qa" (routing_config에서 지정)
      * lm  -> HFLLMWrapper (generate, total_tokens 제공)
      * qa  -> SquadQAModelWrapper (span prediction, 토큰 수는 직접 추정)

    - strong_model: cheap과 동일하게 "lm" 또는 "qa" 가능

    state: build_state_features()로 만든 11차원 벡터
    action: ACCEPT / RETRY / ESCALATE

    reward (최종 step):
        token_ratio = total_cost / token_budget   (0~대략 1 근처)
                      여기서 total_cost는
                      - LM  : Σ (output["total_tokens"] * model_cost_factor)
                      - QA  : Σ (approx_tokens(question+context) * model_cost_factor)

        retry_ratio = retry_count / max_retry       (0~1)
        strong_used ∈ {0,1}

        reward = F1
                 - w_token  * token_ratio
                 - w_retry  * retry_ratio
                 - w_strong * strong_used

    중간 RETRY step:
        → shaping은 안 쓰고 0.0으로 둠 (원하면 작은 penalty로 바꿀 수 있음)
    """

    def __init__(
        self,
        dataset: SQuADDataset,
        cheap_model,
        cheap_kind: str,          # "lm" or "qa"
        strong_model,
        strong_kind: str,         # "lm" or "qa"
        max_retry: int = 2,
        token_budget: int = 512,
        w_token: float = 0.0,
        w_retry: float = 0.2,
        w_strong: float = 0.3,
        cheap_cost_factor: float = 1.0,
        strong_cost_factor: float = 1.0,
    ):
        self.dataset = dataset

        self.cheap_model = cheap_model
        self.cheap_kind = cheap_kind  # "lm" or "qa"

        self.strong_model = strong_model
        self.strong_kind = strong_kind  # "lm" or "qa"

        self.max_retry = max_retry
        self.token_budget = token_budget
        self.w_token = w_token
        self.w_retry = w_retry
        self.w_strong = w_strong

        # 모델별 cost factor (모델 크기 반영용)
        self.cheap_cost_factor = cheap_cost_factor
        self.strong_cost_factor = strong_cost_factor

        # episode state
        self.current_q = ""
        self.current_context = ""
        self.gold_answers = []
        self.retry_count = 0
        self.current_answer = ""
        self.current_output = None
        self.done = False

        # 누적 "cost" (이전 total_tokens 역할)
        # LM: 실제 토큰 수 * cost_factor
        # QA: 추정 토큰 수 * cost_factor
        self.total_tokens = 0.0

    # -------- QA 모델용 토큰 수 추정 --------
    def _approx_qa_tokens(self) -> int:
        """
        QA 모델에서 한 번 forward 할 때의 "대략적인 토큰 수".
        여기서는 단순히 단어 수 기준으로 근사.
        (context가 길어질수록 cost가 커지는 효과를 주기 위한 proxy)
        """
        q_tokens = len(self.current_q.split())
        c_tokens = len(self.current_context.split())
        return q_tokens + c_tokens

    # -------- LM cheap용 프롬프트 빌더 --------
    def _build_prompt_for_retry(self) -> str:
        """
        cheap LM(Qwen 등)용 프롬프트 템플릿. context까지 포함.
        QA cheap일 때는 쓰이지 않음.
        """
        base = f"""You are a reading comprehension assistant.

[Context]
{self.current_context}

[Question]
{self.current_q}
"""

        if self.retry_count == 0:
            prompt = base + """
Answer concisely with a short span copied from the context.
Answer:"""

        elif self.retry_count == 1:
            prompt = base + f"""
Previous answer:
{self.current_answer}

The previous answer might be inaccurate.
Refine your answer. Use only words from the context if possible.
New answer:"""

        else:
            prompt = base + f"""
Previous answer:
{self.current_answer}

Focus strictly on the core meaning and answer again,
as short as possible, using only information from the context.
New answer:"""

        return prompt

    # -------- cheap 모델 호출 (초기) --------
    def _cheap_first_call(self):
        """
        episode 시작 시 cheap 모델 첫 호출.
        kind에 따라 LM 또는 QA로 분기.
        반환: (answer, step_cost, token_logprobs, raw_output)
        """
        if self.cheap_kind == "lm":
            prompt = self._build_prompt_for_retry()
            out = self.cheap_model.generate(prompt)
            answer = out["answer"]
            # LM: 실제 토큰 수 * cost_factor
            step_cost = out.get("total_tokens", 0) * self.cheap_cost_factor
            token_logprobs = out.get("token_logprobs", [])
            return answer, step_cost, token_logprobs, out

        elif self.cheap_kind == "qa":
            # QA: question+context에서 직접 span 추출
            out = self.cheap_model.predict(
                question=self.current_q,
                context=self.current_context,
            )
            answer = out["answer"]
            # QA: 추정 토큰 수 * cost_factor
            approx_tokens = self._approx_qa_tokens()
            step_cost = approx_tokens * self.cheap_cost_factor
            token_logprobs = []  # QA 모델은 토큰 로그 확률 정보 없음
            return answer, step_cost, token_logprobs, out

        else:
            raise ValueError(f"Unknown cheap_kind: {self.cheap_kind}")

    # -------- cheap 모델 호출 (RETRY) --------
    def _cheap_retry_call(self):
        """
        RETRY 액션 시 cheap 모델 재호출.
        LM이면 프롬프트를 바꿔서 generate,
        QA이면 다시 predict (결과는 같더라도 cost는 늘어남).
        """
        if self.cheap_kind == "lm":
            prompt = self._build_prompt_for_retry()
            out = self.cheap_model.generate(prompt)
            answer = out["answer"]
            step_cost = out.get("total_tokens", 0) * self.cheap_cost_factor
            token_logprobs = out.get("token_logprobs", [])
            return answer, step_cost, token_logprobs, out

        elif self.cheap_kind == "qa":
            out = self.cheap_model.predict(
                question=self.current_q,
                context=self.current_context,
            )
            answer = out["answer"]
            approx_tokens = self._approx_qa_tokens()
            step_cost = approx_tokens * self.cheap_cost_factor
            token_logprobs = []
            return answer, step_cost, token_logprobs, out

        else:
            raise ValueError(f"Unknown cheap_kind: {self.cheap_kind}")

    # -------- strong 모델 호출 --------
    def _strong_call(self):
        """
        ESCALATE 액션 시 strong 모델 호출.
        LM 또는 QA 모두 지원.
        반환: (answer, step_cost)
        """
        if self.strong_kind == "lm":
            prompt = f"""You are a strong reading comprehension model.

[Context]
{self.current_context}

[Question]
{self.current_q}

Answer accurately with a concise span from the context.
Answer:"""
            out = self.strong_model.generate(prompt)
            answer = out["answer"]
            step_cost = out.get("total_tokens", 0) * self.strong_cost_factor
            return answer, step_cost

        elif self.strong_kind == "qa":
            out = self.strong_model.predict(
                question=self.current_q,
                context=self.current_context,
            )
            answer = out["answer"]
            approx_tokens = self._approx_qa_tokens()
            step_cost = approx_tokens * self.strong_cost_factor
            return answer, step_cost

        else:
            raise ValueError(f"Unknown strong_kind: {self.strong_kind}")

    # -------- 에피소드 제어 --------
    def reset(self) -> np.ndarray:
        sample = self.dataset.sample()
        self.current_q = sample["question"]
        self.current_context = sample["context"]
        self.gold_answers = sample["answers"]
        self.retry_count = 0
        self.done = False
        self.total_tokens = 0.0  # cost 누적

        # cheap 첫 호출
        answer, step_cost, token_logprobs, out = self._cheap_first_call()
        self.current_output = out
        self.current_answer = answer
        self.total_tokens += step_cost

        state = build_state_features(
            question=self.current_q,
            context=self.current_context,
            cheap_answer=self.current_answer,
            token_logprobs=token_logprobs,
            retry_count=self.retry_count,
            max_retry=self.max_retry,
            gold_answers=self.gold_answers,
            total_tokens=self.total_tokens,
            token_budget=self.token_budget,
        )
        return state

    # -------- cheap/strong answer score --------
    def _compute_answer_score(self, answer: str) -> float:
        return metric_max_over_ground_truths(
            f1_score, answer, self.gold_answers
        )

    # -------- 최종 reward 계산 --------
    def _final_reward(self, f1: float, strong_used: int) -> float:
        # total_tokens는 이제 "cost units" (LM 토큰 또는 QA 추정토큰 × cost_factor)
        if self.token_budget > 0:
            token_ratio = min(self.total_tokens / self.token_budget, 1.0)
        else:
            token_ratio = 0.0

        retry_ratio = (
            self.retry_count / self.max_retry if self.max_retry > 0 else 0.0
        )

        reward = (
            f1
            - self.w_token * token_ratio
            - self.w_retry * retry_ratio
            - self.w_strong * strong_used
        )
        return reward

    # -------- step --------
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if self.done:
            raise RuntimeError("Episode is done. Call reset().")

        info: Dict[str, Any] = {}

        # ----- ACCEPT -----
        if action == ACTION_ACCEPT:
            final_answer = self.current_answer
            f1 = self._compute_answer_score(final_answer)
            strong_used = 0

            reward = self._final_reward(f1, strong_used)

            self.done = True
            info["final_answer"] = final_answer
            info["answer_score"] = f1
            info["total_tokens"] = self.total_tokens

            # 종료이므로 state는 0벡터 반환
            return np.zeros(11, np.float32), reward, True, info

        # ----- ESCALATE -----
        if action == ACTION_ESCALATE:
            final_answer, step_cost = self._strong_call()
            self.total_tokens += step_cost

            f1 = self._compute_answer_score(final_answer)
            strong_used = 1

            reward = self._final_reward(f1, strong_used)

            self.done = True
            info["final_answer"] = final_answer
            info["answer_score"] = f1
            info["total_tokens"] = self.total_tokens

            return np.zeros(11, np.float32), reward, True, info

        # ----- RETRY -----
        if action == ACTION_RETRY:
            # 더 이상 retry 못하면 accept로 처리
            if self.retry_count >= self.max_retry:
                final_answer = self.current_answer
                f1 = self._compute_answer_score(final_answer)
                strong_used = 0

                reward = self._final_reward(f1, strong_used)

                self.done = True
                info["final_answer"] = final_answer
                info["answer_score"] = f1
                info["total_tokens"] = self.total_tokens

                return np.zeros(11, np.float32), reward, True, info

            # retry 가능
            self.retry_count += 1

            answer, step_cost, token_logprobs, out = self._cheap_retry_call()
            self.current_output = out
            self.current_answer = answer
            self.total_tokens += step_cost

            state = build_state_features(
                question=self.current_q,
                context=self.current_context,
                cheap_answer=self.current_answer,
                token_logprobs=token_logprobs,
                retry_count=self.retry_count,
                max_retry=self.max_retry,
                gold_answers=self.gold_answers,
                total_tokens=self.total_tokens,
                token_budget=self.token_budget,
            )

            # shaping: 지금은 0.0 (원하면 -0.01 같은 penalty 넣어도 됨)
            intermediate_reward = 0.0

            return state, intermediate_reward, False, info

        # 그 외 잘못된 action
        raise ValueError(f"Invalid action: {action}")
