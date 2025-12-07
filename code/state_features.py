# state_features.py
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from difflib import SequenceMatcher

# 전역적으로 하나 만들어서 재사용 (속도 문제 때문에)
tfidf = TfidfVectorizer(stop_words="english")


def rough_similarity(a: str, b: str) -> float:
    """대략적인 텍스트 유사도: 공통 부분의 비율."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def contains_number(text: str) -> int:
    return 1 if re.search(r"\d", text) else 0


def contains_entity(text: str) -> int:
    """
    간단한 entity 규칙: 대문자로 시작하는 단어가 있으면 entity라고 봄.
    (NER 대신 heuristic)
    """
    for token in text.split():
        if token and token[0].isupper():
            return 1
    return 0


def build_state_features(
    question: str,
    context: str,
    cheap_answer: str,
    token_logprobs: list,
    retry_count: int,
    max_retry: int,
    gold_answers=None,
    total_tokens: int = 0,
    token_budget: int = 512,
):
    """
    cheap LLM의 출력과 질문/컨텍스트를 기반으로 상태 벡터 11차원 구성.

    state 구성 (각각 float):
      0: answer_length (정규화)
      1: avg_logprob (스케일링)
      2: TF-IDF cosine similarity(question, answer)
      3: has_number (0/1)
      4: has_entity (0/1)
      5: retry_ratio (retry_count / max_retry)
      6: cheap_f1_est (gold과의 rough similarity 최대값)
      7: is_empty (답이 거의 비어있으면 1)
      8: question_length_norm
      9: context_length_norm
     10: token_ratio (total_tokens / token_budget)
    """

    # --- Feature 1: cheap answer length ---
    answer_length = len(cheap_answer.split())

    # --- Feature 2: avg log prob ---
    if token_logprobs and len(token_logprobs) > 0:
        avg_logprob = float(np.mean(token_logprobs))
    else:
        avg_logprob = -5.0   # 매우 낮은 confidence로 설정

    # --- Feature 3: TF-IDF cosine similarity (Q vs A) ---
    try:
        docs = [question, cheap_answer]
        mat = tfidf.fit_transform(docs)
        cosine_sim = (mat[0] @ mat[1].T).A[0][0]
    except Exception:
        cosine_sim = 0.0

    # --- Feature 4: has number ---
    has_number = contains_number(cheap_answer)

    # --- Feature 5: has entity (rough) ---
    has_entity = contains_entity(cheap_answer)

    # --- Feature 6: retry ratio ---
    retry_ratio = retry_count / max_retry if max_retry > 0 else 0.0

    # --- Feature 7: cheap rough F1 with gold ---
    if gold_answers:
        cheap_f1_est = max(rough_similarity(cheap_answer, g) for g in gold_answers)
    else:
        cheap_f1_est = 0.0

    # --- Feature 8: cheap answer empty? ---
    is_empty = 1.0 if len(cheap_answer.strip()) < 2 else 0.0

    # --- Feature 9: question length (정규화) ---
    q_len = len(question.split())
    q_len_norm = min(q_len / 40.0, 1.0)   # 대충 40단어 이상이면 1로 클램프

    # --- Feature 10: context length (정규화) ---
    ctx_len = len(context.split())
    ctx_len_norm = min(ctx_len / 300.0, 1.0)  # 300단어 기준으로 정규화

    # --- Feature 11: token ratio (정규화) ---
    if token_budget > 0:
        token_ratio = min(total_tokens / token_budget, 1.0)
    else:
        token_ratio = 0.0

    state_vec = np.array([
        answer_length,
        avg_logprob,
        cosine_sim,
        has_number,
        has_entity,
        retry_ratio,
        cheap_f1_est,
        is_empty,
        q_len_norm,
        ctx_len_norm,
        token_ratio,
    ], dtype=np.float32)

    # --- 간단 정규화/스케일링 ---
    state_vec[0] = min(state_vec[0] / 20.0, 1.0)          # answer length normalize
    # avg_logprob: 보통 -5 ~ 0 근처 → [-1, 1] 정도로 스케일
    state_vec[1] = max(min(state_vec[1] / -1.0, 1.0), -1.0)

    return state_vec
