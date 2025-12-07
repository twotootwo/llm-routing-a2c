# Cost-aware LLM Routing with A2C on SQuAD

이 저장소는 **두 개의 LLM(cheap / strong)** 을 준비해 두고,  
각 **질의(QA task)**마다 어떤 모델을 사용하는 것이 더 이득인지  
강화학습(Actor–Critic, A2C)으로 학습하는 프로젝트입니다.

- **Cheap LLM**: 상대적으로 성능은 낮지만, 호출 비용이 작은 모델  
- **Strong LLM**: 성능은 우수하지만, 호출 비용이 큰 모델  
- **Agent (A2C)**: cheap의 1차 답변을 보고  
  - 그대로 사용할지 (ACCEPT)  
  - cheap으로 다시 시도할지 (RETRY)  
  - strong으로 escalate할지 (ESCALATE)  
  를 정책으로 학습합니다.

---

### State (요약)

State는 cheap LLM이 생성한 1차 답변을 기반으로 한  
**길이, log-prob, question–answer 유사도, 숫자/엔티티 포함 여부,  
토큰 사용량 비율** 등의 11차원 피처 벡터로 구성됩니다.  

자세한 정의는 `state_features.py`를 참고하세요.

---

### Reward (요약)

최종 reward는 QA F1 점수에서 비용 항을 뺀 형태로 정의됩니다.

`Reward = F1(final_answer)
          - w_token  * token_ratio
          - w_strong * strong_ratio
          - w_retry  * retry_ratio`

- **F1(final_answer)**: 최종 답변과 정답 간의 F1 score  
- **token_ratio**: 에피소드에서 사용한 토큰 수 / token budget  
- **strong_ratio**: strong 모델 호출 비율  
- **retry_ratio**: cheap으로 재시도한 비율  

---

## 1. Project Structure

```text
.
├── training.ipynb           # Colab/노트북에서 전체 실험 실행용
├── train_a2c.py             # A2C 학습 스크립트 (메인)
├── evaluate_models.py       # Baseline + A2C checkpoint 평가
├── evaluate_v1.py           # EM/F1 계산 함수
├── qa_env.py                # QARoutingEnv (Gym 스타일 환경)
├── routing_config.py        # cheap/strong 모델 및 reward weight 설정
├── squad_dataset.py         # HuggingFace SQuAD v1.1 래퍼
├── state_features.py        # 11차원 state feature 추출
├── llm_wrapper.py           # HF LLM wrapper (cheap LM)
├── qa_span_wrapper.py       # QA span extractor wrapper (strong QA)
└── lm+qa/                   # 학습 결과/체크포인트 저장 폴더 (실험 후 생성)

training.ipynb 노트북을 열어
상단 셀에서 환경 설치 → 하단 셀에서 학습 및 평가 명령을 순서대로 실행하면 됩니다.
```
---

## 2. Project Dependencies

```code
!pip install -q \
  torch==2.2.2 torchvision==0.17.2 \
  transformers==4.44.2 \
  accelerate==0.33.0 \
  datasets==2.21.0 \
  sentencepiece \
  matplotlib \
  "numpy>=1.24.0"
```
---
## 3. Training
아래는 예시 하이퍼파라미터로 학습을 수행하는 명령어입니다.
CLI에서 태그(옵션)를 생략하면, train_a2c.py에 정의된 기본값을 그대로 사용합니다.
```code
!python train_a2c.py \
  --num_episodes 1000 \
  --gamma 0.99 \
  --lr 1e-3 \
  --value_loss_coef 0.5 \
  --entropy_coef 0.01 \
  --max_grad_norm 0.5 \
  --w_token 0.01 \
  --w_retry 0.2 \
  --w_strong 0.7
```
---
## 4. Evaluation
학습이 끝난 뒤, 동일한 환경에서 다음 명령어로
베이스라인 정책과 A2C 정책을 평가할 수 있습니다.
```code
!python evaluate_models.py
```
