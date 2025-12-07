import os
import glob
import numpy as np
import torch
import random   # 👈 추가

from qa_env import QARoutingEnv, ACTION_ACCEPT, ACTION_ESCALATE, ACTION_RETRY
from train_a2c import ActorCritic
from routing_config import build_env_and_paths


# ---------- 1. 에피소드 한 번 실행하는 공통 함수 ----------

def run_episode(env: QARoutingEnv, policy_fn, model=None, device="cpu"):
    """
    env + policy_fn(및 optional A2C model)을 사용해
    에피소드 1개를 돌리고,
    총 reward / 최종 F1 / 총 토큰 수 / action 리스트를 반환.
    """
    state = env.reset()
    done = False
    total_reward = 0.0
    final_f1 = 0.0
    total_tokens = 0
    actions = []   # 👈 에피소드 동안의 action 기록

    while not done:
        action = policy_fn(state, env, model, device)
        next_state, reward, done, info = env.step(action)

        total_reward += reward
        state = next_state
        actions.append(int(action))

        if "answer_score" in info:
            final_f1 = info["answer_score"]
        if "total_tokens" in info:
            total_tokens = info["total_tokens"]

    return total_reward, final_f1, total_tokens, actions


# ---------- 2. 여러 에피소드에 대해 평균 내기 ----------

def evaluate_policy(env, policy_fn, model=None, device="cpu", num_episodes=100):
    rewards = []
    f1s = []
    tokens = []
    all_actions = []   # 👈 전체 에피소드 action 모으기

    base_seed = 1234
    np.random.seed(base_seed)
    random.seed(base_seed)
    torch.manual_seed(base_seed)

    for _ in range(num_episodes):
        r, f1, tok, actions = run_episode(env, policy_fn, model, device)
        rewards.append(r)
        f1s.append(f1)
        tokens.append(tok)
        all_actions.extend(actions)

    # 👇 action 비율 계산
    p_accept = p_retry = p_escalate = 0.0
    if len(all_actions) > 0:
        total_actions = len(all_actions)
        num_accept = all_actions.count(ACTION_ACCEPT)
        num_retry = all_actions.count(ACTION_RETRY)
        num_escalate = all_actions.count(ACTION_ESCALATE)

        p_accept = num_accept / total_actions
        p_retry = num_retry / total_actions
        p_escalate = num_escalate / total_actions

    return {
        "avg_reward": float(np.mean(rewards)),
        "avg_f1": float(np.mean(f1s)),
        "avg_tokens": float(np.mean(tokens)),
        "p_accept": float(p_accept),
        "p_retry": float(p_retry),
        "p_escalate": float(p_escalate),
    }


# ---------- 3. Baseline policy 정의들 ----------

def policy_always_cheap(state, env, model, device):
    return ACTION_ACCEPT


def policy_always_escalate(state, env, model, device):
    return ACTION_ESCALATE


def policy_random(state, env, model, device):
    return int(np.random.choice([ACTION_ACCEPT, ACTION_ESCALATE]))


def policy_trained_a2c(state, env, model: ActorCritic, device):
    state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _ = model(state_tensor)
    action = torch.argmax(logits, dim=-1).item()
    return int(action)


# ---------- 4. main ----------

def main():
    # policy network용 디바이스 (env 내부 cheap/strong은 routing_config에서 이미 정해짐)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Eval device (policy net): {device}")

    # ✅ routing_config 기반으로 env + 경로들 한 번에 가져오기
    # split="validation" 고정 evaluation
    env, checkpoint_dir, result_dir, _ = build_env_and_paths(split="validation")

    num_eval_episodes = 100

    # ----- Baselines 평가 -----
    print("\n=== Evaluating baselines ===")
    res_cheap = evaluate_policy(env, policy_always_cheap, None, device, num_eval_episodes)
    print(
        f"Always Cheap    -> reward={res_cheap['avg_reward']:.4f}, "
        f"F1={res_cheap['avg_f1']:.4f}, tokens={res_cheap['avg_tokens']:.1f}, "
        f"action_dist(A/R/E)={res_cheap['p_accept']:.2f}/"
        f"{res_cheap['p_retry']:.2f}/{res_cheap['p_escalate']:.2f}"
    )

    res_escalate = evaluate_policy(env, policy_always_escalate, None, device, num_eval_episodes)
    print(
        f"Always Escalate -> reward={res_escalate['avg_reward']:.4f}, "
        f"F1={res_escalate['avg_f1']:.4f}, tokens={res_escalate['avg_tokens']:.1f}, "
        f"action_dist(A/R/E)={res_escalate['p_accept']:.2f}/"
        f"{res_escalate['p_retry']:.2f}/{res_escalate['p_escalate']:.2f}"
    )

    res_random = evaluate_policy(env, policy_random, None, device, num_eval_episodes)
    print(
        f"Random          -> reward={res_random['avg_reward']:.4f}, "
        f"F1={res_random['avg_f1']:.4f}, tokens={res_random['avg_tokens']:.1f}, "
        f"action_dist(A/R/E)={res_random['p_accept']:.2f}/"
        f"{res_random['p_retry']:.2f}/{res_random['p_escalate']:.2f}"
    )

    # ----- A2C 체크포인트들 평가 -----
    print("\n=== Evaluating trained A2C checkpoints ===")
    state_dim = 11   # 현재 state_features 길이
    action_dim = 3   # [ACCEPT, RETRY, ESCALATE]

    ckpt_paths = glob.glob(os.path.join(checkpoint_dir, "a2c_actor_critic_ep*.pt"))

    def extract_ep_num(path: str) -> int:
        base = os.path.basename(path)
        num_part = base.replace("a2c_actor_critic_ep", "").replace(".pt", "")
        try:
            return int(num_part)
        except ValueError:
            return 0

    ckpt_paths = sorted(ckpt_paths, key=extract_ep_num)

    if not ckpt_paths:
        print("No checkpoint files found under", checkpoint_dir)
    else:
        for ckpt_path in ckpt_paths:
            ep_num = extract_ep_num(ckpt_path)
            model = ActorCritic(state_dim, action_dim).to(device)
            state_dict = torch.load(ckpt_path, map_location=device)
            try:
                model.load_state_dict(state_dict)
            except RuntimeError as e:
                print(f"⚠️ Skipping checkpoint {ckpt_path} (shape mismatch: {e})")
                continue

            model.eval()
            res_a2c = evaluate_policy(env, policy_trained_a2c, model, device, num_eval_episodes)
            print(
                f"A2C (ep={ep_num:5d}) -> reward={res_a2c['avg_reward']:.4f}, "
                f"F1={res_a2c['avg_f1']:.4f}, tokens={res_a2c['avg_tokens']:.1f}, "
                f"action_dist(A/R/E)={res_a2c['p_accept']:.2f}/"
                f"{res_a2c['p_retry']:.2f}/{res_a2c['p_escalate']:.2f} "
                f"[ckpt: {ckpt_path}]"
            )

    # ----- 최종 모델 평가 (RESULT_DIR 아래에 저장된 것) -----
    final_ckpt = os.path.join(result_dir, "a2c_actor_critic_squad_final.pt")
    if os.path.exists(final_ckpt):
        print("\n=== Evaluating FINAL A2C model ===")
        model = ActorCritic(state_dim, action_dim).to(device)
        state_dict = torch.load(final_ckpt, map_location=device)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            print(f"⚠️ FINAL checkpoint shape mismatch, skipping: {e}")
        else:
            model.eval()
            res_final = evaluate_policy(env, policy_trained_a2c, model, device, num_eval_episodes)
            print(
                f"A2C (FINAL)    -> reward={res_final['avg_reward']:.4f}, "
                f"F1={res_final['avg_f1']:.4f}, tokens={res_final['avg_tokens']:.1f}, "
                f"action_dist(A/R/E)={res_final['p_accept']:.2f}/"
                f"{res_final['p_retry']:.2f}/{res_final['p_escalate']:.2f} "
                f"[ckpt: {final_ckpt}]"
            )
    else:
        print("\nNo final A2C checkpoint found:", final_ckpt)


if __name__ == "__main__":
    main()
