# train_a2c.py

import os
import random
import csv
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import argparse
from squad_dataset import SQuADDataset
from qa_span_wrapper import SquadQAModelWrapper


from qa_env import QARoutingEnv
from routing_config import default_routing_config, build_env_and_paths



# ========= Actor-Critic 모델 정의 (그대로) ========= #
class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor):
        x = self.body(state)
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value


# ========= A2C 학습 루프 ========= #
def make_checkpoint_episodes(num_episodes: int):
    ratios = [0.02, 0.1, 0.2, 0.4, 1.0]  # 기존 [10,50,100,200,500] 패턴 유지용
    eps = {max(1, int(num_episodes * r)) for r in ratios}
    return sorted(eps)

def train_a2c(
    env: QARoutingEnv,
    num_episodes: int = 50,
    gamma: float = 0.99,
    lr: float = 1e-3,
    value_loss_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    print_interval: int = 5,
    device: str = "cpu",
    checkpoint_dir: str = "trained_model",
    checkpoint_episodes=None,   # 예: [10, 50, 100, ...]
):
    state_dim = 11   # ✅ state_features: 11차원
    action_dim = 3   # 0: ACCEPT, 1: RETRY, 2: ESCALATE

    # ✅ 체크포인트 폴더 생성
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ✅ None이면 빈 리스트로
    if checkpoint_episodes is None:
        checkpoint_episodes = []

    model = ActorCritic(state_dim, action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    reward_history = []
    f1_history = []
    action_history = []
    tokens_history = []
    policy_loss_history = []
    value_loss_history = []

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        episode_reward = 0.0
        final_f1 = 0.0
        actions_in_episode = []

        log_probs = []
        values = []
        rewards = []
        entropies = []

        done = False

        while not done:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)

            logits, value = model(state_tensor)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

            action_idx = action.item()
            next_state, reward, done, info = env.step(action_idx)

            log_probs.append(log_prob.squeeze(0))
            values.append(value.squeeze(0))
            rewards.append(reward)
            entropies.append(entropy.squeeze(0))

            episode_reward += reward
            actions_in_episode.append(action_idx)

            if "answer_score" in info:
                final_f1 = info["answer_score"]

            state = next_state

        # ---------- 에피소드 종료: return & advantage 계산 ---------- #
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        values_tensor = torch.stack(values).squeeze(-1)
        log_probs_tensor = torch.stack(log_probs)
        entropies_tensor = torch.stack(entropies)

        advantages = returns - values_tensor.detach()

        policy_loss = -(log_probs_tensor * advantages).mean()
        value_loss = (returns - values_tensor).pow(2).mean()
        entropy = entropies_tensor.mean()

        loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        # 로그 저장
        reward_history.append(episode_reward)
        f1_history.append(final_f1)
        action_history.append(actions_in_episode)
        tokens_history.append(getattr(env, "total_tokens", 0))
        policy_loss_history.append(policy_loss.item())
        value_loss_history.append(value_loss.item())

        # ✅ 중간 체크포인트 저장
        if episode in checkpoint_episodes:
            ckpt_path = os.path.join(
                checkpoint_dir, f"a2c_actor_critic_ep{episode}.pt"
            )
            torch.save(model.state_dict(), ckpt_path)
            print(f"📦 Saved checkpoint at episode {episode} -> {ckpt_path}")

        if episode % print_interval == 0:
            recent_rewards = reward_history[-print_interval:]
            recent_f1 = f1_history[-print_interval:]
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            avg_f1 = sum(recent_f1) / max(1, len(recent_f1))
            global_avg_f1 = sum(f1_history) / len(f1_history)

            flat_actions = [a for ep in action_history[-print_interval:] for a in ep]
            if flat_actions:
                num_accept = flat_actions.count(0)
                num_retry = flat_actions.count(1)
                num_escalate = flat_actions.count(2)
                total_actions = len(flat_actions)
                pa = num_accept / total_actions
                pr = num_retry / total_actions
                pe = num_escalate / total_actions
            else:
                pa = pr = pe = 0.0

            print(
                f"[Episode {episode}/{num_episodes}] "
                f"avg_reward={avg_reward:.3f}, "
                f"avg_F1(last_{print_interval})={avg_f1:.3f}, "
                f"global_avg_F1={global_avg_f1:.3f}, "
                f"last_reward={episode_reward:.3f}, last_F1={final_f1:.3f}, "
                f"action_dist(A/R/E)={pa:.2f}/{pr:.2f}/{pe:.2f}"
            )

    return (
        model,
        reward_history,
        f1_history,
        action_history,
        tokens_history,
        policy_loss_history,
        value_loss_history,
    )


# ========= main ========= #

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=500)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--value_loss_coef", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    # env 내부 reward weight (넘기지 않으면 현재 설정 유지)
    parser.add_argument("--w_token", type=float, default=None)
    parser.add_argument("--w_retry", type=float, default=None)
    parser.add_argument("--w_strong", type=float, default=None)

    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    # Env 생성 (routing_config에 따라 자동으로 cheap/strong 래퍼 생성)
    config = default_routing_config()

    # 기존 파일 삭제 
    for d in [config.checkpoint_dir, config.result_dir]:
        if os.path.exists(d):
            print(f"[train_a2c] Removing existing directory: {d}")
            shutil.rmtree(d)

    env, ckpt_dir, result_dir, ckpt_eps = build_env_and_paths(
        split="train",
        config=config,
    )
    if ckpt_eps is None or len(ckpt_eps) == 0:
        ckpt_eps = make_checkpoint_episodes(args.num_episodes)
    else:
        # routing_config에 뭔가 지정돼 있으면, 범위만 num_episodes 안으로 맞추고
        # 마지막 에피소드(num_episodes)는 항상 포함되게 보정
        ckpt_eps = sorted({e for e in ckpt_eps if 1 <= e <= args.num_episodes} | {args.num_episodes})

    print(f"[train_a2c] checkpoint episodes = {ckpt_eps}")
    #QA ENV weight
    if args.w_token is not None:
        env.w_token = args.w_token
    if args.w_retry is not None:
        env.w_retry = args.w_retry
    if args.w_strong is not None:
        env.w_strong = args.w_strong
    policy_device = "cuda" if torch.cuda.is_available() else "cpu"

    model, reward_hist, f1_hist, action_hist, tokens_hist, policy_loss_hist, value_loss_hist = train_a2c(
        env=env,
        num_episodes=args.num_episodes,
        gamma=args.gamma,
        lr=args.lr,
        value_loss_coef=args.value_loss_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        print_interval=5,
        device=policy_device,
        checkpoint_dir=ckpt_dir,
        checkpoint_episodes=ckpt_eps,
    )

    # === 결과 저장 경로 (result_dir 아래에 정리) ===
    logs_path = os.path.join(result_dir, "a2c_logs.csv")
    actions_path = os.path.join(result_dir, "a2c_actions.csv")
    reward_png = os.path.join(result_dir, "reward_curve.png")
    f1_png = os.path.join(result_dir, "f1_curve.png")
    tokens_png = os.path.join(result_dir, "tokens_curve.png")
    policy_loss_png = os.path.join(result_dir, "policy_loss_curve.png")
    value_loss_png = os.path.join(result_dir, "value_loss_curve.png")
    action_hist_png = os.path.join(result_dir, "action_histogram.png")
    final_ckpt = os.path.join(result_dir, "a2c_actor_critic_squad_final.pt")
    hyperparam_txt = os.path.join(result_dir, "hyperparams.txt")
    with open(hyperparam_txt, "w") as f:
        f.write("# A2C training hyperparameters\n")
        f.write(f"num_episodes = {args.num_episodes}\n")
        f.write(f"gamma = {args.gamma}\n")
        f.write(f"lr = {args.lr}\n")
        f.write(f"value_loss_coef = {args.value_loss_coef}\n")
        f.write(f"entropy_coef = {args.entropy_coef}\n")
        f.write(f"max_grad_norm = {args.max_grad_norm}\n")
        f.write("\n# Environment reward weights\n")
        # env 쪽 실제 값을 기록 (CLI로 안 바꿨으면 기본값이 들어감)
        f.write(f"w_token = {getattr(env, 'w_token', 'N/A')}\n")
        f.write(f"w_retry = {getattr(env, 'w_retry', 'N/A')}\n")
        f.write(f"w_strong = {getattr(env, 'w_strong', 'N/A')}\n")

    # === 1) CSV 로그 저장 ===
    with open(logs_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["episode", "reward", "final_f1", "total_tokens", "policy_loss", "value_loss"]
        )
        for i, (r, f1, tok, pl, vl) in enumerate(
            zip(reward_hist, f1_hist, tokens_hist, policy_loss_hist, value_loss_hist),
            start=1,
        ):
            writer.writerow([i, r, f1, tok, pl, vl])

    with open(actions_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "actions"])
        for i, ep_actions in enumerate(action_hist, start=1):
            writer.writerow([i, " ".join(map(str, ep_actions))])

    # === 2) 그래프 PNG 저장 ===

    # Reward curve
    plt.figure(figsize=(10, 4))
    plt.plot(reward_hist)
    plt.title("Reward Curve")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig(reward_png)
    plt.close()

    # F1 curve
    plt.figure(figsize=(10, 4))
    plt.plot(f1_hist)
    plt.title("Final F1 per Episode")
    plt.xlabel("Episode")
    plt.ylabel("F1")
    plt.tight_layout()
    plt.savefig(f1_png)
    plt.close()

    # Tokens curve
    plt.figure(figsize=(10, 4))
    plt.plot(tokens_hist)
    plt.title("Total Tokens per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Tokens")
    plt.tight_layout()
    plt.savefig(tokens_png)
    plt.close()

    # Policy loss curve
    plt.figure(figsize=(10, 4))
    plt.plot(policy_loss_hist)
    plt.title("Policy Loss (Actor) per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Policy Loss")
    plt.tight_layout()
    plt.savefig(policy_loss_png)
    plt.close()

    # Value loss curve
    plt.figure(figsize=(10, 4))
    plt.plot(value_loss_hist)
    plt.title("Value Loss (Critic) per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Value Loss")
    plt.tight_layout()
    plt.savefig(value_loss_png)
    plt.close()

    # Action histogram
    flat_actions = [a for ep in action_hist for a in ep]
    if flat_actions:
        plt.figure(figsize=(6, 4))
        plt.hist(flat_actions, bins=[-0.5, 0.5, 1.5, 2.5], rwidth=0.8)
        plt.title("Action Histogram (0=ACCEPT, 1=RETRY, 2=ESCALATE)")
        plt.xlabel("Action")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(action_hist_png)
        plt.close()

    # === 3) 최종 모델 weight 저장 ===
    torch.save(model.state_dict(), final_ckpt)
    print(f"학습 완료! 최종 모델: {final_ckpt}")
    print(f"로그: {logs_path}, {actions_path}")
    print(f"그래프: {reward_png}, {f1_png}, {tokens_png}, "
          f"{policy_loss_png}, {value_loss_png}, {action_hist_png}")
    print(f"하이퍼파라미터: {hyperparam_txt}")

if __name__ == "__main__":
    main()