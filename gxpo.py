"""
GXPO (Generalized Exploration Policy Optimization) 实现
融合自适应优势估计与多粒度探索的增强版 GRPO
"""

import torch
import torch.nn.functional as F


def gxpo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    rewards: torch.Tensor,
    ref_log_probs: torch.Tensor,
    entropy: torch.Tensor = None,
    beta: float = 0.04,
    epsilon: float = 0.2,
    lambda_entropy: float = 0.01,
    alpha: float = 0.7,
) -> torch.Tensor:
    """
    GXPO Loss 计算 - 自适应多粒度探索

    Args:
        log_probs: 当前策略 log 概率 [batch * group_size, seq_len]
        old_log_probs: 旧策略 log 概率
        rewards: 奖励值 [batch * group_size]
        ref_log_probs: 参考模型 log 概率
        entropy: 策略熵 [batch * group_size, seq_len]
        beta: KL 惩罚系数
        epsilon: clip 范围
        lambda_entropy: 熵正则化系数
        alpha: 自适应优势融合系数 (intra vs inter)

    Returns:
        loss: 标量损失值
    """
    ratio = torch.exp(log_probs - old_log_probs)
    group_size = rewards.shape[0]  # 简化场景

    # 组内优势 (intra-group)
    intra_adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    # 跨组优势 (inter-group): 全局归一化
    all_rewards = rewards.view(-1, group_size)
    inter_adv = (rewards - all_rewards.mean()) / (all_rewards.std() + 1e-8)

    # 自适应融合
    advantages = alpha * intra_adv + (1 - alpha) * inter_adv

    # PPO clip
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    # KL 散度
    kl_div = torch.exp(ref_log_probs - log_probs) - (ref_log_probs - log_probs) - 1
    kl_loss = beta * kl_div.mean()

    # 熵奖励（促进探索）
    entropy_loss = -lambda_entropy * entropy.mean() if entropy is not None else 0

    return pg_loss + kl_loss + entropy_loss
