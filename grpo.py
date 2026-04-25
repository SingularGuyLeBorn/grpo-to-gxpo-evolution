"""
GRPO (Group Relative Policy Optimization) 实现
参考：DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL
"""

import torch
import torch.nn.functional as F


def grpo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    rewards: torch.Tensor,
    ref_log_probs: torch.Tensor,
    beta: float = 0.04,
    epsilon: float = 0.2,
) -> torch.Tensor:
    """
    GRPO Loss 计算

    Args:
        log_probs: 当前策略的 log 概率 [batch * group_size, seq_len]
        old_log_probs: 旧策略的 log 概率 [batch * group_size, seq_len]
        rewards: 奖励值 [batch * group_size]
        ref_log_probs: 参考模型的 log 概率
        beta: KL 惩罚系数
        epsilon: clip 范围

    Returns:
        loss: 标量损失值
    """
    # 概率比
    ratio = torch.exp(log_probs - old_log_probs)

    # 组内归一化优势
    # 假设 rewards 已按组排列
    group_size = rewards.shape[-1]  # 近似，实际需传入 group_size
    rewards_2d = rewards.view(-1, group_size)
    advantages = (rewards_2d - rewards_2d.mean(dim=-1, keepdim=True)) / (
        rewards_2d.std(dim=-1, keepdim=True) + 1e-8
    )
    advantages = advantages.flatten()

    # PPO clip 目标
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    # KL 散度惩罚
    kl_div = torch.exp(ref_log_probs - log_probs) - (ref_log_probs - log_probs) - 1
    kl_loss = beta * kl_div.mean()

    return pg_loss + kl_loss
