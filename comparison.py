"""
GRPO vs GXPO 对比演示脚本
模拟一个简单的推理场景来比较两者的损失表现
"""
import torch
import torch.nn.functional as F

from grpo import grpo_loss
from gxpo import gxpo_loss

# 模拟数据: batch_size=4, group_size=4, seq_len=32
batch_size, group_size, seq_len = 4, 4, 32

log_probs = torch.randn(batch_size * group_size, seq_len).log_softmax(dim=-1)
old_log_probs = torch.randn(batch_size * group_size, seq_len).log_softmax(dim=-1)
ref_log_probs = torch.randn(batch_size * group_size, seq_len).log_softmax(dim=-1)
rewards = torch.randn(batch_size * group_size)
entropy = torch.rand(batch_size * group_size, seq_len)

# GRPO loss
loss_grpo = grpo_loss(log_probs.clone(), old_log_probs.clone(),
                      rewards.clone(), ref_log_probs.clone())

# GXPO loss
loss_gxpo = gxpo_loss(log_probs.clone(), old_log_probs.clone(),
                      rewards.clone(), ref_log_probs.clone(),
                      entropy=entropy.clone())

print("=" * 50)
print("GRPO vs GXPO 对比结果")
print("=" * 50)
print(f"GRPO Loss: {loss_grpo.item():.4f}")
print(f"GXPO Loss: {loss_gxpo.item():.4f}")
print(f"差异: {abs(loss_gxpo.item() - loss_grpo.item()):.4f}")
print()
print("GXPO 比 GRPO 多出的组件:")
print("  - 自适应优势融合 (alpha=0.7)")
print("  - 熵正则化 (lambda=0.01)")
print("  - 多粒度探索支持")
print("=" * 50)
