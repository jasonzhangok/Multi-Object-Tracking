import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import math
import matplotlib.pyplot as plt

class WarmupCosineAnnealingLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, warmup_start_lr=1e-6, base_lr=1e-3, min_lr=1e-6,
                 last_epoch=-1):
        """
        :param optimizer: PyTorch optimizer
        :param warmup_epochs: 预热阶段的 epochs 数
        :param total_epochs: 总的 epochs 数
        :param warmup_start_lr: 预热开始时的学习率
        :param base_lr: 预热结束后的基准学习率
        :param min_lr: 余弦退火最低学习率
        :param last_epoch: 用于恢复训练
        """
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.warmup_start_lr = warmup_start_lr
        self.base_lr = base_lr
        self.min_lr = min_lr
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # 线性 Warmup 计算
            lr = self.warmup_start_lr + (self.base_lr - self.warmup_start_lr) * (self.last_epoch / self.warmup_epochs)
        else:
            # 余弦退火计算
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

        return [lr for _ in self.base_lrs]


# 示例用法：
model = torch.nn.Linear(10, 2)  # 你的模型
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 创建调度器
scheduler = WarmupCosineAnnealingLR(optimizer, warmup_epochs=5, total_epochs=100, base_lr=1e-3, min_lr=1e-6)

# 训练循环
lr_list = []
total_epochs = 256
for epoch in range(total_epochs):
    optimizer.step()  # 训练模型
    scheduler.step()  # 更新学习率
    lr_list.append(scheduler.get_lr()[0])
    print(f"Epoch {epoch + 1}: Learning Rate = {scheduler.get_lr()[0]:.6f}")

plt.figure(figsize=(8, 5))
plt.plot(range(1, total_epochs + 1), lr_list, label="Learning Rate", color='b')
plt.axvline(x=5, linestyle='--', color='r', label="End of Warmup")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Warmup Cosine Annealing Learning Rate Schedule")
plt.legend()
plt.grid()
plt.show()