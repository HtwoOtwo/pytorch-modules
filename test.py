import torch
import torch.nn.functional as F

predicted = torch.tensor([0.9, 1.2, 3.2, 4.0])
target = torch.tensor([1.0, 1.0, 3.0, 4.5])


loss_same_order = F.l1_loss(predicted, target)
loss_different_order = F.l1_loss(predicted, torch.tensor([1.0, 1.0, 3.0, 4.5]))

print(loss_same_order)  # 输出: tensor(0.1750)
print(loss_different_order)  # 输出: tensor(0.1750)
