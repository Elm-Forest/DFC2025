import torch
import torch.nn as nn


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in the Lovasz paper.
    gt_sorted: [P], 其中元素为0或1（已按照预测误差排序后的标签）。
    返回： 对应每个像素位置的梯度值。
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()  # 正样本数量
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


class LovaszSigmoid(nn.Module):
    """
    多标签分割的 Lovasz 损失:
    - logits: [B, C, H, W] (未归一化, 任意实数)
    - labels: [B, C, H, W] (二值0/1)
    在本 loss 中, 我们会先对 logits 做 sigmoid => 得到每通道概率, 然后再计算 Lovasz。
    """

    def __init__(self, reduction='mean'):
        """
        Args:
            reduction (str): 'none'|'mean'|'sum'
              - 'none': 返回各通道的损失组成的向量
              - 'mean': (默认)对所有通道损失求平均
              - 'sum': 对所有通道损失求和
        """
        super(LovaszSigmoid, self).__init__()
        self.reduction = reduction

    def forward(self, logits, labels):
        """
        Args:
            logits (Tensor): [B, C, H, W], 未经概率化的分割输出
            labels (Tensor): [B, C, H, W], 每通道0或1
        Returns:
            Tensor: Lovasz 多标签损失
        """
        # 1) 简单检查
        assert logits.dim() == 4, "logits must be 4D!"
        assert labels.shape == logits.shape, "labels must have the same shape as logits!"

        B, C, H, W = logits.shape
        losses = []

        # 2) 对 logits 做 sigmoid => 概率 [0,1]
        prob = torch.sigmoid(logits)  # [B, C, H, W]

        # 3) 针对每个通道(类别)分开计算 Lovasz
        for c in range(C):
            # 取出第 c 个通道: [B, H, W]
            prob_c = prob[:, c, :, :]
            label_c = labels[:, c, :, :].float()

            # 4) 展平为一维 [B*H*W]
            prob_c_flat = prob_c.contiguous().view(-1)
            label_c_flat = label_c.contiguous().view(-1)

            # 5) 计算预测误差并排序
            errors = (label_c_flat - prob_c_flat).abs()
            errors_sorted, perm = torch.sort(errors, descending=True)
            label_sorted = label_c_flat[perm]

            # 6) 计算 Lovasz 梯度并做加权求和
            grad = lovasz_grad(label_sorted)
            loss_c = torch.dot(errors_sorted, grad)

            losses.append(loss_c)

        # 7) 对所有通道做聚合
        losses = torch.stack(losses)  # [C]
        if self.reduction == 'none':
            return losses
        elif self.reduction == 'sum':
            return losses.sum()
        else:  # 'mean'
            return losses.mean()


# ----------------- 测试示例 -----------------
if __name__ == "__main__":
    # 假设 batch_size=4, 类别数=3, H=256, W=256
    B, C, H, W = 4, 3, 256, 256

    # 构造随机的 logits
    logits = torch.randn(B, C, H, W)  # 形状 [4, 3, 256, 256]

    # 构造随机的 labels：形状相同, [0,1] 二值
    labels = torch.randint(low=0, high=2, size=(B, C, H, W))

    # 创建并计算 loss
    criterion = LovaszSigmoid(reduction='mean')
    loss_val = criterion(logits, labels)
    print(f"LovaszSigmoid Loss = {loss_val.item():.4f}")
