import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label (multi-channel) binary segmentation.
    ---------------------------------------------------
    logits: Tensor of shape (B, C, H, W)
    labels: Tensor of the same shape (B, C, H, W), with each value in {0,1}

    alpha: balancing factor for positive/negative examples
    gamma: focusing parameter to reduce loss contribution from easy examples
    reduction: 'none' | 'mean' | 'sum'
               - 'none': no reduction
               - 'mean': average over all pixels and channels in the batch
               - 'sum': sum over all pixels and channels in the batch
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', eps=1e-6):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps  # avoid log(0)

    def forward(self, logits, labels):
        """
        Compute the focal loss for multi-label (multi-channel) binary classification.
        Arguments:
            logits (Tensor): [B, C, H, W], raw output or logit from the network (not after sigmoid).
            labels (Tensor): [B, C, H, W], each entry is 0 or 1.

        Returns:
            loss (Tensor): focal loss value
        """
        # 1) Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)  # shape = [B, C, H, W]

        # 2) Compute pt = p if label=1, or (1-p) if label=0
        #    pt is the predicted probability of the "true" label
        pt = labels * probs + (1 - labels) * (1 - probs)  # shape = [B, C, H, W]

        # 3) Compute alpha factor
        #    alpha_t = alpha for ground truth = 1, else (1 - alpha) for ground truth = 0
        alpha_t = labels * self.alpha + (1 - labels) * (1 - self.alpha)  # [B, C, H, W]

        # 4) Compute focal weight = (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.gamma  # [B, C, H, W]

        # 5) Compute the final log of pt
        #    add a small eps for numerical stability
        log_pt = torch.log(pt.clamp(min=self.eps, max=1.0))

        # 6) Focal loss = - alpha_t * focal_weight * log_pt
        focal_loss = - alpha_t * focal_weight * log_pt  # [B, C, H, W]

        # 7) Apply reduction
        if self.reduction == 'none':
            return focal_loss
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # default 'mean'
            return focal_loss.mean()
