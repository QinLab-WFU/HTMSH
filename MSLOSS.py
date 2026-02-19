import torch
import torch.nn as nn
from torch.nn.functional import normalize
class MultiSimilarityLoss(nn.Module):
    def __init__(self,thresh,margin,scale_pos,scale_neg):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = thresh
        self.margin = margin

        self.scale_pos = scale_pos
        self.scale_neg = scale_neg

    def forward(self, feats, labels, dataset="MSLOSS", feat2=None):
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        batch_size = feats.size(0)
        if feat2 == None:
            # 自相似度矩阵
            sim_mat = normalize(torch.matmul(feats, torch.t(feats)))
        else:
            # feats * feat2
            sim_mat = normalize(torch.matmul(feats,torch.t(feat2)))

        # 标签处理
        if dataset == "cifar10-1":
            labels = torch.argmax(labels , dim=1)
        else:
            labels = labels @ labels.t() > 0

        epsilon = 1e-5
        loss = list()

        for i in range(batch_size):
            # 筛选出相似性值小于 1 - epsilon 的正样本对。
            # 去除相似性值接近于1的样本对，因为这些对可能是自相似（即同一个样本与自身的相似性为1）
            pos_pair_ = sim_mat[i][labels == labels[i] if dataset =='cifar10-1' else labels[i]]
            pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]

            # 筛选出与第 i 个样本标签不同的样本对的相似性值
            neg_pair_ = sim_mat[i][labels != labels[i] if dataset =='cifar10-1' else (labels[i] == False)]

            if torch.numel(pos_pair_) == 0 or torch.numel(neg_pair_) == 0:
                continue
            # 筛选出那些加上边距后仍然大于最小正样本对相似性值的负样本对
            neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_) ]
            # 筛选出那些减去边距后仍然小于最大负样本对相似性值的正样本对
            pos_pair = pos_pair_[pos_pair_ - self.margin < max(neg_pair_) ]

            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue
                    
            # weighting step
            pos_loss = 1.0 / self.scale_pos * torch.log(
                1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh))))
            neg_loss = 1.0 / self.scale_neg * torch.log(
                1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh))))
            loss.append(pos_loss + neg_loss)

        if len(loss) == 0:
            return torch.zeros([], requires_grad=True)

        loss = sum(loss) / batch_size
        return loss