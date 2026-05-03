# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import get_class_weight, weight_reduce_loss
import numpy as np

def cross_entropy(pred,
                  label,
                  weight=None,
                  class_weight=None,
                  reduction='mean',
                  avg_factor=None,
                  ignore_index=-100):
    """The wrapper function for :func:`F.cross_entropy`"""
    # class_weight is a manual rescaling weight given to each class.
    # If given, has to be a Tensor of size C element-wise losses
    loss = F.cross_entropy(
        pred,
        label,
        weight=class_weight,
        reduction='none',
        ignore_index=ignore_index)

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def _expand_onehot_labels(labels, label_weights, target_shape, ignore_index):
    """Expand onehot labels to match the size of prediction."""
    bin_labels = labels.new_zeros(target_shape)
    valid_mask = (labels >= 0) & (labels != ignore_index)
    inds = torch.nonzero(valid_mask, as_tuple=True)

    if inds[0].numel() > 0:
        if labels.dim() == 3:
            bin_labels[inds[0], labels[valid_mask], inds[1], inds[2]] = 1
        else:
            bin_labels[inds[0], labels[valid_mask]] = 1

    valid_mask = valid_mask.unsqueeze(1).expand(target_shape).float()
    if label_weights is None:
        bin_label_weights = valid_mask
    else:
        bin_label_weights = label_weights.unsqueeze(1).expand(target_shape)
        bin_label_weights *= valid_mask

    return bin_labels, bin_label_weights


def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         class_weight=None,
                         ignore_index=255):
    """Calculate the binary CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, 1).
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (int | None): The label index to be ignored. Default: 255

    Returns:
        torch.Tensor: The calculated loss
    """
    if pred.dim() != label.dim():
        assert (pred.dim() == 2 and label.dim() == 1) or (
                pred.dim() == 4 and label.dim() == 3), \
            'Only pred shape [N, C], label shape [N] or pred shape [N, C, ' \
            'H, W], label shape [N, H, W] are supported'
        label, weight = _expand_onehot_labels(label, weight, pred.shape,
                                              ignore_index)

    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()
    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), pos_weight=class_weight, reduction='none')
    # do the reduction for the weighted loss
    loss = weight_reduce_loss(
        loss, weight, reduction=reduction, avg_factor=avg_factor)

    return loss

def _expand_ecoc_labels(labels,code_book,ignore_index=255):
    # labels n,h,w
    n = len(code_book) #类别数
    k = len(code_book[0]) #码长
    code_book_tensor = torch.tensor(code_book, device = labels.device)
    bin_labels = labels.new_zeros([labels.shape[0],k,labels.shape[1],labels.shape[2]])
    valid_mask = (labels >= 0) & (labels != ignore_index)
    inds = torch.nonzero(valid_mask, as_tuple=True)
  
    if inds[0].numel() > 0:
        bin_labels[inds[0], :, inds[1], inds[2]] = code_book_tensor[labels[valid_mask]]
    return bin_labels, valid_mask  #b,k,h,w


def logsumexp(x,dim,weight = None):
  b = torch.max(x)
  if weight is not None:
    return b + torch.log(torch.sum(torch.exp(x - b)*weight,dim=dim)+1e-3)
  else:
    return b + torch.log(torch.sum(torch.exp(x - b),dim=dim)+1e-3)  

def ecoc_entropy(pred,
                  label,
                  code_book,
                  weight=None,
                  dice = None,
                  class_weight=None,
                  reduction='mean',
                  avg_factor=None,
                  ignore_index=-100):
    """The wrapper function for :func:`F.cross_entropy`"""
    # class_weight is a manual rescaling weight given to each class.
    # If given, has to be a Tensor of size C element-wise losses
    # pred = pred / pred.norm(p=2, dim=1, keepdim=True)/0.05
    prob = torch.sigmoid(pred) # b,k,h,w
    bin_labels, valid_mask =  _expand_ecoc_labels(label,code_book)
    
   
    error_sum = torch.tensor([0])
    correct_sum = torch.tensor([0])
    if weight != None:
        label_target = weight[1]
        mask_target = weight[2]
        
        
        error_sum = torch.sum(label_target!=bin_labels)
        correct_sum = torch.sum(mask_target*(label_target!=bin_labels))
        #uda3
        label_target = mask_target*bin_labels+(1-mask_target)*label_target
        
        weight = weight[0]* valid_mask.float()
        weight = weight.unsqueeze(1)
        
        bin_labels = (weight==1).float()*bin_labels+(weight<1).float()*label_target
        
    else:
        weight = valid_mask.float()
        weight = weight.unsqueeze(1)
    
    loss = -weight*(bin_labels*torch.log(prob+1e-4)+(1-bin_labels)*torch.log(1-prob+1e-4))
    loss = loss.mean()
    # return loss
    
    
    # pred = torch.tanh(pred)
    
    
    code_book_tensor = torch.tensor(code_book, dtype=torch.float16 , device = label.device) #n k
    code_book_tensor -= (code_book_tensor==0).float()
    code_book_tensor = code_book_tensor.unsqueeze(0).unsqueeze(3).unsqueeze(4)
    bin_labels = bin_labels-(bin_labels==0).float()
    
    
    tau = 0.5
    
    pred_ = F.cosine_similarity(pred.unsqueeze(1),code_book_tensor,2)/tau

    loss_cls = -weight.squeeze(1)*torch.log(torch.exp(F.cosine_similarity(pred,bin_labels,1)/tau)/torch.sum(torch.exp(pred_),1))
    loss_cls = loss_cls.mean()


    loss_cos = weight.squeeze(1)*(1-F.cosine_similarity(pred,bin_labels,1))
    loss_cos = loss_cos.mean()
    
    l1 = 5
    l2 = 2
    
    return [l1*loss_cos,l2*loss_cls,loss,error_sum.float(),correct_sum.float()]

class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()
    
    def forward(self, inputs, targets,weight):
        weight = weight.expand(-1,targets.shape[1] ,  -1, -1) #b,k,h,w
        # 获取每个批次的大小 N
        N = targets.shape[0]
        # 平滑变量
        smooth = 1
        # 将宽高 reshape 到同一纬度
        # print(N,inputs.shape,targets.shape,valid_mask.shape,weight.shape)
        weight_flat = weight.contiguous().view(N, -1)
        input_flat = inputs.contiguous().view(N, -1) * weight_flat
        targets_flat = targets.contiguous().view(N, -1) * weight_flat
        
     
        # 计算交集
        intersection = input_flat * targets_flat 
        N_dice_eff = (2 * intersection.sum(1) + smooth) / ((input_flat).sum(1) + (targets_flat).sum(1) + smooth)
        # 计算一个批次中平均每张图的损失
        loss = 1 - N_dice_eff.sum() / N
      
        return loss


def mask_cross_entropy(pred,
                       target,
                       label,
                       reduction='mean',
                       avg_factor=None,
                       class_weight=None,
                       ignore_index=None):
    """Calculate the CrossEntropy loss for masks.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        target (torch.Tensor): The learning label of the prediction.
        label (torch.Tensor): ``label`` indicates the class label of the mask'
            corresponding object. This will be used to select the mask in the
            of the class which the object belongs to when the mask prediction
            if not class-agnostic.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (None): Placeholder, to be consistent with other loss.
            Default: None.

    Returns:
        torch.Tensor: The calculated loss
    """
    assert ignore_index is None, 'BCE loss does not support ignore_index'
    # TODO: handle these two reserved arguments
    assert reduction == 'mean' and avg_factor is None
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    return F.binary_cross_entropy_with_logits(
        pred_slice, target, weight=class_weight, reduction='mean')[None]


@LOSSES.register_module()
class CrossEntropyLoss(nn.Module):
    """CrossEntropyLoss.

    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0):
        super(CrossEntropyLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = get_class_weight(class_weight)

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        ecoc_entropy(cls_score, _expand_ecoc_labels(label))
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls


@LOSSES.register_module()
class CrossEntropyLoss_ecoc(nn.Module):
    """CrossEntropyLoss.

    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 code_book = None):
        super(CrossEntropyLoss_ecoc, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = get_class_weight(class_weight)

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy
        self.code_book = code_book
        self.dice = BinaryDiceLoss()
        
    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')

        
        return ecoc_entropy(cls_score, label,self.code_book,weight,self.dice)
