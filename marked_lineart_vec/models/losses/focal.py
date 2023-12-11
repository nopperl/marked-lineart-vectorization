import torch
import torch.nn as nn

def binary_focal_loss_with_logits(
    input: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = 'none',
    eps: float = 1e-8,
) -> torch.Tensor:
    r"""Function that computes Binary Focal loss.
    .. math::
        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)
    where:
       - :math:`p_t` is the model's estimated probability for each class.
    Args:
        input: input data tensor with shape :math:`(N, 1, *)`.
        target: the target tensor with shape :math:`(N, 1, *)`.
        alpha: Weighting factor for the rare class :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        eps: for numerically stability when dividing.
    Returns:
        the computed loss.
    Examples:
        >>> num_classes = 1
        >>> kwargs = {"alpha": 0.25, "gamma": 2.0, "reduction": 'mean'}
        >>> logits = torch.tensor([[[[6.325]]],[[[5.26]]],[[[87.49]]]])
        >>> labels = torch.tensor([[[1.]],[[1.]],[[0.]]])
        >>> binary_focal_loss_with_logits(logits, labels, **kwargs)
        tensor(4.6052)
    """
    input = 1 - input.mean(dim=1)
    target = 1 - target.mean(dim=1)
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(input)))

    if not len(input.shape) >= 2:
        raise ValueError("Invalid input shape, we expect BxCx*. Got: {}".format(input.shape))

    if input.size(0) != target.size(0):
        raise ValueError(
            'Expected input batch_size ({}) to match target batch_size ({}).'.format(input.size(0), target.size(0))
        )

    probs = torch.sigmoid(input)
    target = target.unsqueeze(dim=1)
    loss_tmp = -alpha * torch.pow((1.0 - probs + eps), gamma) * target * torch.log(probs + eps) - (
        1 - alpha
    ) * torch.pow(probs + eps, gamma) * (1.0 - target) * torch.log(1.0 - probs + eps)

    loss_tmp = loss_tmp.squeeze(dim=1)

    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError("Invalid reduction mode: {}".format(reduction))
    return loss