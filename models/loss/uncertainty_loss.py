import numpy as np
import torch
import torch.nn.functional as F

def laplacian_aleatoric_uncertainty_loss_classification(values, log_variance, reduction='mean'):
    '''
    References:
        MonoPair: Monocular 3D Object Detection Using Pairwise Spatial Relationships, CVPR'20
        Geometry and Uncertainty in Deep Learning for Computer Vision, University of Cambridge
    '''
    assert reduction in ['mean', 'sum', 'None']
    loss = 1.4142 * torch.exp(-0.5*log_variance) * values + 0.5*log_variance  # F.smooth_l1_loss(input, target, reduction=reduction)
    if reduction == 'None':
        return loss
    return loss.mean() if reduction == 'mean' else loss.sum()

def laplacian_aleatoric_uncertainty_loss(input, target, log_variance, reduction='mean'):
    '''
    References:
        MonoPair: Monocular 3D Object Detection Using Pairwise Spatial Relationships, CVPR'20
        Geometry and Uncertainty in Deep Learning for Computer Vision, University of Cambridge
    '''
    assert reduction in ['mean', 'sum', 'None']
    loss = 1.4142 * torch.exp(-0.5*log_variance) * torch.abs(input - target) + 0.5*log_variance  # F.smooth_l1_loss(input, target, reduction=reduction)
    if reduction == 'None':
        return loss
    return loss.mean() if reduction == 'mean' else loss.sum()


def gaussian_aleatoric_uncertainty_loss(input, target, log_variance, reduction='mean'):
    '''
    References:
        What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?, Neuips'17
        Geometry and Uncertainty in Deep Learning for Computer Vision, University of Cambridge
    '''
    assert reduction in ['mean', 'sum', 'None']
    loss = 0.5 * torch.exp(-log_variance) * torch.abs(input - target)**2 + 0.5 * log_variance
    if reduction == 'None':
        return loss
    return loss.mean() if reduction == 'mean' else loss.sum()



if __name__ == '__main__':
    pass
