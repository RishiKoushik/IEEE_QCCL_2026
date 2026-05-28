import torch


def flatten_vectors(X):
    """
    X: torch.Tensor [N, 1, 32, 32]
    Returns: [N, 1024]
    """
    return X.view(X.size(0), -1)


def l2_normalize(X, eps=1e-12):
    """
    Normalize each sample to ||x||_2 = 1
    """
    norms = torch.norm(X, p=2, dim=1, keepdim=True)
    return X / (norms + eps)
