import torch

def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = 0, dim_size: int = None):
    """
    Summarizes `src` tensor values into `dim_size` buckets using the `index` tensor.

    Args:
        src (Tensor): Values to scatter.
        index (LongTensor): Indices to scatter to.
        dim (int): Dimension along which to scatter.
        dim_size (int): Size of the output tensor along `dim`. If None, inferred.

    Returns:
        Tensor: Summed tensor of shape like `src`, but with `dim` replaced by `dim_size`.
    """
    if dim_size is None:
        dim_size = index.max().item() + 1

    out_shape = list(src.shape)
    out_shape[dim] = dim_size
    out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)

    return out.index_add(dim, index, src)

def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = 0, dim_size: int = None):
    """
    Computes mean over values in `src` tensor grouped by `index`.

    Args:
        src (Tensor): Values to scatter.
        index (LongTensor): Indices to scatter to.
        dim (int): Dimension along which to scatter.
        dim_size (int): Size of the output tensor along `dim`. If None, inferred.

    Returns:
        Tensor: Mean tensor of shape like `src`, but with `dim` replaced by `dim_size`.
    """
    sum_out = scatter_sum(src, index, dim=dim, dim_size=dim_size)

    ones = torch.ones_like(src)
    count = scatter_sum(ones, index, dim=dim, dim_size=dim_size)
    count = torch.clamp(count, min=1)  # Avoid division by zero

    return sum_out / count