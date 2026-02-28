import torch
import torch.nn as nn
import torch.nn.functional as F


class VotingLayer(nn.Module):
    """A layer that performs voting by averaging groups of elements along the last dimension.

    This layer reshapes the input tensor along the last dimension into groups of size
    `voting_size` and computes the mean for each group. It optionally enforces that the
    last dimension must be strictly divisible by `voting_size`.

    Attributes:
        voting_size (int): The size of the group to average over. Defaults to 10.
        strict (bool): If True, raises an error if the last dimension is not divisible
            by `voting_size`. If False, truncates the excess elements. Defaults to True.
    """

    def __init__(self, voting_size: int = 10, strict: bool = True) -> None:
        """Initializes the VotingLayer.

        Args:
            voting_size (int): The number of elements to group together for voting.
            strict (bool): Whether to enforce strict divisibility of the last dimension.
        """
        super().__init__()
        self.voting_size = voting_size
        self.strict = strict

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the voting operation on the input tensor.

        Reshapes the last dimension of the input into chunks of `voting_size` and
        calculates the mean across these chunks. If `strict` is False, the trailing
        elements that do not form a complete chunk are discarded.

        Args:
            x (torch.Tensor): Input tensor of shape [..., L], where L is the length
                of the last dimension.

        Returns:
            Output tensor of shape [..., L // voting_size], containing
                the averaged values.

        Raises:
            ValueError: If `strict` is True and the last dimension of `x` is not
                divisible by `voting_size`.
        """
        v = self.voting_size
        L = x.size(-1)
        if self.strict and L % v != 0:
            raise ValueError(f"last dim ({L}) must be divisible by voting_size ({v})")
        L = (L // v) * v
        x = x[..., :L]
        return x.reshape(*x.shape[:-1], L // v, v).mean(dim=-1)


class ClassificationHead(nn.Module):
    """A classification head that averages embeddings over patches and time steps.

    This module takes a multi-dimensional input tensor representing embeddings over
    time and patches, averages them to produce a single representation per batch item,
    and then applies a linear transformation to generate class logits.

    Attributes:
        embed_dims (int): The dimensionality of the input embeddings.
        num_classes (int): The number of output classes.
        head (nn.Module): The linear layer for classification, or an Identity layer
            if `num_classes` is 0 or less.
    """

    def __init__(self, embed_dims: int, num_classes: int) -> None:
        """Initializes the ClassificationHead.

        Args:
            embed_dims (int): The number of features in the input embedding.
            num_classes (int): The number of target classes for classification.
        """
        super().__init__()
        self.embed_dims = embed_dims
        self.num_classes = num_classes
        self.head = (
            nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Processes the input tensor to produce classification logits.

        The input is expected to have the shape [T, B, embed_dims, num_patches].
        The method first averages over the patch dimension, then applies the linear
        classification head, and finally averages over the time dimension.

        Args:
            x (torch.Tensor): Input tensor of shape [T, B, embed_dims, num_patches],
                where:
                - T: Time steps
                - B: Batch size
                - embed_dims: Embedding dimension
                - num_patches: Number of patches

        Returns:
            Output tensor of shape [B, num_classes] (or [B, embed_dims]
                if num_classes <= 0), representing the classification scores.
        """
        x = x.mean(-1)
        x = self.head(x)
        x = x.mean(0)
        return x
