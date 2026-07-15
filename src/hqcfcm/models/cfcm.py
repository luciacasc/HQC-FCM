"""
Classical FCM model for supervised learning on A(t) -> A(t+1).

This module contains only the model definition and model-local utilities.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ClassicalFCM(nn.Module):
    """
    Classical Fuzzy Cognitive Map with supervised learning.

    The model predicts::

        A(t+1) = tanh(A(t) @ W^T + b)

    where:
    - W is the learned causal matrix
    - b is an optional bias term
    - tanh keeps outputs in [-1, 1]
    """

    def __init__(
        self,
        n_concepts: int,
        lambda_l1: float = 1e-4,
        use_bias: bool = False,
        seed: int = 42,
    ):
        super().__init__()
        torch.manual_seed(seed)

        self.n_concepts = n_concepts
        self.lambda_l1 = lambda_l1
        self.use_bias = use_bias

        # W[i, j] = influence from concept j at time t
        #           to concept i at time t+1
        self.W = nn.Parameter(torch.randn(n_concepts, n_concepts) * 0.05)

        if use_bias:
            self.b = nn.Parameter(torch.zeros(n_concepts))
        else:
            self.register_buffer("b", torch.zeros(n_concepts))

        mask = torch.ones(n_concepts, n_concepts)
        for i in range(n_concepts):
            mask[i, i] = 0.0
        self.register_buffer("mask", mask)

    @property
    def W_masked(self) -> torch.Tensor:
        """Return W with zero diagonal, enforcing no self-causality."""
        return self.W * self.mask

    def forward_single(self, A_t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a single sample.

        Parameters
        ----------
        A_t : torch.Tensor
            Input vector with shape (n_concepts,).

        Returns
        -------
        torch.Tensor
            Output vector with shape (n_concepts,).
        """
        logits = A_t @ self.W_masked.T + self.b
        return torch.tanh(logits)

    def forward(self, A_t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a batch (or a single sample).

        Parameters
        ----------
        A_t : torch.Tensor
            Input tensor with shape (batch, n_concepts) or (n_concepts,).

        Returns
        -------
        torch.Tensor
            Output tensor with the same leading shape as the input.
        """
        if A_t.ndim == 1:
            return self.forward_single(A_t)

        logits = A_t @ self.W_masked.T + self.b
        return torch.tanh(logits)

    def compute_loss(
        self,
        A_t: torch.Tensor,
        A_t1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute total loss = MSE + L1(W_masked).

        Parameters
        ----------
        A_t : torch.Tensor
            Inputs with shape (batch, n_concepts).
        A_t1 : torch.Tensor
            Targets with shape (batch, n_concepts).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            total_loss, mse_loss, l1_reg
        """
        pred = self.forward(A_t)
        mse = nn.functional.mse_loss(pred, A_t1)
        n_terms = self.n_concepts * (self.n_concepts - 1)
        l1_reg = self.lambda_l1 * self.W_masked.abs().sum() / n_terms
        total_loss = mse + l1_reg
        return total_loss, mse, l1_reg

    @property
    def n_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
