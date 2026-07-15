"""
MLP baseline for supervised learning on A(t) -> A(t+1).

This module contains only the model definition and model-local utilities.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPFCM(nn.Module):
    """
    Multilayer Perceptron baseline for predicting A(t+1) from A(t).

    Hidden layers use ReLU activations. The output uses tanh so predictions
    stay in [-1, 1], matching the activation range used by the classical FCM
    and QFCM models.

    The MLP has no interpretable causal matrix: its first linear layer weight
    is exposed only for diagnostics, not as an FCM-equivalent ``W``.
    """

    def __init__(
        self,
        n_concepts: int,
        hidden_dims: tuple[int, ...] = (16,),
        lambda_l1: float = 1e-4,
        use_bias: bool = False,
        seed: int = 42,
    ):
        super().__init__()
        torch.manual_seed(seed)

        self.n_concepts = n_concepts
        self.hidden_dims = tuple(hidden_dims)
        self.lambda_l1 = lambda_l1
        self.use_bias = use_bias

        dims = [n_concepts, *self.hidden_dims, n_concepts]
        layers: list[nn.Module] = []

        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=use_bias))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(dims[-2], dims[-1], bias=use_bias))
        self.net = nn.Sequential(*layers)

    def forward(self, A_t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a batch or a single sample.

        Parameters
        ----------
        A_t : torch.Tensor
            Input tensor with shape (batch, n_concepts), or (n_concepts,)
            for a single sample.

        Returns
        -------
        torch.Tensor
            Output tensor with the same leading shape as the input.
        """
        return torch.tanh(self.net(A_t))

    def compute_loss(
        self,
        A_t: torch.Tensor,
        A_t1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute total loss = MSE + L1(weight norm).

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

        weight_params = [p for name, p in self.named_parameters() if "bias" not in name]
        n_terms = sum(p.numel() for p in weight_params)
        l1_reg = self.lambda_l1 * sum(p.abs().sum() for p in weight_params) / n_terms

        total_loss = mse + l1_reg
        return total_loss, mse, l1_reg

    @property
    def n_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def first_layer_weight(self) -> torch.Tensor:
        """
        Return the first linear layer weight matrix.

        This is useful only for diagnostics; it is not an FCM causal matrix.
        """
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                return layer.weight
        raise RuntimeError("No linear layer found in MLP network.")
