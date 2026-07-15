"""
Quantum Fuzzy Cognitive Map (QFCM) model.

This module contains only the model definition: the parametrized quantum
circuit, its forward pass, and its training loss. Recovering an effective
classical FCM matrix from a trained model is a separate concern and lives
in :mod:`hqcfcm.postprocessing.w_recovery`, so that changes to the recovery
algorithm never require touching the model class (and vice versa).
"""

from __future__ import annotations

import pennylane as qml
import torch
import torch.nn as nn

ALLOWED_ENTANGLEMENT = {"none", "chain_cz", "ring_cz"}


def code_coords(
    theta: torch.Tensor,
    alpha: torch.Tensor,
    A_t: torch.Tensor,
) -> torch.Tensor:
    """
    Build effective rotation angles for a single sample.

    Parameters
    ----------
    theta : torch.Tensor
        Base angles with shape (n_concepts, n_layers, 3).
    alpha : torch.Tensor
        Encoding weights with shape (n_concepts, n_layers, 3, n_concepts).
    A_t : torch.Tensor
        Input state vector with shape (n_concepts,).

    Returns
    -------
    torch.Tensor
        Effective angles with shape (n_concepts, n_layers, 3).
    """
    shift = torch.einsum("qlkj,j->qlk", alpha, A_t)
    return theta + shift


def code_coords_batch(
    theta: torch.Tensor,
    alpha: torch.Tensor,
    A_t: torch.Tensor,
) -> torch.Tensor:
    """
    Build effective rotation angles for a batch.

    Parameters
    ----------
    theta : torch.Tensor
        Base angles with shape (n_concepts, n_layers, 3).
    alpha : torch.Tensor
        Encoding weights with shape (n_concepts, n_layers, 3, n_concepts).
    A_t : torch.Tensor
        Batch input with shape (batch, n_concepts).

    Returns
    -------
    torch.Tensor
        Effective angles with shape (batch, n_concepts, n_layers, 3).

    Notes
    -----
    The leading batch dimension is used by PennyLane parameter broadcasting.
    """
    shift = torch.einsum("qlkj,bj->bqlk", alpha, A_t)
    return theta.unsqueeze(0) + shift


class QFCM(nn.Module):
    """
    Quantum FCM with direct data re-uploading on A(t) and optional CZ entanglement.

    Each concept is encoded on one qubit. At every layer, the base rotation
    angles ``theta`` are shifted by a linear, learnable function of the full
    concept vector ``A(t)`` (the "data re-uploading" mechanism), masked to
    forbid a concept from directly influencing its own rotation. The
    expectation values of Pauli-Z on each wire are read out as the predicted
    ``A(t+1)``.
    """

    def __init__(
        self,
        n_concepts: int,
        n_layers: int,
        lambda_l1: float = 1e-4,
        seed: int = 42,
        device_name: str = "lightning.qubit",
        diff_method: str = "best",
        entanglement: str = "none",
    ):
        super().__init__()
        torch.manual_seed(seed)

        if entanglement not in ALLOWED_ENTANGLEMENT:
            raise ValueError(
                f"entanglement must be one of {ALLOWED_ENTANGLEMENT}, "
                f"got {entanglement!r}"
            )

        self.n_concepts = n_concepts
        self.n_layers = n_layers
        self.lambda_l1 = lambda_l1
        self.device_name = device_name
        self.diff_method = diff_method
        self.entanglement = entanglement

        dev = qml.device(device_name, wires=n_concepts, shots=None)

        def apply_entanglement_layer() -> None:
            """Apply the fixed CZ entanglement pattern selected for the model."""
            if n_concepts < 2 or entanglement == "none":
                return

            for q in range(n_concepts - 1):
                qml.CZ(wires=[q, q + 1])

            # Close the ring only when it adds a genuinely new edge.
            if entanglement == "ring_cz" and n_concepts > 2:
                qml.CZ(wires=[n_concepts - 1, 0])

        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def circuit(theta_aux: torch.Tensor):
            """
            Quantum circuit for either a single sample or a batch.

            ``theta_aux`` shape:
            - (n_concepts, n_layers, 3) for a single sample
            - (batch, n_concepts, n_layers, 3) for a batch
            """
            is_batched = theta_aux.ndim == 4

            for layer_idx in range(n_layers):
                for q in range(n_concepts):
                    # In batch mode, PennyLane broadcasts the leading dimension.
                    if is_batched:
                        qml.Rot(
                            theta_aux[:, q, layer_idx, 0],
                            theta_aux[:, q, layer_idx, 1],
                            theta_aux[:, q, layer_idx, 2],
                            wires=q,
                        )
                    else:
                        qml.Rot(
                            theta_aux[q, layer_idx, 0],
                            theta_aux[q, layer_idx, 1],
                            theta_aux[q, layer_idx, 2],
                            wires=q,
                        )

                apply_entanglement_layer()

            return [qml.expval(qml.PauliZ(w)) for w in range(n_concepts)]

        self.circuit = circuit

        self.theta = nn.Parameter(
            torch.rand(n_concepts, n_layers, 3) * 2.0 * torch.pi
        )
        self.alpha = nn.Parameter(
            torch.randn(n_concepts, n_layers, 3, n_concepts) * 0.01
        )

        # Mask out explicit self-causality terms alpha[q, ..., q].
        mask = torch.ones(n_concepts, n_layers, 3, n_concepts)
        for q in range(n_concepts):
            mask[q, :, :, q] = 0.0
        self.register_buffer("mask", mask)

    @property
    def alpha_masked(self) -> torch.Tensor:
        """Return alpha with zeroed self-causality entries."""
        return self.alpha * self.mask

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
        if A_t.ndim != 1 or A_t.shape[0] != self.n_concepts:
            raise ValueError(
                f"Expected A_t with shape ({self.n_concepts},), got {tuple(A_t.shape)}"
            )

        theta_aux = code_coords(self.theta, self.alpha_masked, A_t)
        out = self.circuit(theta_aux)
        return torch.stack(out)

    def forward(self, A_t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for either one sample or a batch.

        Parameters
        ----------
        A_t : torch.Tensor
            Input tensor with shape (n_concepts,) or (batch, n_concepts).

        Returns
        -------
        torch.Tensor
            Output tensor with shape (n_concepts,) or (batch, n_concepts).
        """
        if A_t.ndim == 1:
            return self.forward_single(A_t)

        if A_t.ndim != 2 or A_t.shape[1] != self.n_concepts:
            raise ValueError(
                f"Expected A_t with shape (batch, {self.n_concepts}), "
                f"got {tuple(A_t.shape)}"
            )

        theta_aux = code_coords_batch(self.theta, self.alpha_masked, A_t)
        out = self.circuit(theta_aux)

        if isinstance(out, (list, tuple)):
            return torch.stack(out, dim=-1)

        return out

    def compute_loss(
        self,
        A_t: torch.Tensor,
        A_t1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute total loss = MSE + L1(alpha_masked)."""
        pred = self.forward(A_t)
        mse = nn.functional.mse_loss(pred, A_t1)
        n_terms = self.n_concepts * self.n_layers * 3 * (self.n_concepts - 1)
        l1_reg = self.lambda_l1 * self.alpha_masked.abs().sum() / n_terms
        total_loss = mse + l1_reg
        return total_loss, mse, l1_reg

    @property
    def n_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
