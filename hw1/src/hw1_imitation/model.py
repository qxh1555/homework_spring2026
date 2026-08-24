"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from time import time
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        layers=[]
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(state_dim, hidden_dim))
            layers.append(nn.ReLU())
            state_dim = hidden_dim
        layers.append(nn.Linear(state_dim, chunk_size * action_dim))
        self.model = nn.Sequential(*layers)
            




    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        predicted_actions = self.model(state)
        return nn.functional.mse_loss(predicted_actions, action_chunk.view(predicted_actions.shape))

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        with torch.no_grad():
            predicted_actions = self.model(state)
        return predicted_actions.view(-1, self.chunk_size, self.action_dim)


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        layers = []
        for i, hidden_dim in enumerate(hidden_dims):
            if i == 0:
                layers.append(nn.Linear(state_dim + action_dim * chunk_size +1, hidden_dim))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(state_dim, hidden_dim))
                layers.append(nn.ReLU())
            state_dim = hidden_dim
        layers.append(nn.Linear(state_dim, chunk_size * action_dim))
        self.model = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = state.shape[0]
        t = torch.rand((batch_size, 1,1), device=state.device)
        noise = torch.randn_like(action_chunk)
        target_velocity = (action_chunk - noise)
        noisy_actions =(t  * action_chunk + noise * (1 - t)).reshape(state.shape[0], -1)
        input = torch.cat([state, noisy_actions,t.reshape(batch_size,-1)], dim=1)
        predicted_actions = self.model(input)
        return nn.functional.mse_loss(predicted_actions,  target_velocity.reshape(predicted_actions.shape))



    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        
        with torch.no_grad():
            time_vec = torch.full((state.shape[0], 1), 0.0, device=state.device)
            predicted_actions = torch.randn((state.shape[0], self.chunk_size * self.action_dim), device=state.device)
            for step in range(num_steps):
                input = torch.cat([state, predicted_actions,time_vec], dim=1)
                predicted_actions +=1/ num_steps * self.model(input)
                time_vec += 1/num_steps
        return predicted_actions.view(-1, self.chunk_size, self.action_dim)
                


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
