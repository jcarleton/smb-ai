from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from gym import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
import torch
from torch import nn

class CustomNetwork(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Using SELU to prevent dead neurons
        # testing 6 hidden layer depth
        # Policy network
        self.policy_net = nn.Sequential(
            # nn.Linear(feature_dim, last_layer_dim_pi), nn.ReLU()
            nn.Linear(feature_dim, last_layer_dim_pi),
            nn.SELU(),
            nn.Linear(last_layer_dim_pi, last_layer_dim_pi),
            nn.SELU(),
            nn.Linear(last_layer_dim_pi, last_layer_dim_pi),
            nn.SELU(),
            nn.Flatten(), #new
            nn.Linear(last_layer_dim_pi, last_layer_dim_pi),
            nn.SELU()
        )
        # Value network
        self.value_net = nn.Sequential(
            # nn.Linear(feature_dim, last_layer_dim_vf), nn.ReLU()
            nn.Linear(feature_dim, last_layer_dim_vf),
            nn.SELU(),
            nn.Linear(last_layer_dim_vf, last_layer_dim_vf),
            nn.SELU(),
            nn.Linear(last_layer_dim_vf, last_layer_dim_vf),
            nn.SELU(),
            nn.Flatten(), #new
            nn.Linear(last_layer_dim_vf, last_layer_dim_vf),
            nn.SELU()
            # nn.CELU(alpha=1.0)
        )


    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (torch.Tensor, torch.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)

class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)
