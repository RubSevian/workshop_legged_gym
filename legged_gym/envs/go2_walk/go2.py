from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils.isaacgym_utils import get_euler_xyz
from legged_gym.envs import LeggedRobot
from isaacgym import gymtorch, gymapi
from legged_gym.utils.helpers import class_to_dict
import numpy as np
import os
import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym.envs.go2_walk.go2_config import Go2_Walk_Cfg

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils.isaacgym_utils import get_euler_xyz
from legged_gym.envs import LeggedRobot
from isaacgym import gymtorch, gymapi

import torch

class Go2_Walk(LeggedRobot):
    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((  self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec