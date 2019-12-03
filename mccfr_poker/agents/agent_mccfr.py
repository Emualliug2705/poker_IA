"""Player based on a trained neural network"""
import json
# pylint: disable=wrong-import-order
import logging
import time

import numpy as np
import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.models import model_from_json
from keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.core import Processor
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy

from gamegym.algorithms import (BestResponse, OutcomeMCCFR, RegretStrategy, approx_exploitability,
                                exploitability)
from gym_env.env import Action

autplay = True  # play automatically if played against keras-rl

window_length = 1
nb_max_start_steps = 20  # random action
train_interval = 100  # random action
nb_steps_warmup = 75  # before training starts, should be higher than start steps
nb_steps = 200000
memory_limit = int(nb_steps / 10)
batch_size = 500  # items sampled from memory to train
enable_double_dqn = False

log = logging.getLogger(__name__)


class Player:
    """Mandatory class with the player methods"""

    def __init__(self, name='MCCFR', seed=52, load_model=None, env=None):
        """Initiaization of an agent"""
        self.equity_alive = 0
        self.actions = []
        self.last_action_in_stage = ''
        self.temp_stack = []
        self.name = name
        self.autoplay = True
        self.seed = seed
        self.dqn = None
        self.model = None
        self.env = env
        self.mc = None

    def initiate_agent(self, env):
        self.env = env
#        nb_actions = self.env.action_space.n
        self.mc = OutcomeMCCFR(self.env, seed=52)

    def train(self, it, burn_lcfr=0.0):

        """Compute"""
        self.mc.compute(iterations=it, burn=burn_lcfr)

    def action(self, action_space, observation, info):  # pylint: disable=no-self-use
        """Mandatory method that calculates the move based on the observation array and the action space."""
        _ = observation  # not using the observation for random decision
        _ = info

        this_player_action_space = {Action.FOLD, Action.CHECK, Action.CALL, Action.RAISE_POT, Action.RAISE_HALF_POT,
                                    Action.RAISE_2POT}
        _ = this_player_action_space.intersection(set(action_space))

        action = None
        return action