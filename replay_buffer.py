"""
Play the game of Wordle a bunch and store a replay buffer.

@author: Riley Smith
Created: 8-2-2022
"""

import numpy as np

from wordle_env import WordleEnv
from agents import RNNAgent

def train():
    # Build target and agent network
    t_net = RNNAgent()
    q_net = RNNAgent()

    env = WordleEnv()
    for i in range(tqdm(100)):
        # Get to random starting state
        guesses_made = np.random.randint(low=0, high=6)
        for _ in range(guesses_made):
            env.step(env.action_space.sample())
