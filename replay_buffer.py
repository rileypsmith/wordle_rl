"""
Class for a replay buffer that can store information.

@author: Riley Smith
Created: 9-10-2022
"""

import numpy as np
import tensorflow as tf
from tqdm import tqdm

class ReplayBuffer():
    def __init__(self, random_seed=1234):
        # Blank list to populate with experience as it is collected
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.new_states = []

        # Seed random number generator for reproducibility
        self.rng = np.random.default_rng(random_seed)

    def store_experience(self, state, action, reward, done, new_state):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.new_states.append(new_state)

    def to_numpy(self):
        self.states = np.stack(self.states, axis=0)
        self.actions = np.array(self.actions)
        self.rewards = np.array(self.rewards)
        self.dones = np.array(self.dones)
        self.new_states = np.stack(self.new_states, axis=0)

    def sample(self, idx=None):
        return self.states[idx]

    def batch(self, batch_size, shuffle=True):
        # Make sure self.experience is an ndarray
        if not isinstance(self.states, np.ndarray):
            self.to_numpy()
        indices = np.arange(self.states.shape[0])
        if shuffle:
            # Randomize order of indices for batching
            self.rng.shuffle(indices)
        # Iterate over experience in batches
        for i in range(0, indices.size, batch_size):
            batch_indices = indices[i: i + batch_size]
            states = self.states[batch_indices]
            actions = self.actions[batch_indices]
            rewards = self.rewards[batch_indices]
            dones = self.dones[batch_indices]
            new_states = self.new_states[batch_indices]
            yield states, actions, rewards, dones, new_states

    def num_batches(self, batch_size):
        return int(np.ceil(len(self.states) / batch_size))

    def play_game(self, env, agent):
        """
        Use the given agent to play one game on the given env and store the
        result in the replay buffer.
        """
        done = False
        steps = []
        while not done:
            state = env.state
            action = agent.act(tf.convert_to_tensor(np.expand_dims(state, axis=0), dtype=tf.float32))
            new_state, reward, done, _ = env.step(action)
            self.store_experience(state, action, reward, done, new_state)

    def populate(self, env, agent, steps=1000):
        """Build, fill, and return an initial replay buffer"""
        # Populate it with steps
        print('Populating replay buffers')
        for _ in tqdm(range(steps)):
            self.play_game(env, agent)
            env.reset()
        self.to_numpy()

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.new_states = []
