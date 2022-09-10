"""
Neural network agents to learn Wordle via deep Q-learning.

@author: Riley Smith
Created: 8-2-2022
"""
import tensorflow as tf
from tensorflow.keras import Model, layers

with open('words.txt', 'r') as fp:
    WORDS = [l.strip() for l in fp.readlines()]
NUM_WORDS = len(WORDS)

class LSTMAgent(Model):
    """A super simple agent for playing Wordle with RNN"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.rnn = layers.LSTM(256)
        self.dense = layers.Dense(NUM_WORDS, activation='sigmoid')

    def call(self, x):
        x = self.rnn(x)
        return self.dense(x)

    def train(self, X, y, optimizer, loss, **kwargs):
        """Run Tensorflow built-in training"""
        self.compile(optimizer, loss=loss)
        self.fit(X, y, **kwargs)
