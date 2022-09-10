"""
Neural network agents to learn Wordle via deep Q-learning.

@author: Riley Smith
Created: 8-2-2022
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers

import utils

with open('words.txt', 'r') as fp:
    WORDS = [l.strip() for l in fp.readlines()]
NUM_WORDS = len(WORDS)

# Get embeddings of words as one-hot vectors
EMBEDDINGS = np.stack([utils.encode_word(word) for word in WORDS], axis=0)
EMBEDDINGS = tf.convert_to_tensor(EMBEDDINGS, dtype=tf.float32)

class LSTMAgent(Model):
    """A super simple agent for playing Wordle with RNN"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.rnn = layers.LSTM(256)
        self.hidden = layers.Dense(200, activation='relu')
        self.top = layers.Dense(130, activation=None)

    def call(self, x):
        x = self.rnn(x)
        x = self.hidden(x)
        x = self.top(x)
        return tf.linalg.matmul(x, tf.transpose(EMBEDDINGS))

    def act(self, state):
        """Apply forward pass and use outcome to select an action"""
        pred = self(state).numpy().ravel()
        return np.argmax(pred)

    def train(self, X, y, optimizer, loss, **kwargs):
        """Run Tensorflow built-in training"""
        self.compile(optimizer, loss=loss)
        self.fit(X, y, **kwargs)
