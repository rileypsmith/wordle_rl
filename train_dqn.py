"""
Play the game of Wordle a bunch and store a replay buffer.

@author: Riley Smith
Created: 8-2-2022
"""
import copy
import csv
from pathlib import Path
import time

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from wordle_env import WordleEnv
from nn_agents import LSTMAgent
from replay_buffer import ReplayBuffer
import utils

NUM_WORDS = 216

def make_logfile(logfile):
    with open(logfile, 'w+', newline='') as csvfile:
        writer = csv.writer(csvfile, skipinitialspace=True)
        writer.writerow(['Epoch', 'Loss', 'Time'])

def update_logfile(logfile, epoch, loss, time):
    with open(logfile, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile, skipinitialspace=True)
        writer.writerow([epoch, loss, time])

def train(out_dir, epochs=50, buffer_steps=100, easy=True):
    # Setup output directory
    out_dir = utils.setup_output_directory(out_dir)

    # Make subdirectory for saving weights
    weight_dir = Path(out_dir, 'weights')
    weight_dir.mkdir()

    # Also start a CSV logfile
    logfile = str(Path(out_dir, 'training_log.csv'))
    make_logfile(logfile)

    # Build target and agent network
    t_net = LSTMAgent()
    q_net = LSTMAgent()

    # Training constants
    gamma = 0.5
    opt = tf.keras.optimizers.Adam()

    # Build Wordle ENV
    env = WordleEnv(easy=easy)

    # Make and populate initial replay buffer
    buffer = ReplayBuffer()
    buffer.populate(env, q_net, steps=buffer_steps)

    best_loss = None
    start = time.time()
    for epoch in range(epochs):
        print(f'EPOCH {epoch+1}/{epochs}')
        avg_loss = do_epoch(q_net, t_net, buffer, gamma, opt)
        print('AVG LOSS: ', avg_loss)

        # Reset and repopulate buffer
        buffer.reset()
        buffer.populate(env, q_net, buffer_steps)

        # Save weights of best network
        if (best_loss is None) or (avg_loss < best_loss):
            # Save weights of network
            q_net.save_weights(str(Path(weight_dir, 'checkpoint')))
            best_loss = avg_loss

        # Print progres to CSV file
        elapsed = round((time.time() - start) / 60, 2)
        update_logfile(logfile, epoch + 1, avg_loss, elapsed)

def do_epoch(q_net, t_net, buffer, gamma, opt, bs=8, epsilon=0.2, num_steps=200):

    batches = tqdm(buffer.batch(bs), total=buffer.num_batches(bs))
    total_loss = 0
    num_batches = 0
    for state, action, reward, done, new_state in batches:

        # Prepare state for making a prediction
        state = tf.convert_to_tensor(state, dtype=tf.float32) # has shape (bs, 6, 5)
        new_state = tf.convert_to_tensor(new_state, dtype=tf.float32)

        # Predict state-action values
        with tf.GradientTape() as tape:
            q_pred = q_net(state)

            # Reduce to only the predicted values for the actions actually taken
            indices = tf.stack([tf.range(q_pred.shape[0]), tf.convert_to_tensor(action, dtype=tf.int32)], axis=1)
            q_pred = tf.gather_nd(q_pred, indices)

            # Use target network to predict value of future states
            future_rewards = t_net(new_state)
            future_rewards = tf.math.reduce_max(future_rewards, axis=1)

            # Set future rewards to 0 wherever the episode ended
            future_rewards = tf.where(done, tf.zeros_like(future_rewards), future_rewards)

            # Compute target vector - current reward plus discounted future reward
            target = (future_rewards * gamma) + tf.convert_to_tensor(reward, dtype=tf.float32)

            # Compute MSE relative to Q_net predicitons (objective function)
            loss = tf.math.reduce_mean(tf.math.square(q_pred - tf.stop_gradient(target)))

            # Update weights of Q network
            grad = tape.gradient(loss, q_net.trainable_weights)
            opt.apply_gradients(zip(grad, q_net.trainable_weights))

        # Save loss and number of batches
        total_loss += loss
        num_batches += 1

    # Reset target weights to align with q network
    t_net.set_weights(q_net.get_weights())

    return float(total_loss / num_batches)

if __name__ == '__main__':
    train('output/TEST')
