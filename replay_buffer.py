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

def train(out_dir, epochs=50):
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

    best_loss = None
    start = time.time()
    for epoch in range(epochs):
        print(f'EPOCH {epoch+1}/{epochs}')
        avg_loss = do_epoch(q_net, t_net, gamma, opt)

        # Save weights of best network
        if (best_loss is None) or (avg_loss < best_loss):
            # Save weights of network
            q_net.save_weights(str(Path(weight_dir, 'checkpoint')))
            best_loss = avg_loss

        # Print progres to CSV file
        elapsed = round((time.time() - start) / 60, 2)
        update_logfile(logfile, epoch + 1, avg_loss, elapsed)

def do_epoch(q_net, t_net, gamma, opt, num_steps=200):

    # # Initialize environment
    # env = WordleEnv()

    pbar = tqdm(range(num_steps))
    total_rewards = 0
    num_batches = 0
    batch_size = 10
    epsilon = 0.2
    for i in pbar:
        guesses_made = 0
        states = []
        envs = [WordleEnv() for _ in range(batch_size)]
        for j in range(batch_size):
            # Get to random starting state
            local_env = envs[j]
            for _ in range(guesses_made % 5):
                local_env.step(local_env.action_space.sample())
            guesses_made += 1

            # Populate replay buffer
            states.append(local_env.state)

        # Prepare states for Q network to choose an action
        state_batch = tf.convert_to_tensor(np.stack(states, axis=0))

        # Start tracking gradient here
        with tf.GradientTape() as tape:
            # Get network's chosen action
            q_pred = q_net(state_batch)
            chosen_actions = tf.argmax(q_pred, axis=-1)

            # Implement epsilon-greedy randomness
            # random_actions = tf.cast(tf.random.uniform((batch_size,), minval=0, maxval=NUM_WORDS, dtype=tf.int64), tf.float32)
            random_actions = tf.random.uniform((batch_size,), minval=0, maxval=NUM_WORDS, dtype=tf.int64)
            actions = tf.where(tf.random.uniform((batch_size,)) < epsilon, random_actions, chosen_actions)

            q_acted = tf.math.reduce_sum(q_pred * tf.one_hot(actions, NUM_WORDS))
            # q_acted = tf.math.reduce_max(q_pred, axis=-1)

            # Get the real rewards, next states, and whether or not the game is over
            rewards = []
            done = []
            next_states = []
            for j, action in enumerate(actions):
                local_env = envs[j]
                next_state, reward, local_done, _ = local_env.step(action)
                rewards.append(reward)
                done.append(local_done)
                next_states.append(next_state)
            rewards = tf.constant(rewards, dtype=tf.float32)
            done = tf.constant(done, dtype=tf.float32)

            # Estimate the future value of each action using the target network
            next_state_batch = tf.convert_to_tensor(np.stack(next_states, axis=0))
            t_pred = t_net(next_state_batch)
            future_rewards = tf.math.reduce_max(t_pred, axis=-1)

            # Finally compute "true" rewards
            true_rewards = rewards + ((1 - done) * gamma * future_rewards)
            if i == 150:
                breakpoint()

            # Loss is just MSE between q_net's predicted value and "true" value
            loss = tf.math.reduce_mean(tf.math.square(q_acted - tf.stop_gradient(true_rewards)))

            # Update weights of Q network
            grad = tape.gradient(loss, q_net.trainable_weights)
            opt.apply_gradients(zip(grad, q_net.trainable_weights))

        # # Figure out the "true" values of each action at this state
        # values = []
        # for i in range(env.action_space.n):
        #     local_env = copy.deepcopy(env)
        #     state, reward, done, _ = local_env.step(i)
        #     # if not done:
        #     # Apply target network to get estimate of future rewards
        #     future_reward =
        #     next_action = int(tf.math.argmax(t_net(tf.expand_dims(tf.convert_to_tensor(state), axis=0)), axis=1))
        #     _, future_reward, _, _ = local_env.step(next_action)
        #     # breakpoint()
        #     values.append(reward + (gamma * future_reward))
        #     # else:
        #     #     if local_env.correct:
        #     #         values.append(10)
        #
        # # Turn values into a tensor so it can be used to train primary network
        # values = tf.expand_dims(tf.constant(values, dtype=tf.float32), axis=0)
        #
        # # Now train primary network using that as "truth"
        # with tf.GradientTape() as tape:
        #     # Predict value of each action at current state
        #     q_pred = q_net(tf.expand_dims(tf.convert_to_tensor(env.state), axis=0))
        #     loss = tf.math.reduce_mean(tf.math.square(q_pred - values))
        #     # Update weights of Q network
        #     grad = tape.gradient(loss, q_net.trainable_weights)
        #     opt.apply_gradients(zip(grad, q_net.trainable_weights))

        # Display total rewards for the batch along with progres bar
        total_rewards += tf.math.reduce_mean(true_rewards)
        num_batches += 1
        pbar.set_description(f'Rewards: {total_rewards / num_batches:.2f}')

        # # Reset environment
        # env.reset()

    # Reset target weights to align with q network
    t_net.set_weights(q_net.get_weights())

    return float(total_rewards / num_batches)

if __name__ == '__main__':
    train('output/TEST')
