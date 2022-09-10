"""
Script to evaluate agent performance by having them play (and display) a game
of Wordle.

@author: Riley Smith
Created: 8-7-2022
"""
import numpy as np
import tensorflow as tf

from wordle_env import WordleEnv

from nn_agents import LSTMAgent

def evaluate(weights_file, n_trials=100):
    """
    Instantiate an agent, load the weights, build an environment, and then play
    Wordle a bunch of times to see how the network does.
    """
    # Instantiate network
    q_net = LSTMAgent()
    q_net.load_weights(weights_file)

    # Run a bunch of trials of Wordle
    num_guesses = []
    for _ in range(n_trials):
        # Build environment
        env = WordleEnv()
        # Use agent to play
        done = False
        state = env.state
        while not done:
            # Choose an action
            net_input = tf.expand_dims(tf.convert_to_tensor(state), axis=0)
            action = int(tf.math.argmax(q_net(net_input), axis=1))
            state, _, done, _ = env.step(action)
            env.render()

        # Done playing the game, see how many guesses it took or if the network failed
        if env.correct:
            num_guesses.append(env.num_guesses)
        else:
            num_guesses.append(-1)

    num_guesses = np.array(num_guesses)

    # Return success rate and average number of guesses
    breakpoint()
    successes = num_guesses[np.where(num_guesses > 0)]
    success_rate = successes.size / num_guesses.size
    avg_guesses = successes.mean()

    return success_rate, avg_guesses

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='TEST',
                        help='The directory to evaluate on.')
    parser.add_argument('--n_trials', type=int, default=100,
                        help='The number of trials to run for evaluation.')
    args = parser.parse_args()
    weights = r"C:\Users\thehu\OneDrive\Documents\Code_Projects\wordle_rl\output\{}\weights\checkpoint".format(args.output_dir)
    success_rate, avg_guesses = evaluate(weights, n_trials=args.n_trials)
    breakpoint()
