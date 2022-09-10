"""
A custom environment for OpenAI Gym based on the popular word game Wordle.

@author: Riley Smith
Created: 4-10-2022
"""
from collections import defaultdict
import string
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

from gym import Env
from gym.spaces import Box, Discrete

import agents
import utils
from count_remaining_words import count_possible_words

# Read words from file
with open('words.txt', 'r') as fp:
    WORDS = [l.strip() for l in fp.readlines()]

def get_word(idx=None):
    if idx is None:
        idx = np.random.randint(low=0, high=len(WORDS))
    return WORDS[idx]

def plot_tile(letter, color=0):
    tile = np.ones((28, 28, 3), dtype=np.uint8)
    tile[:] = COLORS[color]
    cv2.putText(tile, letter, (4, 24), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
    return tile

COLORS = [(128, 128, 128), (252, 219, 3), (0, 186, 68)]

REWARD = 10

class WordleEnv(Env):
    def __init__(self, num_guesses=6, easy=False, **kwargs):
        super().__init__(**kwargs)

        # Choose any one of the valid Wordle words
        # self.action_space = Discrete(2309)
        self.action_space = Discrete(len(WORDS))

        # Observation space is 5 tiles (one for each letter) with three possible
        # states for each one (gray[1], yellow[2], or green[3]) and 26 possible values.
        # So in total there are 26 * 3 = 78 possible states for each tile. We will
        # add one additional state, where all tiles have state '0', representing
        # no guesses made yet (the initial state).
        self.observation_space = Box(low=np.zeros((num_guesses, 5)),
                                     high=np.ones((num_guesses, 5)) * 78,
                                     dtype=np.uint8)

        # Keep track of how many guesses have been made and maximum number of guesses
        self.num_guesses = num_guesses
        self.guesses = 0

        # For this environment, state is a 130-vector for each guess that one-hot
        # encodes the guesses made at that step. It starts as just a 130-zero vector
        self.state = np.zeros((self.num_guesses,130))

        # Choose a secret word
        idx = 0 if easy else None
        self.answer = get_word()

        # Set canvas for rendering
        self.canvas = np.ones((28*num_guesses, 28*5, 3), dtype=np.uint8)

    def draw_on_canvas(self, guess):
        """Update the canvas to show the latest guess"""
        # Get state for each letter and calculate reward (2 points for green, 1 for yellow, 0 for gray)
        tile_states = []
        new_state = []
        letters_guessed = defaultdict(int)

        for guessed_letter, true_letter in zip(guess, self.answer):
            if guessed_letter == true_letter:
                tile_state = 2
                letters_guessed[guessed_letter] += 1
                letter_position = string.ascii_lowercase.index(guessed_letter) + 1
                local_state = 52 + letter_position
            else:
                tile_state = -1
                local_state = -1
            new_state.append(local_state)
            tile_states.append(tile_state)

        for i, (guessed_letter, tile_state) in enumerate(zip(guess, tile_states)):
            if tile_state != -1:
                continue
            letters_guessed[guessed_letter] += 1
            num_occurrences = self.answer.count(guessed_letter)
            if letters_guessed[guessed_letter] <= num_occurrences:
                tile_state = 1
            else:
                tile_state = 0
            # Now account for which letter of the alphabet it is
            letter_position = string.ascii_lowercase.index(guessed_letter) + 1
            # Compute state. First 26 are gray, next 26 yellow, last 26 green
            local_state = tile_state * 26 + letter_position
            new_state[i] = local_state
            # Also store the tile_state
            tile_states[i] = tile_state

        rendered_tiles = [plot_tile(letter, color=state) for letter, state in zip(guess, tile_states)]
        new_row = np.concatenate(rendered_tiles, axis=1)
        self.canvas[(self.guesses - 1) * 28: self.guesses * 28, :, :] = new_row

    def step(self, action, display=False):
        """
        Step function for custom Wordle Env.

        Take the given action (integer from 0 to 2309) and retrieve the word
        corresponding to that guess. Compute the new state based on that word
        and the answer.

        Parameters
        ----------
        action : int
            The integer for the index of the word guessed.
        """
        # Count this as a new guess
        self.guesses += 1

        # Retrieve word from list
        guess = get_word(idx=action)

        # If guessed correctly, give reward, otherwise no reward
        done = False
        reward = 0
        if guess == self.answer:
            # Make sure this isn't a random first guess
            reward = 0 if (self.guesses == 1) else REWARD
            done = True
        elif self.guesses == self.num_guesses:
            reward = -1 * REWARD
            done = True

        # Set state for this guess
        current_state = np.expand_dims(utils.encode_word(guess), axis=0)
        self.state[self.guesses - 1] = current_state
        # self.state = np.concatenate([self.state, current_state], axis=0)

        # Update canvas for rendering purposes
        self.draw_on_canvas(guess)

        # Placeholder for info (not used here, but required by OpenAI Gym)
        info = {'guess': guess}

        return self.state, reward, done, info

    def render(self):
        plt.imshow(self.canvas, vmin=0, vmax=255)
        plt.show()
        time.sleep(2)

    def reset(self):
        # Reset number of guesses made
        self.guesses = 0
        # Reset state
        self.state = np.zeros((self.num_guesses, 130))
        # Choose a new word
        self.answer = get_word()
#
# env = WordleEnv()
# env.state
# env.action_space.n
# dir(env.action_space)
#
# test = np.array([[1, 2], [3, 4]])
# np.array([1, 2]) in test
#
# env.step(env.action_space.sample())
#
#
# env.render()
#
# env.state[np.where(env.state > 0)].shape
#
# sub = env.state[np.where(env.state[:,0] > 0)]
# (sub % 26)[:,0]
#
# from importlib import reload
# reload(agents)
#
# def test_env():
#     env = WordleEnv()
#     state = env.state
#     print('Answer is: ', env.answer)
#
#     done = False
#     while not done:
#         action = agents.fixed_score_agent(state, WORDS)
#         # action = env.action_space.sample()
#         state, _, done, info = env.step(action)
#         env.render()
#         print('Guess: ', info['guess'])
#
# test_env()
