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

def leave_one_out(word_in, idx):
    word = word_in.copy()
    return word[idx], np.delete(word, idx)

def count_possible_words(state):
    if state.sum() == 0:
        return len(WORDS)
    # Mask out zeros in state
    state = state[np.where(state > 0)]
    # Separate state into letter guesses and tile colors
    tile_colors, guessed_letters = np.divmod(state - 1, 26)
    conditions = defaultdict()
    for row_idx in state.shape[0]:
        row_colors = tile_colors[row_idx]
        row_letters = guessed_letters[row_idx]
        for i in range(len(row_letters)):
            letter, remaining_letters = leave_one_out(row_letters, i)
            color, remaining_colors = leave_one_out(row_colors, i)
            if color == 2:
                # Tile is green, so store the index at which this letter is found
                conditions[string.ascii_lowercase[letter]] = i
            elif color == 1:




def get_reward(previous_state, new_state):


    if state.sum() == 0:
        reward = 0
        for i, (guessed, correct) in enumerate(zip(guess, answer)):
            if guessed == correct:
                reward += 3
            elif guessed in answer:
                reward += 2
            else:
                reward += 1

    # Start counting rewards
    reward = 0
    for i, letter in enumerate(guess):
        idx = string.ascii_lowercase.index(letter)

        if idx in guessed_letters[:,i]:
            guessed_pos = (guessed_letters[:,i] == idx).argmax()
            if tile_colors[guessed_pos,i] == 2:
                reward += 2
                continue
            else:
                reward -= 1
                continue

        elif idx in guessed_letters:
            guessed_positions = np.where(guessed_letters == idx)
            if np.any(tile_colors[guessed_positions] == 0):
                reward -= 1
            else:
                # Is the tile green or yellow?
                if answer[i] == letter:
                    reward += 3
                else:
                    reward += 1

        else:
            if answer[i] == letter:
                reward += 3
            elif letter in answer:
                reward += 2
            else:
                reward += 1

    return reward

class WordleEnv(Env):
    def __init__(self, num_guesses=6, **kwargs):
        super().__init__(**kwargs)

        # Choose any one of the 2309 valid Wordle words
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

        # Set a starting state, all zeros
        self.state = np.zeros((num_guesses, 5))

        # Choose a secret word
        self.answer = get_word()

        # Set canvas for rendering
        self.canvas = np.ones((28*num_guesses, 28*5, 3), dtype=np.uint8)

    def draw_on_canvas(self, tile_states, guess):
        """Update the canvas to show the latest guess"""
        rendered_tiles = [plot_tile(letter, color=state) for letter, state in zip(guess, tile_states)]
        new_row = np.concatenate(rendered_tiles, axis=1)
        self.canvas[self.guesses * 28: (self.guesses + 1) * 28, :, :] = new_row

    def step(self, action):
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
        # Retrieve word from list
        guess = get_word(idx=action)
        print('guess: ', guess)
        print('idx: ', action)

        # Get state for each letter and calculate reward (2 points for green, 1 for yellow, 0 for gray)
        tile_states = []
        new_state = []
        reward = 0
        letters_guessed = defaultdict(int)

        for guessed_letter, true_letter in zip(guess, self.answer):
            if guessed_letter == true_letter:
                tile_state = 2
                reward += 2
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
                reward += 1
            else:
                tile_state = 0
            # Now account for which letter of the alphabet it is
            letter_position = string.ascii_lowercase.index(guessed_letter) + 1
            # Compute state. First 26 are gray, next 26 yellow, last 26 green
            local_state = tile_state * 26 + letter_position
            new_state[i] = local_state
            # Also store the tile_state
            tile_states[i] = tile_state


        # for guessed_letter, true_letter in zip(guess, self.answer):
        #     letters_guessed[guessed_letter] += 1
        #     # First figure out if it is gray, yellow, or green
        #     if guessed_letter == true_letter:
        #         tile_state = 2
        #         reward += 2
        #     elif guessed_letter in self.answer:
        #         num_occurrences = self.answer.count(guessed_letter)
        #         if letters_guessed[guessed_letter] <= num_occurrences:
        #             tile_state = 1
        #             reward += 1
        #         else:
        #             tile_state = 0
        #     else:
        #         tile_state = 0
        #     # Now account for which letter of the alphabet it is
        #     letter_position = string.ascii_lowercase.index(guessed_letter) + 1
        #     # Compute state. First 26 are gray, next 26 yellow, last 26 green
        #     new_state.append(tile_state * 26 + letter_position)
        #     # Also store the tile_state
        #     tile_states.append(tile_state)

        # Set state for this guess
        self.state[self.guesses] = np.array(new_state, dtype=np.uint8)

        # Update canvas for rendering purposes
        self.draw_on_canvas(tile_states, guess)

        # Count this as a new guess
        self.guesses += 1

        # If you have hit the max guesses or have guessed the word exactly, you are done
        done = (self.guesses >= self.num_guesses) or (reward == 10)

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
        self.state = np.zeros((self.num_guesses, 5))
        # Choose a new word
        self.answer = get_word()

env = WordleEnv()
env.state

test = np.array([[1, 2], [3, 4]])
np.array([1, 2]) in test

env.step(env.action_space.sample())


env.render()

env.state[np.where(env.state > 0)].shape

sub = env.state[np.where(env.state[:,0] > 0)]
(sub % 26)[:,0]

from importlib import reload
reload(agents)

def test_env():
    env = WordleEnv()
    state = env.state
    print('Answer is: ', env.answer)

    done = False
    while not done:
        action = agents.fixed_score_agent(state, WORDS)
        # action = env.action_space.sample()
        state, _, done, info = env.step(action)
        env.render()
        print('Guess: ', info['guess'])

test_env()
