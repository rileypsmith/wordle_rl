"""
Sript for agents to play Wordle game.

@author: Riley Smith
Created: 07-02-2022
"""
import string

import numpy as np

def word_to_num(word):
    """Turn a word into its index values (integers)"""
    return np.array([string.ascii_lowercase.index(letter) + 1 for letter in word])

def num_to_word(num):
    return ''.join([string.ascii_lowercase[i - 1] for i in num])


def fixed_score_agent(state, words):
    """
    Agent that will make a guess which is the highest scoring word that has not
    already been guessed.
    """
    if state.sum() == 0:
        return np.random.randint(0, len(words))

    # Turn words into integers
    words = [word_to_num(word) for word in words]

    # Subselect state to guesses that have been made
    valid_state = state[np.where(state[:,0] > 0)]

    # Make mapping for each letter position to letter's score for that position
    letters_guessed = valid_state % 26
    letter_scores = valid_state // 26

    # # Find any matching tiles (green)
    # matches = {}
    # for col_idx in range(letters_guessed.shape[1]):
    #     local_scores = letter_scores[:,col_idx]
    #     if np.any(local_scores == 2):
    #         correct_letter = letters_guessed[:,col_idx][np.argmax(local_scores == 2)]
    #         matches[col_idx] = correct_letter
    #     else:
    #         matches[col_idx] = -1

    # Now find score for non-matching tiles
    letter_mapping = {}

    # Find all letters that are yellow and all letters that are not in the word
    yellow = np.unique(letters_guessed[np.where(letter_scores == 1)])
    gray = np.unique(letters_guessed[np.where(letter_scores == 0)])
    for col_idx in range(letters_guessed.shape[1]):
        col_guessed = letters_guessed[:,col_idx]
        col_scores = letter_scores[:,col_idx]
        local_mapping = {}
        for guess, score in zip(col_guessed, col_scores):
            if score == 2:
                local_mapping[guess] = 2
            else:
                local_mapping[guess] = -1
        for guess in gray:
            if not guess in local_mapping:
                local_mapping[guess] = -1
        for guess in yellow:
            if not guess in local_mapping:
                local_mapping[guess] = 1
        for i in range(1, 27):
            if i not in local_mapping:
                local_mapping[i] = 0
        letter_mapping[col_idx] = local_mapping

    # Score each word
    scores = []
    for word in words:
        score = 0
        for i, letter in enumerate(word):
            score += letter_mapping[i][letter]
        scores.append(score)
    scores = np.array(scores)

    # See which words have already been guessed
    words_guessed = np.array([np.any(np.all(word == letters_guessed, axis=1)) for word in words]).astype(bool)
    scores[np.where(words_guessed)] = 0

    return np.argmax(scores)
