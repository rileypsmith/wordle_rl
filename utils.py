"""
Utility functions for training Wordle agent.

@author: Riley Smith
Created: 8-7-2022
"""
from pathlib import Path
import string

import numpy as np

def setup_output_directory(output_dir):
    """
    Setup the output directory. If it already exists, append an integer suffix
    and create a new one.
    """
    checked = False
    i = 0
    while Path(output_dir).exists():
        if checked:
            output_dir = str(output_dir)[:-3] + f'_{i:02}'
        else:
            output_dir = str(output_dir) + f'_{i:02}'
        checked = True
        i += 1
    Path(output_dir).mkdir()
    return output_dir

def one_hot(indices):
    """
    Convert a vector of indices to a concatenated one-hot vector encoding each
    position out of 26 possible positions.
    """
    out = np.zeros((indices.size, 26), dtype=np.int32)
    out[np.arange(indices.size), indices] = 1
    return out.reshape((indices.size * 26,))

def encode_word(word):
    """
    Map a word from string form to a vector representation. The representation
    is a 130-vector. The first 26 positions are a 1-hot vector for the first
    letter in the word, the next 26 are a 1-hot vector for the next letter in the
    word, etc.

    Parameters
    ----------
    word : str
        The word (5 letters) to convert to vector representation.
    """
    # Get index of each letter
    indices = [string.ascii_lowercase.index(l) for l in word.lower()]
    return one_hot(np.array(indices))

encode_word('hello')
