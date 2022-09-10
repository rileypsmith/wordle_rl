"""
Utility functions for training Wordle agent.

@author: Riley Smith
Created: 8-7-2022
"""
from pathlib import Path

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
