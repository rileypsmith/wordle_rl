from collections import defaultdict

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

    # Make a bunch of container lists/dicts that will be populated as we loop over letters
    correct_positions = {}
    incorrect_positions = {}
    max_occurences = {}
    min_occurences = {}
    for row_idx in range(state.shape[0]):
        row_colors = tile_colors[row_idx]
        row_letters = guessed_letters[row_idx]

        for i, (letter_idx, color) in enumerate(zip(row_letters, row_colors)):
            # Get the actual character letter instead of its index
            letter = string.ascii_lowercase[letter_idx]

            if color == 2:
                # Tile is green so just store it in the right position
                correct_positions[i] = letter
            else:
                # Mark this as being in the wrong position
                if i in incorrect_positions:
                    incorrect_positions[i].append(letter)
                else:
                    incorrect_positions[i] = [letter]
                # Get all occurences of this letter in the current word
                letter_occurences = row_colors[np.where(row_letters == letter_idx)]
                # Count how many times it was guessed and how many times it is not gray
                num_guessed = len(letter_occurences)
                num_gray = (letter_occurences == 0).sum()
                if num_gray > 0:
                    # We have a ceiling on how many times the letter appears in the word
                    max_occurences[letter] = min(num_guessed - num_gray, max_occurences.get(letter, np.inf))
                else:
                    # Letter appears at least once, giving us a minimum
                    min_occurences[letter] = max(num_guessed, min_occurences.get(letter, 0))

    def is_valid(word):
        for i, letter in correct_positions.items():
            if not word[i] == letter:
                return False
        for letter, occurences in min_occurences.items():
            if not word.count(letter) >= occurences:
                return False
        for letter, occurences in max_occurences.items():
            if word.count(letter) > occurences:
                return False
        return True
