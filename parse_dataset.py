import io
import os
import sys
import logo
import chess
import chess.pgn
import serializer
import numpy as np


def parse_dataset(location, max_n=None):
    """
    Parses each move from every game in the dataset as an observation  / example.
    Each observation is a 6x8x8 `np.array` as dictated in `serializer.py`.

    Args:
        `location`: Dataset file location.
        `max_n`: Maximum number of observations.

    Returns:
        `X, y`: `n`x6x8x8 design matrix, `n`x1 response
    """

    X, y = [], []

    result_map = {'1/2-1/2': 0,  '1-0': 1, '0-1': -1}
    i = 0  # Game number
    n = 0  # Observation number

    # PGN's are dispersed over many lines; and seperated by a newline, and so we must
    # concatenate the lines until a newline is observered.
    current_pgn = ''
    with open(location, 'r') as raw_data:
        for pgn_line in raw_data:
            pgn_line = pgn_line.strip()
            if not pgn_line and current_pgn:
                game = chess.pgn.read_game(io.StringIO(current_pgn))
                result = result_map[game.headers['Result']]
                i += 1

                # Erase previous 2 command-line entries
                print('\x1b[1A\x1b[2K'*2)
                print(f'Processing game {i}. Total observations: {n}')

                # Serialize each move in the game
                szr = serializer.Serializer(None)
                for move in game.mainline_moves():
                    if max_n is not None and n >= max_n:
                        X, y = reshape(X, y, n)
                        return X, y
                    n += 1
                    szr.board.push(move)
                    X.append(szr.serialize())
                    y.append(result)

                current_pgn = ''
            else:
                current_pgn += f' {pgn_line}'
        X, y = reshape(X, y, n)
        return X, y


def reshape(X, y, n):
    """
    Ensures the shape of X and y is correct; and is typed as an np.array.
    """
    return np.array(X).reshape((n, 6, 8, 8)), np.array(y)


if __name__ == '__main__':
    logo.show()
    print('\n\n')

    if len(sys.argv) == 1 or not os.path.isfile(sys.argv[1]):
        print('Dataset file location not found. Pass as an argument:')
        print('\t> python parse_dataset.py ./dataset/raw/dataset.pgn\n')
    else:
        X, y = parse_dataset(sys.argv[1], 10000)

        save_loc = './dataset/processed/dataset.npz'
        np.savez(save_loc, X, y)
        print(f'Saved dataset in {save_loc}')
