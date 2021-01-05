import io
import os
import sys
import logo
import chess
import chess.pgn
import serializer
import numpy as np
from collections import defaultdict


def parse_dataset(location, max_n=None):
    """
    Parses each move from every game in the dataset as an observation  / example.
    Each observation is a `6⨯8⨯8` `np.array` as dictated in `serializer.py`.

    Args:
        `location`:     Dataset file location.
        `max_n`:        Maximum number of observations.

    Returns:
        `X_train`:      `6⨯8⨯8` training design matrix
        `y_train`:      `n⨯1` training response
        `X_test`:       `6⨯8⨯8` test design matrix
        `y_test`:       `n⨯1` test response
        `g`:            number of games in dataset 
    """

    data = defaultdict(list)

    # We only want to include wins.
    # https://arxiv.org/pdf/1711.09667.pdf shows there's to advantage
    # to including draws in the training set.
    result_map = {'1-0': 1, '0-1': 0}
    g = 0  # Game number
    n = 0  # Observation number

    # PGN's are dispersed over many lines; and seperated by a newline, and so we must
    # concatenate the lines until a newline is observered.
    current_pgn = ''
    with open(location, 'r') as raw_data:
        for pgn_line in raw_data:
            pgn_line = pgn_line.strip()
            if not pgn_line and current_pgn:
                game = chess.pgn.read_game(io.StringIO(current_pgn))

                # Apparently there's a special token "*" which indicates
                # an unknown or otherwise unavailable result. I found this
                # out the hard way :)
                if game.headers['Result'] not in result_map:
                    current_pgn = ''
                    continue
                result = result_map[game.headers['Result']]

                g += 1
                # Erase previous 2 command-line entries
                print('\x1b[1A\x1b[2K'*2)
                print(f'Processing game {g}. Total observations: {n}')

                # Serialize each move in the game
                szr = serializer.Serializer(None)
                for move in game.mainline_moves():
                    if max_n is not None and n >= max_n:
                        X_train, X_test, y_train, y_test = process(data)
                        return X_train, X_test, y_train, y_test, g
                    n += 1
                    szr.board.push(move)

                    data[szr.board.fen().split()[0]].append(result)

                current_pgn = ''
            else:
                current_pgn += f' {pgn_line}'
        X_train, X_test, y_train, y_test = process(data)
        return X_train, X_test, y_train, y_test, g


def process(data, split=0.2):
    """
    Shuffles, splits and ensures the shape of X and y is correct.
    """
    print('Serializing moves...')

    fens = list(data.keys())
    results = list(data.values())
    n = len(fens)

    random = np.arange(n)
    np.random.shuffle(random)
    szr = serializer.Serializer(None)
    X = np.array(list(map(szr.serialize_fen, fens)), dtype=np.int8).reshape((n, 6, 8, 8))[
        random]
    y = np.array(list(map(np.mean, results)), dtype=np.float)[random]
    assert(len(X) == len(y))

    # split into train, test
    cutoff = int(n * split)
    return X[cutoff:], X[:cutoff], y[cutoff:], y[:cutoff]


if __name__ == '__main__':
    logo.show()
    print('\n\n')

    if len(sys.argv) == 1 or not os.path.isfile(sys.argv[1]):
        print('Dataset file location not found. Pass as an argument:')
        print('\t> python parse_dataset.py ./dataset/raw/dataset.pgn\n')
    else:
        X_train, X_test, y_train, y_test, g = parse_dataset(
            sys.argv[1], 1500000)

        save_loc = f'./dataset/processed/dataset_{g}.npz'
        np.savez(save_loc,
                 X_train=X_train, X_test=X_test,
                 y_train=y_train, y_test=y_test)
        print(f'Saved dataset in {save_loc}')
