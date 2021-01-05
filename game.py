from move_selection import Minimax
import chess
import chess.pgn
import serializer
import logo
import numpy as np
import tensorflow as tf
import os
import io
import sys

"""
TODO:

    Implement backend in Torch/Django + React frontend using a chessboard library
        - Run game as a webserver

"""


def display_board(board, side):
    unicode = None
    if side:
        unicode = board.unicode(borders=True)
    else:
        unicode = board.transform(chess.flip_vertical).unicode(borders=True)
    print(unicode+'\n')


def play(location, option):

    model: tf.keras.Model = tf.keras.models.load_model(location)
    minimax = Minimax(model)

    logo.show()
    print('\n\n')

    side = None
    while side not in ['W', 'B']:
        side = input('Play as:\nWhite (W) | Black (B) | Random (R) > ')
        if side == 'R':
            side = 'W' if np.random.uniform() < 0.5 else 'B'
    side = side == 'W'

    szr = serializer.Serializer(None)
    if option == 'P':
        pgn = chess.pgn.read_game(io.StringIO(input('Paste PGN:\n> ')))
        for move in pgn.mainline_moves():
            szr.board.push(move)

    while not szr.board.is_game_over():
        logo.show()
        print('\n\n')

        display_board(szr.board, side)
        score = model.predict(np.expand_dims(szr.serialize(), 0))
        print(f'ch0ss evaluation: {score}')

        if szr.board.turn == side:
            uci = None
            while not szr.board.is_legal(uci):
                if uci is not None:
                    print('Illegal move')
                try:
                    uci = chess.Move.from_uci(input('Type your move:\n > '))
                except ValueError:
                    print('Invalid move')
                    uci = None
            if szr.board.is_legal(uci):
                szr.board.push(uci)
            else:
                print('Illegal move.')
        else:
            print('ch0ss is thinking...')
            move = minimax.search(szr.board, 3, side)
            szr.board.push(move[1])



    # value = model.predict(np.expand_dims(bitboard, 0))


if __name__ == '__main__':
    logo.show()
    print('\n\n')

    if len(sys.argv) == 1 or not os.path.isdir(sys.argv[1]):
        print('Model file location not found. Pass as an argument:')
        print('\t> python game.py ./models/model\n')
    else:
        while True:
            opt = input(
                'Play ch0ss:\nNew game (N) | From PGN (P) | Exit (X) > ')
            if opt == 'X':
                sys.exit()
            play(sys.argv[1], opt)
