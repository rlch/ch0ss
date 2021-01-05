import tensorflow as tf
import chess
import chess.pgn
import serializer
import logo
import numpy as np
import io
import os
import sys


def test(location):
    """
    Evaluates a given FEN using the model at `location`.
    """
    model = tf.keras.models.load_model(location)
    while True:
        logo.show()
        print('\n\n')
        fen = input('Paste FEN\n').split()[0]

        szr = serializer.Serializer(None)
        bitboard = szr.serialize_fen(fen)

        value = model.predict(np.expand_dims(bitboard, 0))

        print(szr.board.unicode())
        print(f'ch0ss evaluation: {value}\n\n\n')
        input('Press anything to continue...')


if __name__ == '__main__':
    logo.show()
    print('\n\n')

    if len(sys.argv) == 1 or not os.path.isdir(sys.argv[1]):
        print('Model file location not found. Pass as an argument:')
        print('\t> python test_value.py ./models/model\n')
    else:
        test(sys.argv[1])
