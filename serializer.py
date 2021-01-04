import chess
import numpy as np


class Serializer(object):
    """
    Initialize with a chess board.
    No bit-wise operations to be seen here, I ain't a sadist
    """

    # Ascii piece -> number representation
    # Constructed in a way such that abs(number representation)-1 gives the index of the piece
    # in the 6x8x8 np.array; and sign(number representation) gives the associated value in the array.
    piece_map = {
        'P': 1, 'p': -1,
        'N': 2, 'n': -2,
        'B': 3, 'b': -3,
        'R': 4, 'r': -4,
        'Q': 5, 'q': -5,
        'K': 6, 'k': -6
    }

    def __init__(self, board: chess.Board):
        if board is None:
            self.board = chess.Board()
        else:
            self.board = board

    def serialize(self):
        """
        Serializes the chess board into a 6x8x8 `np.array`. A new game will look as follows:

                ♟︎                ♞                ♝                ♜                ♛                 ♚
         0 0 0 0 0 0 0 0 | 0-1 0 0 0 0-1 0 | 0 0-1 0 0-1 0 0 |-1 0 0 0 0 0 0-1 | 0 0 0-1 0 0 0 0 | 0 0 0 0-1 0 0 0
        -1-1-1-1-1-1-1-1 | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0
         0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0
         0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0
         0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0
         0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0
         1 1 1 1 1 1 1 1 | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0
         0 0 0 0 0 0 0 0 | 0 1 0 0 0 0 1 0 | 0 0 1 0 0 1 0 0 | 1 0 0 0 0 0 0 1 | 0 0 0 1 0 0 0 0 | 0 0 0 0 1 0 0 0
        """

        # Serialized array
        ser = np.zeros((6, 64), np.int8)

        for i in range(64):
            # Obtain number representation of piece on ith square.
            # Iterates through chess board left to right, bottom to top.
            p = self.board.piece_at(i)
            if p is not None:
                pn = self.piece_map[p.symbol()]
                ser[abs(pn)-1, i] = np.sign(pn)

        # Numpy reshapes from left to right, top to bottom, so must flip on x-axis.
        return ser.reshape((6, 8, 8))[:, :, ::-1]

    def legal_moves(self):
        """
        Uses python-chess to return a list of legal moves.
        """
        return list(self.board.legal_moves)
