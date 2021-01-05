import chess
import tensorflow as tf
import serializer as szr
import numpy as np

"""
TODO:

    Maybe order the moves in terms of likeliness of being a good move, to increase prob of pruning.
        - Pruning happens left to right, so good moves first will result in bad moves second not being considered.
        - For instance, checks, captures, castling would have a higher likelihood of being a good move - than pushing a pawn.

"""


class Minimax(object):
    """
    Provides move selection via minimax with alpha-beta pruning.

    Args:
        `location`: location of value function model.
    """

    def __init__(self, location):
        self.model: tf.keras.Model = tf.keras.models.load_model(location)

    def search(self, board: chess.Board, depth, side, alpha=-2, beta=2):
        """
        Performs a minimax search with alpha-beta pruning.
        Alpha, beta are set to -2, 2 respectively; because the suppport of the CNN is [0,1].

        Args:
            `board`:    Chess board state at current node. 
            `depth`:    Search depth.
            `side`:     White (True), Black (False). White is the maximising player.
            `alpha`:    Current minimum score White is assured of.
            `beta`:     Current maximum score Black is assured of.

        Returns:
            (evaluation, uci-encoded move that `side` should play)
        """

        if depth == 0:
            input = szr.Serializer(board).serialize()
            return (self.model.predict(np.expand_dims(input, 0)), board.pop())

        if board.is_game_over():
            res = board.result()
            if res == '0-1':
                return -1
            elif res == '1-0':
                return 1
            else:
                return 0

        if side:
            max_score = -2
            best_move = ''
            for move in board.legal_moves:
                new_board = board.copy()
                new_board.push(move)
                score = self.search(new_board, depth-1, alpha, beta, False)[0]
                if score > max_score:
                    max_score = score
                    best_move = move
                alpha = max(alpha, score)
                if alpha >= beta:
                    # black has a better option available earlier on in the tree.
                    break
            return (max_score, best_move)
        else:
            min_score = 2
            best_move = ''
            for move in board.legal_moves:
                new_board = board.copy()
                new_board.push(move)
                score = self.search(new_board, depth-1, alpha, beta, True)[0]
                min_score = min(score, min_score)
                if score < min_score:
                    min_score = score
                    best_move = move
                beta = min(beta, score)
                if beta <= alpha:
                    # white has a better option avilable earlier on in the tree.
                    break
            return (min_score, best_move)
