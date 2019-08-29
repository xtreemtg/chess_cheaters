import numpy as np


class FEN_convertor():

    def __init__(self, matrix):
        self.matrix = np.array(matrix)

    def _decide_castling(self):
        # TODO: make it more smart
        castle_str = ''
        board = self.matrix
        K, k, KR, QR, kr, qr = (7, 4), (0, 4), (7, 7), (7, 0), (0, 7), (0, 0)
        if board[K] != 'K' and board[k] != 'k':
            return '-'
        if board[K] == 'K':
            if board[KR] == 'R':
                castle_str += 'K'
            if board[QR] == 'R':
                castle_str += 'Q'
        if board[k] == 'k':
            if board[kr] == 'r':
                castle_str += 'k'
            if board[qr] == 'r':
                castle_str += 'q'
        castle_str = castle_str if len(castle_str) > 0 else '-'
        return castle_str

    def _enpassant(self):
        # TODO: smart decision with enpassing
        return '-'

    def _50move_rule(self):
        # TODO: smart decision with 50 move rule
        return '0'

    def _full_moves(self):
        # TODO: smart decision with number of moves passed
        return '1'

    def _get_rank_data(self, rank):
        if (rank == '.').all():
            return '8'
        counter = 0
        rank_str = ''
        for square in rank:
            if square == '.':
                counter += 1
            else:
                if counter == 0:
                    rank_str += square
                else:
                    rank_str += str(counter) + square
                    counter = 0
        rank_str = rank_str + str(counter) if counter != 0 else rank_str
        return rank_str

    def convert(self, whos_move):
        whos_move = 'b' if whos_move == 'black' else 'w'
        piece_placement = ''
        for rank in self.matrix:
            piece_placement += self._get_rank_data(rank) + '/'
        piece_placement = piece_placement[:-1]
        castling = self._decide_castling()
        enpassant = self._enpassant()
        fifty_move = self._50move_rule()
        moves = self._full_moves()
        final = f'{piece_placement} {whos_move} {castling} {enpassant} {fifty_move} {moves}'
        return final








