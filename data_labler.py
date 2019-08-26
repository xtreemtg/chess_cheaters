import sys

import chess
import chess.engine
from stockfish import Stockfish

from FEN_convertor import FEN_convertor

from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtWidgets import QApplication, QWidget, QLabel

import threading
import queue

sf_file = '/Users/xtreemtg888/Downloads/stockfish-10-mac/Mac/stockfish-10-64'

class MainWindow(QWidget):
    """
    Create a surface for the chessboard.
    """

    def __init__(self, board, stockfish):
        """
        Initialize the chessboard.
        """
        super().__init__()

        self.setWindowTitle("Chess GUI")
        self.setGeometry(300, 300, 800, 800)

        self.widgetSvg = QSvgWidget(parent=self)
        self.widgetSvg.setGeometry(10, 10, 600, 600)

        self.boardSize = min(self.widgetSvg.width(),
                             self.widgetSvg.height())
        self.coordinates = True
        self.margin = 0.05 * self.boardSize if self.coordinates else 0
        self.squareSize = (self.boardSize - 2 * self.margin) / 8.0
        self.pieceToMove = [None, None]
        self.second_click = None

        self.board = board
        self.engine = stockfish
        self.predict_label = QLabel('test', self)
        self.predict_label.move(50, 600)
        self.engine_pred_move()
        self.drawBoard()


    def engine_pred_move(self):
        self.predict_label.setText('predicting moves..')
        self.predict_label.adjustSize()
        moves = []
        count = 0
        predict_board = self.board.copy()
        while not predict_board.is_game_over() and count < 10:
            result = engine.play(predict_board, chess.engine.Limit(time = 0.1))
            move = result.move
            moves.append(move)
            predict_board.push(move)
            count += 1
        moves = self.board.variation_san(moves)
        NEXT_MOVES = 'Next few predicted moves:\n'
        self.predict_label.setText(NEXT_MOVES + moves)
        self.predict_label.adjustSize()


    @pyqtSlot(QWidget)
    def mousePressEvent(self, event):
        """
        Handle left mouse clicks and enable moving chess pieces by
        clicking on a chess piece and then the target square.

        Moves must be made according to the rules of chess because
        illegal moves are suppressed.
        """
        if event.x() <= self.boardSize and event.y() <= self.boardSize:
            if event.buttons() == Qt.LeftButton:
                if self.margin < event.x() < self.boardSize - self.margin and self.margin < event.y() < self.boardSize - self.margin:
                    file = int((event.x() - self.margin) / self.squareSize)
                    rank = 7 - int((event.y() - self.margin) / self.squareSize)
                    square = chess.square(file, rank)
                    piece = self.board.piece_at(square)
                    coordinates = "{}{}".format(chr(file + 97), str(rank + 1))
                    self.second_click = self.pieceToMove[0]
                    if self.pieceToMove[0] is not None:
                        move = chess.Move.from_uci("{}{}".format(self.pieceToMove[1], coordinates))
                        if move in self.board.legal_moves:
                            self.board.push(move)
                        piece = None
                        coordinates = None
                    self.pieceToMove = [piece, coordinates]
                    self.drawBoard()
                    if self.second_click:
                        t = threading.Thread(target=self.engine_pred_move)
                        t.start()

    def drawBoard(self):
        """
        Draw a chessboard with the starting position and then redraw
        it for every new move.
        """
        self.boardSvg = self.board._repr_svg_().encode("UTF-8")
        self.drawBoardSvg = self.widgetSvg.load(self.boardSvg)

        return self.drawBoardSvg


if __name__ == "__main__":
    matrix = [
        ['.', 'K', '.', '.', 'k', '.', '.', '.'],
        ['.', '.', '.', '.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.', '.', '.', '.'],
        ['.', '.', '.', 'p', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.', '.', '.', 'R']

    ]

    cnvrtr = FEN_convertor(matrix)
    result = cnvrtr.convert('white')
    engine = chess.engine.SimpleEngine.popen_uci(sf_file)
    print(result)

    OG = chess.Board(result)
    chessGui = QApplication(sys.argv)
    window = MainWindow(OG, engine)

    window.show()
    sys.exit(chessGui.exec_())
