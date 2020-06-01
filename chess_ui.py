import sys

import chess
import chess.engine
import chess.svg

from FEN_convertor import FEN_convertor

from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QInputDialog, QMessageBox, QLineEdit
from image_processing import do_magic

import faulthandler
import time

faulthandler.enable()

import threading

sf_file = '/Users/xtreemtg888/Downloads/stockfish-10-mac/Mac/stockfish-10-64'
komodo_file = '/Users/xtreemtg888/Downloads/komodo-10_ae4bdf/OSX/komodo-10-64-osx'
engine = chess.engine.SimpleEngine.popen_uci(sf_file)
engine2 = chess.engine.SimpleEngine.popen_uci(komodo_file)
STOCKFISH = 'stockfish'
KOMODO = 'komodo'
ENGINES = {STOCKFISH: engine, KOMODO: engine2}
LOCK = threading.RLock()
PLAYER = 'white'
CHESSGUI = None


class MainWindow(QWidget):
    """
    Create a surface for the chessboard.
    """

    def __init__(self, board):
        """
        Initialize the chessboard..
        """
        super().__init__()

        self.setWindowTitle("Chess Cheaters")
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
        self.predict_label1 = QLabel(STOCKFISH, self)
        self.predict_label1.move(50, 600)
        self.predict_label2 = QLabel(KOMODO, self)
        self.predict_label2.move(50, 650)
        self.result_label = QLabel('', self)
        self.result_label.move(650, 195)
        self.arrows = []
        button = QPushButton('Takeback', self)
        button.setToolTip('Take back a move')
        button.move(650, 75)
        button.clicked.connect(self.take_back)
        self.load_image = QPushButton('Load image', self)
        self.load_image.setToolTip('Load an image')
        self.load_image.move(650, 115)
        self.load_image.clicked.connect(self.get_image_path)
        self.drawBoard()
        if not self.check_position():
            self.predict_threads()
        self.image_path = None


    def get_image_path(self):
        text, okPressed = QInputDialog.getText(self, "", "Path_to_image:", QLineEdit.Normal, "")
        if okPressed and text != '':
            try:
                matrix = do_magic(text)
                # matrix = [
                #     ['.', '.', '.', '.', '.', '.', '.', '.'],
                #     ['.', '.', '.', '.', '.', '.', '.', '.'],
                #     ['.', '.', '.', '.', '.', '.', '.', '.'],
                #     ['.', '.', '.', '.', '.', 'K', '.', '.'],
                #     ['.', '.', '.', 'k', '.', '.', '.', '.'],
                #     ['.', '.', '.', '.', 'r', '.', '.', '.'],
                #     ['.', '.', '.', '.', '.', '.', '.', '.'],
                #     ['.', '.', '.', '.', '.', '.', '.', '.']
                #
                # ]
                print(matrix)

                cnvrtr = FEN_convertor(matrix)
                result = cnvrtr.convert(PLAYER)
                self.board = chess.Board(result)
                self.arrows = []
                self.drawBoard()
                #threading.Thread(target=self.check_position).start():
                if not self.check_position():
                    self.predict_threads()
            except AttributeError as e:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setText(str(e))
                msg.exec_()
            except TypeError as e:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setText(str(e))
                msg.exec_()


    def predict_threads(self):
        t1 = threading.Thread(target=self.engine_pred_move, args=(STOCKFISH, self.predict_label1))
        t2 = threading.Thread(target=self.engine_pred_move, args=(KOMODO, self.predict_label2))
        t1.start()
        t2.start()

    def engine_pred_move(self, engine_name, label):
        label.setText('predicting moves..')
        label.adjustSize()
        moves = []
        count = 0
        predict_board = self.board.copy()
        while not predict_board.is_game_over() and count < 10:
            result = ENGINES[engine_name].play(predict_board, chess.engine.Limit(time=0.1))
            move = result.move
            moves.append(move)
            predict_board.push(move)
            count += 1
        right_color = (PLAYER == 'white' and self.board.turn == chess.WHITE) or \
                      (PLAYER == 'black' and self.board.turn == chess.BLACK)
        if engine_name == STOCKFISH and len(moves) > 0 and right_color:
            if len(moves) == 1 or len(moves) == 2:
                time.sleep(0.6)
            self.pred_arrows(moves[0])
        moves = self.board.variation_san(moves)
        NEXT_MOVES = f'{engine_name} predicted moves:\n'
        label.setText(NEXT_MOVES + moves)
        label.adjustSize()

    def pred_arrows(self, move):
        #TODO Fix the Godamn threading problem that crashes the whole program
        self.arrows = [(move.from_square, move.to_square)] if move else None
        self.drawBoard()


    def check_promotion(self, coordinates):
        P = self.pieceToMove[0].symbol()
        if P == 'P' and self.board.turn:
            return coordinates[-1] == '8'
        elif P == 'p' and not self.board.turn:
            return coordinates[-1] == '1'
        return False

    def choose_promotion(self):
        items = {"Queen": 'q', "Rook": 'r', "Bishop": 'b', "Knight": 'n'}
        item, okPressed = QInputDialog.getItem(self, "Promotion", "Choose the piece to promote to:", items.keys(), 0,
                                               False)
        if okPressed and item:
            return items[item]
        else:
            return None

    def check_position(self):
        status = str(self.board.status())
        print(status)
        if self.board.status() != chess.STATUS_VALID:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText(status)
            msg.exec_()
            return False
        elif self.board.is_checkmate():
            self.result_label.setText('checkmate!')
            self.result_label.adjustSize()
        elif self.board.is_stalemate():
            self.result_label.setText('stalemate!')
            self.result_label.adjustSize()
        elif self.board.is_insufficient_material():
            self.result_label.setText('Draw by insufficient material!')
            self.result_label.adjustSize()
        else:
            self.result_label.setText('')
            self.result_label.adjustSize()

    @pyqtSlot()
    def take_back(self):
        try:
            self.board.pop()
            self.predict_threads()
            threading.Thread(target=self.check_position).start()
            self.drawBoard()
        except IndexError as e:
            print('Cannot takeback!')

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
                    if self.second_click is not None:
                        if self.check_promotion(coordinates):
                            new_piece = self.choose_promotion()
                            if new_piece:
                                move = chess.Move.from_uci("{}{}{}".format(self.pieceToMove[1], coordinates, new_piece))
                            else:
                                return
                        else:
                            if self.pieceToMove[1] == coordinates:
                                print('illegal move!!')
                                return
                            move = chess.Move.from_uci("{}{}".format(self.pieceToMove[1], coordinates))
                        if move in self.board.legal_moves:
                            self.board.push(move)
                            self.predict_threads()
                            self.arrows = []
                            piece = None
                            coordinates = None
                            self.drawBoard()
                            threading.Thread(target=self.check_position).start()
                    self.pieceToMove = [piece, coordinates]

    def render_lock(self, board, lastmove, check):
        global LOCK
        LOCK.acquire()
        try:
            self.boardSvg = chess.svg.board(board=board, size=self.boardSize, arrows=self.arrows,
                                            lastmove=lastmove, check=check).encode("UTF-8")
            self.drawBoardSvg = self.widgetSvg.load(self.boardSvg)
        finally:
            LOCK.release()

    def drawBoard(self):
        """
        Draw a chessboard with the starting position and then redraw
        it for every new move.
        """
        board = self.board
        lastmove = board.peek() if board.move_stack else None
        check = board.king(board.turn) if board.is_check() else None
        self.render_lock(board, lastmove, check)
        return self.drawBoardSvg

def load(matrix = None):
    print(sys.executable)
    matrix = [
        ['.', '.', '.', '.', '.', '.', 'k', '.'],
        ['.', 'p', '.', '.', '.', 'p', 'p', 'p'],
        ['.', '.', '.', '.', '.', '.', '.', '.'],
        ['.', '.', 'p', '.', 'p', '.', 'r', '.'],
        ['.', '.', '.', '.', '.', '.', '.', 'n'],
        ['.', 'P', 'N', '.', '.', '.', '.', '.'],
        ['R', '.', 'P', 'r', '.', 'P', 'P', 'P'],
        ['.', '.', '.', '.', 'R', '.', 'K', '.']

    ] if not matrix else matrix


    cnvrtr = FEN_convertor(matrix)
    result = cnvrtr.convert(PLAYER)
    print(result)
    # result = "4N2R/4P3/2b1Q1NP/4P3/p4P2/p2K4/4r3/n2k2B1 w - - 0 1"

    OG = chess.Board(result)
    global CHESSGUI
    if not CHESSGUI:
        CHESSGUI = QApplication(sys.argv)
    else:
        CHESSGUI.exit()
        CHESSGUI = QApplication(sys.argv)

    window = MainWindow(OG)
    window.show()
    sys.exit(CHESSGUI.exec_())



if __name__ == "__main__":
   load()
