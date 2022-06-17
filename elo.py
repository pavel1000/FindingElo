import chess.pgn
from sklearn.feature_extraction import DictVectorizer
from collections import defaultdict, Counter
import chess
from chess import QUEEN
import pickle
import sys
import numpy as np
import pandas as pd

def get_games(filename, n_games=sys.maxsize):
    '''Считывает данные о партиях из файла "filename".
    При этом количество считываемых игр ограниченно параметром n_games'''
    try:
        with open(filename) as pgn:
            game = chess.pgn.read_game(pgn)
            count = 0
            while game and count < n_games:
                count += 1
                yield game
                game = chess.pgn.read_game(pgn)
    except:
        with open(filename, encoding="utf-8") as pgn:
            game = chess.pgn.read_game(pgn)
            count = 0
            while game and count < n_games:
                count += 1
                yield game
                game = chess.pgn.read_game(pgn)


def extract_game_features(game):
    """Извлекает свойства из игры (game)"""
    # Флаги, необходимые для записи соответствующих первых значений
    first_white_check = True
    first_white_queen_move = True
    first_black_check = True
    first_black_queen_move = True

    features = defaultdict(int)
    node = game
    stockfish_scores = []

    while node.variations:  # and node.board().fullmove_number < cut_off:
        move = node.variation(0).move
        score = game.variation(0).eval().white().score()
        if score is None:
            # score может быть None, если движок видит мат
            if game.variation(0).eval().is_mate():
                # Проверяем какой стороне грозит мат
                moves_to_checkmate = game.variation(0).eval().white().mate()
                # В случае, когда движок видит мат - повышаю значимость позиции
                # Использую значение 1300, чтобы иметь запас "поощерительных очков"
                if moves_to_checkmate>0:
                    stockfish_scores.append(1300-moves_to_checkmate*10)
                else:
                    stockfish_scores.append(-1300-moves_to_checkmate*10)
            else:
                # Если мата нет, то это пропущенное значение
                if stockfish_scores == []:
                    stockfish_scores.append(0)
                else:
                    stockfish_scores.append(stockfish_scores[-1])
        elif abs(score)>1000:
            if board.turn:
                stockfish_scores.append(1000)
            else:
                stockfish_scores.append(-1000)
        else:
            stockfish_scores.append(game.variation(0).eval().white().score())
        board = node.board()

        # Фигура с начальной позиции хода (откуда шагнула)
        moved_piece = board.piece_type_at(move.from_square)
        # Фигура с конечной позиции хода (куда шагнула)
        captured_piece = board.piece_type_at(move.to_square)

        # Сохраняем номер хода, на котором произошло первое движение ферзя
        if moved_piece == QUEEN:
            if board.turn and first_white_queen_move:
                features['white_queen_moved_at'] = board.fullmove_number
                first_white_queen_move = False
            elif (not board.turn) and first_black_queen_move:
                features['black_queen_moved_at'] = board.fullmove_number
                first_black_queen_move = False
        # Ход взятия ферзя
        if captured_piece == QUEEN:
            if board.turn:
                features['black_queen_taken_at'] = board.fullmove_number
                if first_black_queen_move:
                    # Если ферзь до взятия не двигался, то установим ход его
                    # первого движения как ход взятия
                    features['black_queen_moved_at'] = board.fullmove_number
                    first_black_queen_move = False
            else:
                features['white_queen_taken_at'] = board.fullmove_number
                if first_white_queen_move:
                    features['white_queen_moved_at'] = board.fullmove_number
                    first_white_queen_move = False
        # Подсчет числа проведенных пешек(?)
        if move.promotion:
            if board.turn:
                features['white_promotion'] += 1
            else:
                features['black_promotion'] += 1
        # подсчет количества шахов
        if board.is_check():
            if board.turn:
                if first_white_check:
                    features['first_white_check_at'] = board.fullmove_number
                    first_white_check = False
                features['total_white_checks'] += 1
            else:
                if first_black_check:
                    features['first_black_check_at'] = board.fullmove_number
                    first_black_check = False
                features['total_black_checks'] += 1
        # рокировка
        if board.is_kingside_castling(move):
            if board.turn:
                features['white_king_castle'] = board.fullmove_number
            else:
                features['black_king_castle'] = board.fullmove_number
        elif board.is_queenside_castling(move):
            if board.turn:
                features['white_queen_castle'] = board.fullmove_number
            else:
                features['black_queen_castle'] = board.fullmove_number
        node = node.variation(0)
    # Если ферзи не двигались и не были срублены, то устанавливаем им в качестве значений
    # последний ход партии.
    if first_white_queen_move:
        features['white_queen_moved_at'] = board.fullmove_number
    if first_black_queen_move:
        features['black_queen_moved_at'] = board.fullmove_number
    # Общее количество шахов
    features['total_checks'] = features['total_white_checks'] + features['total_black_checks']
    features['promotion'] = features['white_promotion'] + features['black_promotion']
    # Добавляем оценки стокфиша для партий
    #features['stockfish_scores'] = stockfish_scores
    # Проверяем позицию на наличие мата
    if board.is_checkmate():
        features['is_checkmate'] += 1
    # Проверяем позицию на наличие пата
    if board.is_stalemate():
        features['is_stalemate'] += 1
    # Проверяем позицию на наличие достаточного количества выигрышного материала
    if board.is_insufficient_material():
        features['insufficient_material'] += 1
    # Проверяем может ли игрок, сделавший последний ход, претендовать на ничью
    # по правилу 50 ходов или из-за троекратного повторения
    if board.can_claim_draw():
        features['can_claim_draw'] += 1
    # Сохраняем количество ходов партии
    features['total_moves'] = board.fullmove_number

    # Подсчет количества фигур в конечной позиции
    piece_placement = board.board_fen()
    end_pieces = Counter(x for x in piece_placement if x.isalpha())
    features.update({'end_' + piece: cnt
                     for piece, cnt in end_pieces.items()})
    return features, stockfish_scores


def games_features(games):
    '''Для каждой партии получает список характеристик'''
    # Проходимся по всем играм и сохраняем их характеристики в список
    features = []
    stockfish = []
    for game in games:
        f, s  = extract_game_features(game)
        features.append(f)
        stockfish.append(s)
    if features == []:
        return pd.DataFrame()
    # Преобразуем список характеристик в numpy-массивы,
    # параллельно производя one_hot_encoding для строковых значений.
    vec = DictVectorizer()
    X = vec.fit_transform(features)
    df_X = pd.DataFrame(X.toarray(), columns=vec.get_feature_names_out())
    df_stockfish = pd.DataFrame(np.array(stockfish, dtype=object).T, columns=['stockfish_scores'])
    return df_X.join(df_stockfish, how='outer')

def goodmovecounts(evals, color, threshold, result, move):
    if color == 'white':
        moves = 0
        for j in range(2, len(evals), 2):
            if move == 'good':
                if evals[j] - evals[j-1] > threshold:
                    moves+=1
            elif move == 'bad':
                if evals[j] - evals[j-1] < threshold:
                    moves+=1
        if result != 'count':
            moves = moves/(len(evals)/2)
    elif color == 'black':
        moves = 0
        for j in range(1, len(evals), 2):
            if move == 'good':
                if evals[j] - evals[j-1] < threshold:
                    moves+=1
            elif move == 'bad':
                if evals[j] - evals[j-1] > threshold:
                    moves+=1
        if result != 'count':
            moves = moves/(len(evals)/2)
    return moves

def partitiondiffs(stockfishScores, pieces, eval_cutoff = 10000, first_halfmove = 1, last_halfmove = 1000):
    if pieces == 'white':
        gameDiffs = []
        if len(stockfishScores) > first_halfmove:
            for moveIndex in range(1+first_halfmove, (min(len(stockfishScores), last_halfmove)),2):
                if ((-eval_cutoff) < stockfishScores[moveIndex] < eval_cutoff):
                    gameDiffs.append(stockfishScores[moveIndex] - stockfishScores[moveIndex-1])        
        return gameDiffs
    
    elif pieces == 'black':
        gameDiffs = []
        if len(stockfishScores) > first_halfmove:
            for moveIndex in range(first_halfmove, (min(len(stockfishScores), last_halfmove)),2):
                if ((-eval_cutoff) < stockfishScores[moveIndex] < eval_cutoff):
                    gameDiffs.append(-1*(stockfishScores[moveIndex] - stockfishScores[moveIndex-1]))    
        return gameDiffs

def extract_score_features(scores):
    features = dict()
    if len(scores) == 0:
        return features
    # Добавляем 0 перед оценками стокфиша
    scores = np.r_[0, scores]
    # Вычисляем разницу последовательных оценок стокфиша
    diffs = np.diff(scores)
    abs_diffs = np.abs(diffs)
    white_diffs = diffs[::2]
    black_diffs = diffs[1::2]
    # Для полученных разностей вычисляем минимум максимум, среднее и медиану
    subset_names = ['diffs', 'abs_diffs', 'white_diffs', 'black_diffs']
    subsets = [diffs, abs_diffs, white_diffs, black_diffs]
    stats = [np.min, np.max, np.std, np.mean, lambda x: np.median(np.abs(x))]
    stat_names = ['min', 'max', 'std', 'mean', 'median_abs']
    for subset, subset_name in zip(subsets, subset_names):
        for stat, stat_name in zip(stats, stat_names):
            features[stat_name + '_' + subset_name] = stat(subset)
    # Сохраняем первый ход, на котором преимущество было больше заданного значения
    # или последний ход, если такого преимущества не было
    # 100 соответсвует 70% вероятности выигрыша, 275 - 90% вероятность
    abs_scores = np.abs(scores)
    features['advantage100_first'] = np.argmin(abs_scores > 100) or len(scores)
    features['advantage300_first'] = np.argmin(abs_scores > 275) or len(scores)
    # Сохраняем последний ход, на котором преимущество было больше
    # заданного значения или 0, если такого преимущества не было
    features['advantage100_last'] = np.argmax(abs_scores > 100) or 0
    features['advantage300_last'] = np.argmax(abs_scores > 275) or 0

    # Вычисляем "хорошие" ходы с параметрами "share" и "count"
    features['goodmovesw'] = goodmovecounts(scores, 'white', -10, 'share','good')
    features['goodmovesb'] = goodmovecounts(scores, 'white', 10, 'share','good')
    features['goodmovescw'] = goodmovecounts(scores, 'white', -10, 'count','good')
    features['goodmovescb'] = goodmovecounts(scores, 'white', 10, 'count','good')
    
    # Вычисляем "плохие" ходы с параметрами "share" и "count"
    features['blundersw'] = goodmovecounts(scores, 'white', -100, 'share','bad')
    features['blundersb'] = goodmovecounts(scores, 'white', 100, 'share','bad')
    features['blunderscw'] = goodmovecounts(scores, 'white', -100, 'count','bad')
    features['blunderscb'] = goodmovecounts(scores, 'white', 100, 'count','bad')
    
    #features['deltaw1'] = partitiondiffs(scores, 'white', first_halfmove = 1, last_halfmove = 20)
    #features['deltaw2'] = partitiondiffs(scores, 'white', first_halfmove = 21, last_halfmove = 40)
    #features['deltaw3'] = partitiondiffs(scores, 'white', first_halfmove = 41, last_halfmove = 60)
    #features['deltaw4'] = partitiondiffs(scores, 'white', first_halfmove = 61, last_halfmove = 90)
    #features['deltaw5'] = partitiondiffs(scores, 'white', first_halfmove = 91)
    
    #features['deltab1'] = partitiondiffs(scores, 'black', first_halfmove = 1, last_halfmove = 20)
    #features['deltab2'] = partitiondiffs(scores, 'black', first_halfmove = 21, last_halfmove = 40)
    #features['deltab3'] = partitiondiffs(scores, 'black', first_halfmove = 41, last_halfmove = 60)
    #features['deltab4'] = partitiondiffs(scores, 'black', first_halfmove = 61, last_halfmove = 90)
    #features['deltab5'] = partitiondiffs(scores, 'black', first_halfmove = 91)

    return features


def score_features(stockfish_scores):
    stockfish_features = []
    for s in stockfish_scores:
        stockfish_features.append(extract_score_features(s))
    vec = DictVectorizer()
    X = vec.fit_transform(stockfish_features)
    df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names_out())
    return df
