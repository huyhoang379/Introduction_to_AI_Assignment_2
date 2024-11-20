import random
import time
import copy
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class Node:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.value = 0

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_valid_moves)

    def best_child(self, exploration_weight=1.41):
        if not self.children:
            return None
        choices_weights = [
            (child.value / child.visits) + exploration_weight * np.sqrt(
                np.log(self.visits) / child.visits
            )
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def expand(self):
        untried_moves = [
            move for move in self.state.get_valid_moves
            if move not in [child.move for child in self.children]
        ]
        if untried_moves:
            move = random.choice(untried_moves)
            next_state = copy.deepcopy(self.state)
            next_state.act_move(move)
            child_node = Node(state=next_state, parent=self, move=move)
            self.children.append(child_node)
            return child_node
        return None


# def is_winning_move(state, move, player):
#     test_state = copy.deepcopy(state)
#     test_state.act_move(move)
#     result = test_state.game_result(test_state.global_cells.reshape(3, 3))
#     return result == player


def is_winning_line_in_block(state, player, block_index):
    # Xác định góc trên bên trái của block
    start_row, start_col = (block_index // 3) * 3, (block_index % 3) * 3
    block_cells = state.global_cells[start_row:start_row+3, start_col:start_col+3].flatten()

    # Kiểm tra từng dòng
    for i in range(3):
        line = block_cells[i*3:(i+1)*3]
        if list(line).count(player) == 2 and list(line).count(0) == 1:
            y = list(line).index(0)  # Tọa độ cột trong block
            x = i  # Tọa độ dòng trong block
            return UltimateTTT_Move(index_local_board=block_index, x=x, y=y, value=player)

    # Kiểm tra từng cột
    for i in range(3):
        column = block_cells[i::3]
        if list(column).count(player) == 2 and list(column).count(0) == 1:
            x = list(column).index(0)  # Tọa độ dòng trong block
            y = i  # Tọa độ cột trong block
            return UltimateTTT_Move(index_local_board=block_index, x=x, y=y, value=player)

    # Kiểm tra đường chéo chính
    diag1 = block_cells[0::4]  # Các phần tử từ 0, 4, 8
    if list(diag1).count(player) == 2 and list(diag1).count(0) == 1:
        idx = list(diag1).index(0)  # Chỉ số trong đường chéo
        x, y = idx, idx  # Tọa độ dòng và cột giống nhau
        return UltimateTTT_Move(index_local_board=block_index, x=x, y=y, value=player)

    # Kiểm tra đường chéo phụ
    diag2 = block_cells[2:7:2]  # Các phần tử từ 2, 4, 6
    if list(diag2).count(player) == 2 and list(diag2).count(0) == 1:
        idx = list(diag2).index(0)  # Chỉ số trong đường chéo phụ
        x, y = idx, 2 - idx  # Tọa độ dòng và cột ngược nhau
        return UltimateTTT_Move(index_local_board=block_index, x=x, y=y, value=player)

    # Không tìm thấy nước đi hợp lệ
    return None

def select_best_move(state):
    valid_moves = state.get_valid_moves
    player = state.player_to_move
    
    # Xác định block hiện tại từ nước đi trước
    previous_move = state.previous_move
    block_index = previous_move.x * 3 + previous_move.y  # Chỉ số của block

    # Kiểm tra xem có thể chiến thắng trong block hiện tại không
    winning_move = is_winning_line_in_block(state, player, block_index)
    if winning_move is not None and winning_move in valid_moves:
        return winning_move

    # Kiểm tra các nước đi nguy hiểm của đối thủ trong block hiện tại
    opponent = -player
    losing_move = is_winning_line_in_block(state, opponent, block_index)
    if losing_move is not None and losing_move in valid_moves:
        return losing_move

    # Nếu không có chiến thắng hoặc ngăn chặn, trả về None để MCTS xử lý
    return None


def minimax(state, depth, alpha, beta, maximizing_player):
    if depth == 0 or state.game_over:
        return evaluate_state(state)

    if maximizing_player:
        max_eval = -float('inf')
        for move in state.get_valid_moves:
            next_state = copy.deepcopy(state)
            next_state.act_move(move)
            eval = minimax(next_state, depth - 1, alpha, beta, False)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in state.get_valid_moves:
            next_state = copy.deepcopy(state)
            next_state.act_move(move)
            eval = minimax(next_state, depth - 1, alpha, beta, True)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval


def evaluate_state(state):
    result = state.game_result(state.global_cells.reshape(3, 3))
    if result == 1:
        return 100
    elif result == -1:
        return -100
    else:
        return 0


def rollout(state):
    max_depth = 3
    while not state.game_over:
        move = select_best_move(state)
        state.act_move(move)
    result = state.game_result(state.global_cells.reshape(3, 3))
    return result if result is not None else 0


def backpropagate(node, result):
    while node:
        node.visits += 1
        node.value += result if node.state.player_to_move == -1 else -result
        node = node.parent


def single_iteration(root):
    node = root
    while node.is_fully_expanded() and node.children:
        node = node.best_child()

    if not node.is_fully_expanded() and not node.state.game_over:
        node = node.expand()

    simulation_result = rollout(copy.deepcopy(node.state))
    return node, simulation_result


def mcts_search(state, itermax, time_limit):
    root = Node(state)
    start_time = time.time()

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(single_iteration, root) for _ in range(itermax)]
        for future in futures:
            node, simulation_result = future.result()
            backpropagate(node, simulation_result)

    return root.best_child(0).move


def select_move(cur_state, remain_time):
    best_move = select_best_move(cur_state)
    if best_move:
        return best_move

    time_limit = min(remain_time, 10)
    itermax = 100
    return mcts_search(cur_state, itermax, time_limit)
