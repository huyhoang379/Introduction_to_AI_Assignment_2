import random
import time
import copy
import os
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
            valid_moves = self.state.get_valid_moves
            if valid_moves:
                move = random.choice(valid_moves)
                next_state = copy.deepcopy(self.state)
                next_state.act_move(move)
                child_node = Node(state=next_state, parent=self, move=move)
                self.children.append(child_node)
                return child_node
            else:
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


def rollout(state):
    while not state.game_over:
        valid_moves = state.get_valid_moves
        if not valid_moves:
            break
        move = random.choice(valid_moves)
        state.act_move(move)

    result = state.game_result(state.global_cells.reshape(3, 3))
    if result is None:
        return 0
    return result


def backpropagate(node, result):
    while node:
        node.visits += 1
        node.value += result if node.state.player_to_move == -1 else -result
        node = node.parent


def simulate(node):
    simulation_result = rollout(copy.deepcopy(node.state))
    return simulation_result


def mcts_search(state, itermax, time_limit):
    root = Node(state)
    start_time = time.time()
    # max_workers = 6  # Adjust based on available threads or system capacity
    max_workers = os.cpu_count()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for _ in range(itermax):
            if time.time() - start_time > time_limit:
                break

            # Selection
            node = root
            while node.is_fully_expanded() and node.children:
                node = node.best_child()

            # Expansion
            if not node.is_fully_expanded() and not node.state.game_over:
                node = node.expand()

            if node:
                # Parallel Simulation
                futures = [
                    executor.submit(simulate, node) for _ in range(max_workers)
                ]

                # Collect Results
                for future in futures:
                    result = future.result()
                    backpropagate(node, result)

    if not root.children:
        valid_moves = root.state.get_valid_moves
        if not valid_moves:
            return None
            
    return root.best_child(0).move


def check_block_win(state, move):
    """
    Check if making a move will win the block.
    """
    temp_state = copy.deepcopy(state)
    local_board = temp_state.blocks[move.index_local_board]
    local_board[move.x][move.y] = move.value
    return temp_state.game_result(local_board) == move.value


def check_block_threat(state, move):
    """
    Check if making a move will prevent the opponent from winning the block.
    """
    temp_state = copy.deepcopy(state)
    local_board = temp_state.blocks[move.index_local_board]
    local_board[move.x][move.y] = -move.value
    return temp_state.game_result(local_board) == -move.value


def select_move(cur_state, remain_time):
    start_time = time.time()  # Bắt đầu đo thời gian
    time_limit = min(remain_time, 10)  # Đảm bảo không vượt quá giới hạn 10 giây
    itermax = 100  # Điều chỉnh số lần lặp dựa trên hiệu năng
    
    # Lấy danh sách các nước đi hợp lệ
    valid_moves = cur_state.get_valid_moves

    # Bước 1: Ưu tiên các nước đi giúp thắng block
    for move in valid_moves:
        if check_block_win(cur_state, move):
            return move

    # Bước 2: Ưu tiên các nước đi chặn đối thủ thắng block
    for move in valid_moves:
        if check_block_threat(cur_state, move):
            return move 

    # Tính thời gian còn lại
    elapsed_time = time.time() - start_time
    remaining_time_for_mcts = time_limit - elapsed_time

    # Nếu không còn thời gian, trả về nước đi ngẫu nhiên
    if remaining_time_for_mcts <= 0:
        return random.choice(valid_moves)

    # Bước 3: Chạy MCTS với thời gian còn lại
    return mcts_search(cur_state, itermax, remaining_time_for_mcts)

