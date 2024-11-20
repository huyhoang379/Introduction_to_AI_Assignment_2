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


def is_winning_move(state, move, player):
    test_state = copy.deepcopy(state)
    test_state.act_move(move)
    result = test_state.game_result(test_state.global_cells.reshape(3, 3))
    return result == player


def is_dangerous_move(state, move):
    opponent = -state.player_to_move
    test_state = copy.deepcopy(state)
    test_state.act_move(move)
    valid_moves = test_state.get_valid_moves
    for opponent_move in valid_moves:
        if is_winning_move(test_state, opponent_move, opponent):
            return True
    return False


def select_best_move(state):
    valid_moves = state.get_valid_moves
    player = state.player_to_move

    for move in valid_moves:
        if is_winning_move(state, move, player):
            return move

    safe_moves = [move for move in valid_moves if not is_dangerous_move(state, move)]
    if safe_moves:
        return random.choice(safe_moves)
    else:
        return random.choice(valid_moves)


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
