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

    def uct_value(self, exploration_weight=1.41):
        if self.visits == 0:
            return float('inf')
        return (self.value / self.visits) + exploration_weight * np.sqrt(np.log(self.parent.visits) / self.visits)

    def best_child(self, exploration_weight=1.41):
        return max(self.children, key=lambda child: child.uct_value(exploration_weight))

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

def evaluate_state(state):
    """
    Advanced state evaluation function:
    - Counts controlled cells.
    - Evaluates potential win lines and threats.
    - Prioritizes strategic positions.
    """
    score = 0
    player = state.player_to_move

    # Controlled cells
    score += sum(1 for cell in state.global_cells if cell == player)
    score -= sum(1 for cell in state.global_cells if cell == -player)

    # Strategic positions (corners, center, edges)
    strategic_positions = [0, 2, 4, 6, 8]  # Indices of corners and center
    score += sum(2 for idx in strategic_positions if state.global_cells[idx] == player)
    score -= sum(2 for idx in strategic_positions if state.global_cells[idx] == -player)

    # Evaluate potential lines
    for line in state.global_cells.reshape(3, 3):
        player_count = sum(1 for cell in line if cell == player)
        opponent_count = sum(1 for cell in line if cell == -player)
        if player_count > 0 and opponent_count == 0:
            score += player_count ** 2
        elif opponent_count > 0 and player_count == 0:
            score -= opponent_count ** 2

    return score

def rollout(state):
    while not state.game_over:
        valid_moves = state.get_valid_moves
        if not valid_moves:
            break

        # Prioritize winning moves
        for move in valid_moves:
            temp_state = copy.deepcopy(state)
            temp_state.act_move(move)
            if temp_state.game_result(temp_state.global_cells.reshape(3, 3)) == state.player_to_move:
                state.act_move(move)
                return state.player_to_move

        # Fallback: Use evaluation-based move prioritization
        move_scores = [(move, evaluate_state(copy.deepcopy(state))) for move in valid_moves]
        move = max(move_scores, key=lambda x: x[1])[0]
        state.act_move(move)

    result = state.game_result(state.global_cells.reshape(3, 3))
    return 0 if result is None else result

def backpropagate(node, result):
    while node:
        node.visits += 1
        if result == 0:
            node.value += evaluate_state(node.state)
        else:
            node.value += result if node.state.player_to_move == -1 else -result
        node = node.parent

def mcts_search(state, itermax, time_limit):
    root = Node(state)
    start_time = time.time()

    def simulate():
        node = root
        while node.is_fully_expanded() and node.children:
            node = node.best_child()

        if not node.is_fully_expanded() and not node.state.game_over:
            node = node.expand()

        simulation_result = rollout(copy.deepcopy(node.state))
        backpropagate(node, simulation_result)

    with ThreadPoolExecutor(max_workers=4) as executor:  # Use 4 threads for parallel simulations
        futures = [executor.submit(simulate) for _ in range(itermax)]
        for future in futures:
            if time.time() - start_time > time_limit:
                break
            future.result()

    return root.best_child(0).move

def select_move(cur_state, remain_time):
    time_limit = min(remain_time, 10)
    itermax = int(5000 * time_limit)  # Dynamic iteration scaling
    return mcts_search(cur_state, itermax, time_limit)
