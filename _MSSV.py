import random
import time
import copy
# import os
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
    max_workers = 4  # Adjust based on available threads or system capacity
    # max_workers = os.cpu_count()

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


def select_move(cur_state, remain_time):
    time_limit = min(remain_time, 10)
    itermax = 100
    return mcts_search(cur_state, itermax, time_limit)

