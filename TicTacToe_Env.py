import numpy as np
from copy import deepcopy
import gym
from rl.agents.dqn import DQNAgent
import random


class TicTacToeEnv(gym.Env):
    def __init__(self, size: int = 3, agent: DQNAgent = None):
        self.name = 'TicTacToe'
        self._size = size
        self._board = np.array([[0] * self._size for _ in range(3)])
        self._agent = agent  # opponent
        self._symbol_map = {
            -1: 'X',
            0: '-',
            1: 'O'
        }
        self._legal_move_mask = np.array([1]*self._size**2)

        # base class setup
        self.nS = self._size ** 2
        self.nA = self.nS
        self.action_space = gym.spaces.Discrete(self.nA)

    def render(self, mode=0):
        screen = deepcopy(self._board)
        for i in range(self._size):
            for j in range(self._size):
                print(self._symbol_map[self._board[j, i]], end=' ')
            print('')
        print('')
        return screen

    def reset(self):
        self._board = np.array([[0] * self._size for _ in range(3)])
        self._legal_move_mask = np.array([1] * (self._size ** 2))

        # randomly start the player on either side
        if random.uniform(0, 1) > .5:
            self._play_opponent()

        return self._board.flatten()

    def _play_opponent(self):
        if self._agent is not None:
            q_values = self._agent.compute_q_values([self._board.flatten()])
            legal_q_values = q_values * self._legal_move_mask
            opp_action = np.argmax(legal_q_values)
            x, y = self._play_move(opp_action)
            return self._game_ended(x, y)

    def step(self, action: int):
        if self._legal_move_mask[action] == 0:
            return self._board.flatten(), -1, False, {'prob': 'place holder'}
        x, y = self._play_move(action)
        if self._game_ended(x, y):
            return self._board.flatten(), 1, True, {'prob': 'place holder'}

        ended = self._play_opponent()
        if ended:
            return self._board.flatten(), -1, True, {'prob': 'place holder'}

        return self._board.flatten(), 0, False, {'prob': 'place holder'}

    def _play_move(self, action: int):
        # check action validity
        if not self.action_space.contains(action):
            raise Exception("unknown action {}".format(int(action)))

        # perform action
        x = action % 3
        y = int(action / 3)
        if self._board[x, y] != 0:
            raise Exception("illegal move on ({}, {})\n board {}\n {}".format(x, y, self._board, self._legal_move_mask))
        self._board[x, y] = 1
        self._legal_move_mask[action] = 0
        self._flip_board()
        return x, y

    def _flip_board(self):
        for i in range(self._size):
            for j in range(self._size):
                self._board *= -1

    # Checks if the game is ended
    def _game_ended(self, x: int, y: int):
        if abs(self._board[0, 0] + self._board[1, 1] + self._board[2, 2]) == 3:
            return True
        if abs(self._board[2, 0] + self._board[1, 1] + self._board[0, 2]) == 3:
            return True
        if abs(self._board[x, 0] + self._board[x, 1] + self._board[x, 2]) == 3:
            return True
        if abs(self._board[0, y] + self._board[1, y] + self._board[2, y]) == 3:
            return True
        a = sum(self._legal_move_mask)
        if a == 0:
            return True
        return False
