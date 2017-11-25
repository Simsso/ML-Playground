import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
from .agent import Agent
from .card import Card


class HighCardEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    starting_stack = 100
    small_blind_fee = 1
    big_blind_fee = 2

    def __init__(self):
        self._reset()

    def _step(self, action):
        """
        Environment step
        :param action: (fold/call/raise, amount). (fold/call/raise) is a value between 0 and 2.
                       Amount is a scalar between 0 and 1. 1 corresponds to an all-in.
        :return:
        """

        action_code: int = action[0]
        raise_portion: float = action[1]  # percentage of stack
        if raise_portion < 0 or raise_portion > 1:
            raise ValueError("Invalid raise portion value. Legal values are in [0,1], found " + str(raise_portion))

        # fold
        if action_code == 0:
            self._turn.fold()

        # call
        amount_to_call: int = self._amount_to_call(self._turn)
        if action_code == 1:
            actual_call_amount = min(amount_to_call, self._turn.get_amount())  # all in
            self._turn.take(actual_call_amount)
            self._pot += actual_call_amount

        # raise
        if action_code == 2:
            raise_amount = max(amount_to_call + self._last_bet_size, int(self._turn.get_amount() * raise_portion))
            # player all-in check
            raise_amount = min(raise_amount, self._turn.get_amount())  # all in
            self._last_bet_size = raise_amount - amount_to_call
            self._turn.take(raise_amount)
            self._pot += raise_amount

        reward = 0
        game_over = self._is_game_over()
        if game_over:
            player_index = self._players.index(self._turn)
            payout = self._get_payouts()[player_index]
            reward = self._turn.get_amount() + payout

        self._next_player()
        return self.get_state(self._turn), reward, game_over, {}

    def _reset(self):
        """
        Resets the environment to its default state.
        Each player will be given new cards, stack sizes will be reset, etc.
        """
        self._players = []
        self._pot = HighCardEnv.small_blind_fee + HighCardEnv.big_blind_fee
        self._highest_bet = HighCardEnv.big_blind_fee

        # create player objects
        player_a = Agent("Player A", self.starting_stack, Card.random_card())
        player_b = Agent("Player B", self.starting_stack, Card.random_card())
        self._players.append(player_a)
        self._players.append(player_b)

        # take blinds
        player_a.take(HighCardEnv.small_blind_fee)
        player_b.take(HighCardEnv.big_blind_fee)
        self._last_bet_size = HighCardEnv.big_blind_fee

        # small blind starts
        self._turn: Agent = player_a

    def _next_player(self):
        """
        Updates the _turn member variable which points to the player object that is acting next.
        """
        index = (self._players.index(self._turn) + 1) % len(self._players)
        self._turn = self._players[index]

    def _amount_to_call(self, player: Agent):
        """
        :param player: A player participating in the game.
        :return: The amount that is missing for the player to stay in the game.
        """
        bet_so_far = player.get_amount()
        missing = self._highest_bet - bet_so_far
        if missing < 0:
            raise ValueError("Internal state corrupted. The highest bet size is lower than the "
                             "amount of money that the player has in the pot.")
        return missing

    def _is_game_over(self):
        max_amount_placed = 0
        for player in self._players:
            if player.has_folded():
                continue

            # players have place equal amount of money
            if max_amount_placed == self._amount_placed(player) or max_amount_placed == 0:
                max_amount_placed = self._amount_placed(player)
            else:
                return False
        return True

    def _is_winner(self, player):
        for other_player in self._players:
            if other_player is player:
                continue
            if other_player.has_folded():
                continue
            if other_player.get_card().loses_against(player.get_card()):
                continue
            return False

    def _get_winning_players(self):
        winning_players = []
        for player in self._players:
            if self._is_winner(player):
                winning_players.append(player)
        return winning_players

    def _get_payouts(self):
        """
        Amount of money that each player will get from the pot.
        :return: Array, where each entry corresponds to the same index in the _players array.
        """
        winning_players = self._get_winning_players()

        payouts = []
        for player in self._players:
            amount = 0
            if winning_players.index(player) != -1:
                amount = self._pot / len(winning_players)
            payouts.append(amount)

        return payouts

    def _render(self, mode='human', close=False):
        return

    def _get_stack_sizes(self, except_for):
        stack_sizes = []
        for player in self._players:
            if player is except_for:
                continue
            stack_sizes.append(player.get_amount())
        return stack_sizes

    def get_state(self, player):
        """
        :param player: Perspective of this player.
        :return: The game state from one player's point of view.
                 [pot size, stack sizes of all other players, player stack, player card rank]
        """
        return np.array([self._pot, self._get_stack_sizes()[0], player.get_amount(), player.get_card().get_rank()])

    @staticmethod
    def _amount_placed(self, player: Agent):
        """
        Amount of money that a player has put into the pot this round.
        """
        return HighCardEnv.starting_stack - player.get_amount()
