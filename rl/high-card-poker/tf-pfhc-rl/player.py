from card import Card
from strategy import *


class Player:
    def __init__(self, name, stack_size, card: Card = None, strategy: Strategy = StrategyRandom):
        """
        Constructor for a new player object. A player participates in the heads up high card game.
        :param name: The players name.
        :param stack_size: The starting stack size.
        :param card: The first card that the player has.
        """
        self.name = name
        if card is None:
            card = Card.random_card()
        self._card = card
        self._folded = False
        self._bet_size = 0
        self._all_in = False

        if stack_size < 0:
            raise ValueError("The stack size must be greater than zero.")
        self._stack_size = stack_size
        self._initial_stack_size = stack_size

        self.strategy = strategy

    def take(self, amount):
        """
        Removes the given amount from the player's stack.
        Adds the amount to the bet size variable.
        :param amount: The amount to subtract.
        :return: The amount argument.
        """
        if self._stack_size <= amount:
            amount = self._stack_size
            self._all_in = True
        self._stack_size = self._stack_size - amount
        self._bet_size += amount
        return amount

    def fold(self, folded=True):
        """
        Fold, i.e. give up the pod. Changes the folded attribute.
        :param folded: Whether to fold or not.
        """
        self._folded = folded

    def stack(self):
        """
        :return: The player's stack size.
        """
        return self._stack_size

    def set_stack(self, amount):
        if amount < 0:
            raise ValueError("The amount to set the player's stack to must be a positive integer.")
        self._stack_size = amount

    def ship_over(self, amount):
        """
        Adds chips to the players stack
        :param amount: The amount to add to the players stack. Must be a positive integer.
        """
        if amount < 0:
            raise ValueError("The amount to add to the player's stack must be a positive integer.")
        self._stack_size += amount
        if self._stack_size > 0:
            self._all_in = False  # reset all in flag if the stack is now non-zero

    def has_folded(self):
        """
        :return: True if the player has folded, False otherwise.
        """
        return self._folded

    def get_card(self):
        """
        :return: The card which the player is holding at the moment.
        """
        return self._card

    def set_card(self, card: Card):
        """
        Set the player's card.
        :param card: The new card.
        """
        self._card = card

    def reset_bet_size(self):
        self._bet_size = 0

    def bet_size(self):
        return self._bet_size

    def is_all_in(self):
        return self._all_in

    def reset(self):
        self._all_in = False
        self._stack_size = self._initial_stack_size
        self.reset_bet_size()
        self._folded = False