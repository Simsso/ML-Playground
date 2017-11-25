from .card import Card


class Agent:
    def __init__(self, name, stack_size, card: Card):
        self._name = name
        self._card = card
        self._folded = False

        if stack_size < 0:
            raise ValueError("The stack size must be greater than zero.")
        self._stack_size = stack_size

    def take(self, amount):
        if self._stack_size < amount:
            raise ValueError("Amount is too high.")
        self._stack_size = self._stack_size - amount

    def fold(self):
        self._folded = True

    def get_amount(self):
        return self._stack_size

    def has_folded(self):
        return self._folded

    def get_card(self):
        return self._card