from random import *


class Card:
    rank_names = ['2', '3', '4', '5', '6', '7', '8', '9', 't', 'j', 'q', 'k', 'a']

    def __init__(self, rank):
        # rank value check
        if rank < 0 or rank > 12:
            raise ValueError("Rank must be an integer in [0,12]")
        self._rank = rank

    def __str__(self):
        return self.rank_names[self._rank]

    def get_rank(self):
        return self._rank

    def loses_against(self, card):
        return card.get_rank() > self.get_rank()

    @staticmethod
    def random_card():
        rank = randint(0, 12)
        card = Card(rank)
        return card
