from .player import Player
from .pfhc_game import PreFlopHighCardGame


class HighCardEnv:
    """
    Environment class for the pre-flop high card poker game.
    In every round, each of the player gets a single card.
    """
    INITIAL_STACK = 100

    def __init__(self, player1 : Player, player2 : Player):
        self.player1 = player1
        self.player2 = player2
        self.pfhc_game: PreFlopHighCardGame = None
        self.reset()

    def step(self, action):
        """
        :param action: Tuple containing a char ['b', 'f', 'c'] and a positive integral value.
                       The latter is only taken into account if 'b' is chosen, as it is the bet size.
        :return:
        """
        return

    def reset(self):
        self.player1.reset()
        self.player2.reset()
        self.pfhc_game = PreFlopHighCardGame(self.player1, self.player2)

    def __str__(self):
        return
