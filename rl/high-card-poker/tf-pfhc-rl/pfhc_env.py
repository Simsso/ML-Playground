from player import Player
from pfhc_game import PreFlopHighCardGame


class HighCardEnv:
    """
    Environment class for the pre-flop high card poker game.
    In every round, each of the player gets a single card.
    """
    INITIAL_STACK = 100

    def __init__(self, player1 : Player, player2 : Player):
        self.p1 = player1
        self.p2 = player2
        self.pfhc_game: PreFlopHighCardGame = None
        self.reset()

    def step(self, action):
        """
        :param action: Tuple containing a char ['b', 'f', 'c'] and a positive integral value.
                       The latter is only taken into account if 'b' is chosen, as it is the bet size.
        :return: Tuple (new state, rewards, game over, misc). Rewards is an array with two elements,
                 one reward for each player. It is [0, 0] unless a payout has happened.
                 The new state is, as seen from the perspective of the player who is acting next.
                 This might be the same player as the one who is conducting the current action.
        """
        action_code = action[0]
        bet_size = action[1]
        game = self.pfhc_game
        player = game.next_player
        count_ba = game.round_count  # round count before action

        if action_code == 'b':
            # scale bet size to [0, player stack]
            game.bet(amount=max(0, min(player.stack(), bet_size)))
        elif action_code == 'c':
            game.call()
        else:
            game.fold()

        reward = [0, 0]
        if count_ba < game.round_count:  # new round (showdown has taken place or a player folded)
            reward = game.last_gain

        return self.get_state(game.next_player), reward, game.over, {}

    def reset(self):
        """
        Resets the environment. That is resetting the players (but keeping the object instances) and creating a new
        pre-flop high card game object.
        """
        self.p1.reset()
        self.p2.reset()
        self.pfhc_game = PreFlopHighCardGame(self.p1, self.p2)

    def get_state(self, me: Player):
        """
        :return: The private game state, i.e. game as seen from one player's point of view.
                 A 5-tuple containing (card, stack, opponent stack, bet size, opponent bet size)
        """
        game = self.pfhc_game
        opponent = game.other(me)
        state = [me.get_card().get_rank(), me.stack(), opponent.stack(), me.bet_size(), opponent.bet_size()]
        return state

    def __str__(self):
        return str(str(self.p1.name) + "'s state: " + str(self.get_state(self.p1))) + \
            str(str(self.p2.name) + "'s state: " + str(self.get_state(self.p2)))