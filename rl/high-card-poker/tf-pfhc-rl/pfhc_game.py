from player import Player
from card import Card


class PreFlopHighCardGame:
    BIG_BLIND = 2
    SMALL_BLIND = 1

    def __init__(self, player1: Player, player2: Player):
        self.p1: Player = player1
        self.p2: Player = player2
        self.button = player2  # corresponds to big blind in heads-up
        self.pot: int = 0
        self.over = False
        self.next_player: Player = None
        self.round_count = 0
        self.last_gain = [0, 0]  # the amount that the players have gained during the last round
        self.next_round()

    def next_round(self):
        # check consistency
        if not ((self.p1.stack() - self.p1.bet_size() + self.p2.stack() - self.p2.bet_size() + self.pot) % 2 == 0):
            raise ValueError("Invalid amount of money in the game")

        if self.p1.stack() < PreFlopHighCardGame.BIG_BLIND or self.p2.stack() < PreFlopHighCardGame.BIG_BLIND:
            self.over = True
            return

        self.round_count += 1
        self.pot = 0
        self.p1.reset_bet_size()
        self.p2.reset_bet_size()

        # distribute new cards
        self.p1.set_card(Card.random_card())
        self.p2.set_card(Card.random_card())
        self.next_player = self.button  # next small blind
        self.button = self.other(self.button)  # move button

        # place blinds
        self.pot = self.button.take(PreFlopHighCardGame.BIG_BLIND)  # big blind
        self.pot += self.other(self.button).take(PreFlopHighCardGame.SMALL_BLIND)  # small blind

    def call(self, player=None):
        """
        :param player: The player who makes the call.
        """
        if player is None:
            player = self.next_player
        opponent_bet = self.other(player).bet_size()
        own_bet = player.bet_size()
        missing = opponent_bet - own_bet

        self.update_next_player()  # after a call, the player to act always switches

        if missing < 0:
            raise ValueError("Can not call an opponent who has placed a smaller bet.")

        # player calls all-in
        if missing >= player.stack():
            self.pot += player.take(player.stack())
            return self.show_down()

        # standard call
        self.pot += player.take(missing)

        if player.is_all_in():  # if a player calls all-in, a showdown always follows
            return self.show_down()

        # calling usually leads to a show down except the small blind called the big blind
        if player is not self.button and player.bet_size() == PreFlopHighCardGame.BIG_BLIND:
            # player is sb and called to bb
            return
        # otherwise (normal call, not pre from sb)
        self.show_down()

    def fold(self, player=None):
        """
        :param player: The player who folds.
        """
        if player is None:
            player = self.next_player
        self.payout(self.other(player))

    def bet(self, player=None, bet_size=0):
        """
        :param player: The player who bets / raises.
        :param bet_size: The amount by which the player raises.
                       If the raise amount is invalid, it will be rounded to a call or a raise.
        """
        if player is None:
            player = self.next_player
        if bet_size < 0:
            raise ValueError("Bet amount can not be negative");

        # betting is equivalent to calling if the opponent is all-in
        if self.other(player).is_all_in():
            return self.call(player)

        opponent_bet = self.other(player).bet_size()
        own_bet = player.bet_size()
        bet_difference = opponent_bet - own_bet

        # amount must be at least the bb and at least the bet difference
        bet_size = max(bet_size, PreFlopHighCardGame.BIG_BLIND, bet_difference)
        if player.bet_size() == 0:
            bet_size = max(bet_size, 2 * PreFlopHighCardGame.BIG_BLIND- PreFlopHighCardGame.SMALL_BLIND)

        # amount is at most an opponent all-in
        bet_size = min(bet_size, self.other(player).stack())

        amount = min(bet_difference + bet_size, player.stack())  # amount to put in (call and bet) is at most all-in

        if amount <= bet_difference:  # bet is actually a call all-in
            return self.show_down()

        self.pot += player.take(amount)
        self.update_next_player()

    def show_down(self):
        p1_card = self.p1.get_card()
        p2_card = self.p2.get_card()
        if p1_card.loses_against(p2_card):
            return self.payout(self.p2)
        if p2_card.loses_against(p1_card):
            return self.payout(self.p1)
        self.payout()  # split

    def payout(self, player=None):
        """
        Pays the pot to the passed player and starts the next round.
        :param player: The player who has won. If no player is passed, the pot will be split.
        """
        if player is None:
            # split
            p1bonus = 0
            p2bonus = 0
            if self.pot % 2 == 1:
                if not (self.p1.is_all_in() or self.p2.is_all_in()):
                    raise ValueError("Pot contains an uneven number of chips. This state is invalid.");
                else:
                    # one player is all in with less than 1 bb
                    # pot cant be divided by two
                    if self.p1.is_all_in():
                        p2bonus = 1
                    else:
                        p1bonus = 1
            self.p1.ship_over(int(self.pot / 2) + p1bonus)
            self.p2.ship_over(int(self.pot / 2) + p2bonus)
            self.last_gain = [0, 0]
        else:
            player.ship_over(self.pot)

            # set last gain array
            # each player has gained the amount that they win (pot or 0) minus their contribution to the pot.
            if player is self.p1:
                self.last_gain = [self.pot - self.p1.bet_size(), - self.p2.bet_size()]
            else:
                self.last_gain = [- self.p1.bet_size(), self.pot - self.p2.bet_size()]

        self.next_round()

    def other(self, player):
        """
        :param player: One of the two players in the game.
        :return: The player who was not the argument, i.e. the other player.
        """
        if player == self.p1:
            return self.p2
        return self.p1

    def update_next_player(self):
        self.next_player = self.other(self.next_player)

    def _equal_stacks(self):
        """
        :return: True if both players have the same stack size.
        """
        return self.p1.stack() == self.p2.stack()

    def __str__(self):
        return "Round #" + str(self.round_count) + "\n" \
               "=======================\n" + \
               "Player " + self.p1.name + " | Player " + self.p2.name + "\n" + \
               "Stack: " + str(self.p1.stack()) + " | " + str(self.p2.stack()) + "\n" + \
               "Cards: " + str(self.p1.get_card()) + " | " + str(self.p2.get_card()) + " \n" + \
               "Bet sizes: " + str(self.p1.bet_size()) + " | " + str(self.p2.bet_size()) + "\n\n"
