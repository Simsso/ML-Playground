from collections import deque
import pfhc_env as pfhc
from player import Player
from strategy import *


STEPS = int(1e4)
INIT_STACK = 50

memory = deque()
p1 = Player("A", INIT_STACK, strategy=StrategyRandom())
p2 = Player("B", INIT_STACK, strategy=StrategyNetwork())
env = pfhc.HighCardEnv(p1, p2)


def main(args=None):
    for step in range(STEPS):
        env.reset()
        over = False

        history = []  # holds 6-tuples (player, state, action, next_state, reward, over)

        hand = 0
        while not over:
            """
            Two players are involved, namely pa and pb. pa starts in state1, takes action1 and transitions to state2.
            State2 is not relevant for pa though, because they can not affect the transition from state2 to state3.
            For pa, the transition from state1 to state3 is relevant.  
            """
            hand += 1

            pa = env.game.next_player  # either p1 or p2
            state1 = env.get_state(pa)
            action1 = (pa.strategy.action(state1), 0)
            state2, reward1, over, _ = env.step(action1)
            history.append((pa, state1, action1, state2, reward1, over))
            if not over:
                pb = env.game.next_player  # the opposite of pa
                action2 = (pb.strategy.action(state2), 0)
                state3, reward2, over, _ = env.step(action2)
                history.append((pb, state2, action2, state3, reward2, over))

        # one game played (i.e. one player has no chips left)
        winner = p1 if p1.stack() > p2.stack() else p2
        print(winner.name + " won after " + str(hand) + " hands")


def get_memory_from_mdp(mdp, player):
    """
    :param mdp: Markov decision process samples. 6-tuples (player, state, action, next_state, reward, over)
    :param player: The player to get memory samples for.
    :return:
    """

if __name__ == '__main__':
    tf.app.run()
