import random
import model
import tensorflow as tf
from collections import deque


class Strategy:
    action_keys = ['f', 'c', 'b']

    def action(self, state):
        return None


class StrategyRandom(Strategy):
    def action(self, state):
        return Strategy.action_keys[random.randint(0, 2)]


class StrategyHardcodedSelfish(Strategy):
    """
    Strategy that takes actions solely based on the own holding.
    """
    def action(self, state):
        """
        :param state: A 5-tuple containing (card, stack, opponent stack, bet size, opponent bet size)
        :return: The action to take.
        """
        card = state[0]
        if card == 12:  # ace
            return 2  # raise
        if card == 0:  # two
            return 0  # fold
        if random.randint(0, 12) < card:  # the higher the holding, the more likely is a raise
            return 2  # raise
        return 1  # call


class StrategyNetwork(Strategy):
    def __init__(self):
        # model init
        self.state_batch = tf.placeholder(tf.float32, shape=[None, model.INPUT_SIZE], name='state_batch')
        self.action_values = model.q_fn(self.state_batch)
        self.chosen_action = tf.argmax(self.action_values, axis=1, name='chosen_action')
        self.model = model
        self.sess = tf.Session()
        self.sess.run(tf.local_variables_initializer())
        self.sess.run(tf.global_variables_initializer())

        self.memory = deque(maxlen=100000)

    def action(self, state):
        action = self.sess.run(self.chosen_action, feed_dict={
            self.state_batch: [state]
        })
        return Strategy.action_keys[action[0]]  # unpack batch and convert to char

    def add_memories(self, new):
        self.memory.extend(new)


class HumanPlayer(Strategy):
    def action(self, state):
        print(state)
        invalid = True
        while invalid:
            action = input("Action {f,c,b}: ")
            if action == 'f' or action == 'c' or action == 'b':
                return action
