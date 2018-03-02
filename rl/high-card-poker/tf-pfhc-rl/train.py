def get_memory_from_mdp(mdp, player):
    """
    Converts the raw history into (state, action, reward, next state) tuples.
    If the next state did not exist because the game ended, it is None.
    In any case, the next state is the next state that the given player observed. Not the one the opponent saw.
    :param mdp: Markov decision process samples. 6-tuples (player, state, action, next_state, reward, over)
    :param player: The player to get memory samples for.
    :return: 4-tuples of shape (state, action, reward, next state)
    """

    return []

