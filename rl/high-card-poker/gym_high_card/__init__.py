from gym.envs.registration import register

register(
    id='high-card-v0',
    entry_point='gym_high_card.envs:HighCardEnv',
)
