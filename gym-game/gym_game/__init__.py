from gym.envs.registration import register

register(
    id='game-v0',
    entry_point='gym_game.envs:GameEnv',
)