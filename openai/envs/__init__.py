from gym.envs.registration import registry, register, make, spec

# Classic
# ----------------------------------------
register(
    id='10ArmedBanditStationary-v0',
    entry_point='openai.envs.classic.narmedbandit:NArmedBanditEnv',
    kwargs={'arms' : 10, 'type' : 'stationary'},
    timestep_limit=2000,
)
