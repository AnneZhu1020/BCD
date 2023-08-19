from minirl.envs.wrapper import vectorize_gym


def atari_vec_env(num_envs, env_name, use_subproc=True):
    from minirl.envs.atari.atari_wrappers import make_atari_deepmind

    env = vectorize_gym(
        num=num_envs,
        env_fn=make_atari_deepmind,
        env_kwargs={"env_id": env_name},
        use_subproc=use_subproc,
    )
    return env


def procgen_vec_env(num_envs, env_name, **kwargs):
    from procgen.env import ProcgenGym3Env, ENV_NAMES

    if env_name not in ENV_NAMES:
        # default error message in procgen is unfriendly
        raise ValueError(f"No environment named '{env_name}'. Choose from {ENV_NAMES}.")
    else:
        return ProcgenGym3Env(num=num_envs, env_name=env_name, **kwargs)
