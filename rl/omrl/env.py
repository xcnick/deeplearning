import os
import numpy as np
import numpy.random as rd
import gym

from .builder import ENVS


@ENVS.register_module()
class BaseEnv(gym.Wrapper):
    def __init__(self, env_name, data_type=np.float32):
        super().__init__(gym.make(env_name))
        self.data_type = data_type
        self.reset = self.reset_type
        self.step = self.step_type

    def reset_type(self) -> np.ndarray:
        """ state = env.reset()

        convert the data type of state from float64 to float32

        :return array state: state.shape==(state_dim, )
        """
        state = self.env.reset()
        return state.astype(self.data_type)

    def step_type(self, action) -> (np.ndarray, float, bool, dict):
        """ next_state, reward, done = env.step(action)

        convert the data type of state from float64 to float32,
        adjust action range to (-action_max, +action_max)

        :return array state:  state.shape==(state_dim, )
        :return float reward: reward of one step
        :return bool  done  : the terminal of an training episode
        :return dict  info  : the information save in a dict. OpenAI gym standard. Send a `None` is OK
        """
        state, reward, done, info = self.env.step(action * self.action_max)
        return state.astype(self.data_type), reward, done, info

    def get_gym_env_info(self, if_print=True) -> (str, int, int, int, int, bool, float):
        """get information of a standard OpenAI gym env.

        The DRL algorithm AgentXXX need these env information for building networks and training.
        env_name: the environment name, such as XxxXxx-v0
        state_dim: the dimension of state
        action_dim: the dimension of continuous action; Or the number of discrete action
        action_max: the max action of continuous action; action_max == 1 when it is discrete action space
        if_discrete: Is this env a discrete action space?
        target_reward: the target episode return, if agent reach this score, then it pass this game (env).
        max_step: the steps in an episode. (from env.reset to done). It breaks an episode when it reach max_step

        :env: a standard OpenAI gym environment, it has env.reset() and env.step()
        :bool if_print: print the information of environment. Such as env_name, state_dim ...
        """
        gym.logger.set_level(
            40
        )  # Block warning: 'WARN: Box bound precision lowered by casting to float32'
        assert isinstance(self.env, gym.Env)

        env_name = self.env.unwrapped.spec.id

        state_shape = self.env.observation_space.shape
        state_dim = (
            state_shape[0] if len(state_shape) == 1 else state_shape
        )  # sometimes state_dim is a list

        target_reward = getattr(self.env, "target_reward", None)
        target_reward_default = getattr(self.env.spec, "reward_threshold", None)
        if target_reward is None:
            target_reward = target_reward_default
        if target_reward is None:
            target_reward = 65536

        max_step = getattr(self.env, "max_step", None)
        max_step_default = getattr(self.env, "_max_episode_steps", None)
        if max_step is None:
            max_step = max_step_default
        if max_step is None:
            max_step = 1024

        if_discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        if if_discrete:  # make sure it is discrete action space
            action_dim = self.env.action_space.n
            action_max = int(1)
        elif isinstance(
            self.env.action_space, gym.spaces.Box
        ):  # make sure it is continuous action space
            action_dim = self.env.action_space.shape[0]
            action_max = float(self.env.action_space.high[0])
        else:
            raise RuntimeError(
                "| Please set these value manually: if_discrete=bool, action_dim=int, action_max=1.0"
            )

        print(
            f"\n| env_name:  {env_name}, action space if_discrete: {if_discrete}"
            f"\n| state_dim: {state_dim:4}, action_dim: {action_dim}, action_max: {action_max}"
            f"\n| max_step:  {max_step:4}, target_reward: {target_reward}"
        ) if if_print else None
        return env_name, state_dim, action_dim, action_max, max_step, if_discrete, target_reward


@ENVS.register_module()
class OrdinaryEnv(BaseEnv):
    def __init__(self, env_name):
        super().__init__(env_name)
        (
            self.env_name,
            self.state_dim,
            self.action_dim,
            self.action_max,
            self.max_step,
            self.if_discrete,
            self.target_reward,
        ) = self.get_gym_env_info()


@ENVS.register_module()
class AtariEnv(BaseEnv):
    def __init__(self, env_name, screen_size, frame_skip):
        super().__init__(env_name)
        self.env = gym.wrappers.AtariPreprocessing(
            self.env, screen_size=84, grayscale_obs=True, frame_skip=4, noop_max=0,
        )
        self.env = gym.wrappers.FrameStack(self.env, 4)
        self.env = gym.wrappers.FlattenObservation(self.env)

        (
            self.env_name,
            self.state_dim,
            self.action_dim,
            self.action_max,
            self.max_step,
            self.if_discrete,
            self.target_reward,
        ) = self.get_gym_env_info()


def get_avg_std__for_state_norm(env_name) -> (np.ndarray, np.ndarray):
    """return the state normalization data: neg_avg and div_std

    ReplayBuffer.print_state_norm() will print `neg_avg` and `div_std`
    You can save these array to here. And PreprocessEnv will load them automatically.
    eg. `state = (state + self.neg_state_avg) * self.div_state_std` in `PreprocessEnv.step_norm()`
    neg_avg = -states.mean()
    div_std = 1/(states.std()+1e-5) or 6/(states.max()-states.min())


    :str env_name: the name of environment that helps to find neg_avg and div_std
    :return array avg: neg_avg.shape=(state_dim)
    :return array std: div_std.shape=(state_dim)
    """
    avg = None
    std = None
    if env_name == "LunarLanderContinuous-v2":
        avg = np.array(
            [
                1.65470898e-02,
                -1.29684399e-01,
                4.26883133e-03,
                -3.42124557e-02,
                -7.39076972e-03,
                -7.67103031e-04,
                1.12640885e00,
                1.12409466e00,
            ]
        )
        std = np.array(
            [
                0.15094465,
                0.29366297,
                0.23490797,
                0.25931464,
                0.21603736,
                0.25886878,
                0.277233,
                0.27771219,
            ]
        )
    elif env_name == "BipedalWalker-v3":
        avg = np.array(
            [
                1.42211734e-01,
                -2.74547996e-03,
                1.65104509e-01,
                -1.33418152e-02,
                -2.43243194e-01,
                -1.73886203e-02,
                4.24114229e-02,
                -6.57800099e-02,
                4.53460692e-01,
                6.08022244e-01,
                -8.64884810e-04,
                -2.08789053e-01,
                -2.92092949e-02,
                5.04791247e-01,
                3.33571745e-01,
                3.37325723e-01,
                3.49106580e-01,
                3.70363115e-01,
                4.04074671e-01,
                4.55838055e-01,
                5.36685407e-01,
                6.70771701e-01,
                8.80356865e-01,
                9.97987386e-01,
            ]
        )
        std = np.array(
            [
                0.84419678,
                0.06317835,
                0.16532085,
                0.09356959,
                0.486594,
                0.55477525,
                0.44076614,
                0.85030824,
                0.29159821,
                0.48093035,
                0.50323634,
                0.48110776,
                0.69684234,
                0.29161077,
                0.06962932,
                0.0705558,
                0.07322677,
                0.07793258,
                0.08624322,
                0.09846895,
                0.11752805,
                0.14116005,
                0.13839757,
                0.07760469,
            ]
        )
    elif env_name == "ReacherBulletEnv-v0":
        avg = np.array(
            [
                0.03149641,
                0.0485873,
                -0.04949671,
                -0.06938662,
                -0.14157104,
                0.02433294,
                -0.09097818,
                0.4405931,
                0.10299437,
            ],
            dtype=np.float32,
        )
        std = np.array(
            [
                0.12277275,
                0.1347579,
                0.14567468,
                0.14747661,
                0.51311225,
                0.5199606,
                0.2710207,
                0.48395795,
                0.40876198,
            ],
            dtype=np.float32,
        )
    elif env_name == "AntBulletEnv-v0":
        avg = np.array(
            [
                -2.2785307e-01,
                -4.1971792e-02,
                9.2752278e-01,
                8.3731368e-02,
                1.2131270e-03,
                -5.7878396e-03,
                1.8127944e-02,
                -1.1823924e-02,
                1.5717462e-01,
                1.2224792e-03,
                -1.9672018e-01,
                6.4919023e-03,
                -2.0346987e-01,
                5.1609759e-04,
                1.6572942e-01,
                -6.0344036e-03,
                -1.6024958e-02,
                -1.3426526e-03,
                3.8138664e-01,
                -5.6816568e-03,
                -1.8004493e-01,
                -3.2685725e-03,
                -1.5989083e-01,
                7.0396746e-03,
                7.2912598e-01,
                8.3666992e-01,
                8.2824707e-01,
                7.6196289e-01,
            ],
            dtype=np.float32,
        )
        std = np.array(
            [
                0.09652393,
                0.33918667,
                0.23290202,
                0.13423778,
                0.10426794,
                0.11678293,
                0.39058578,
                0.28871638,
                0.5447721,
                0.36814892,
                0.73530555,
                0.29377502,
                0.5031936,
                0.36130348,
                0.71889997,
                0.2496559,
                0.5484764,
                0.39613277,
                0.7103549,
                0.25976712,
                0.56372136,
                0.36917716,
                0.7030704,
                0.26312646,
                0.30555955,
                0.2681793,
                0.27192947,
                0.29626447,
            ],
            dtype=np.float32,
        )
    #     avg = np.array([
    #         0.4838, -0.047, 0.3500, 1.3028, -0.249, 0.0000, -0.281, 0.0573,
    #         -0.261, 0.0000, 0.0424, 0.0000, 0.2278, 0.0000, -0.072, 0.0000,
    #         0.0000, 0.0000, -0.175, 0.0000, -0.319, 0.0000, 0.1387, 0.0000,
    #         0.1949, 0.0000, -0.136, -0.060])
    #     std = np.array([
    #         0.0601, 0.2267, 0.0838, 0.2680, 0.1161, 0.0757, 0.1495, 0.1235,
    #         0.6733, 0.4326, 0.6723, 0.3422, 0.7444, 0.5129, 0.6561, 0.2732,
    #         0.6805, 0.4793, 0.5637, 0.2586, 0.5928, 0.3876, 0.6005, 0.2369,
    #         0.4858, 0.4227, 0.4428, 0.4831])
    # elif env_name == 'MinitaurBulletEnv-v0': # need check
    #     # avg = np.array([0.90172989, 1.54730119, 1.24560906, 1.97365306, 1.9413892,
    #     #                 1.03866835, 1.69646277, 1.18655352, -0.45842347, 0.17845232,
    #     #                 0.38784456, 0.58572877, 0.91414561, -0.45410697, 0.7591031,
    #     #                 -0.07008998, 3.43842258, 0.61032482, 0.86689961, -0.33910894,
    #     #                 0.47030415, 4.5623528, -2.39108079, 3.03559422, -0.36328256,
    #     #                 -0.20753499, -0.47758384, 0.86756409])
    #     # std = np.array([0.34192648, 0.51169916, 0.39370621, 0.55568461, 0.46910769,
    #     #                 0.28387504, 0.51807949, 0.37723445, 13.16686185, 17.51240024,
    #     #                 14.80264211, 16.60461412, 15.72930229, 11.38926597, 15.40598346,
    #     #                 13.03124941, 2.47718145, 2.55088804, 2.35964651, 2.51025567,
    #     #                 2.66379017, 2.37224904, 2.55892521, 2.41716885, 0.07529733,
    #     #                 0.05903034, 0.1314812, 0.0221248])
    # elif env_name == "BipedalWalkerHardcore-v3": # need check
    #     avg = np.array([-3.6378160e-02, -2.5788052e-03, 3.4413573e-01, -8.4189959e-03,
    #                     -9.1864385e-02, 3.2804706e-04, -6.4693891e-02, -9.8939031e-02,
    #                     3.5180664e-01, 6.8103075e-01, 2.2930240e-03, -4.5893672e-01,
    #                     -7.6047562e-02, 4.6414185e-01, 3.9363885e-01, 3.9603019e-01,
    #                     4.0758255e-01, 4.3053803e-01, 4.6186063e-01, 5.0293463e-01,
    #                     5.7822973e-01, 6.9820738e-01, 8.9829963e-01, 9.8080903e-01])
    #     std = np.array([0.5771428, 0.05302362, 0.18906464, 0.10137994, 0.41284004,
    #                     0.68852615, 0.43710527, 0.87153363, 0.3210142, 0.36864948,
    #                     0.6926624, 0.38297284, 0.76805115, 0.33138904, 0.09618598,
    #                     0.09843876, 0.10035378, 0.11045089, 0.11910835, 0.13400233,
    #                     0.15718603, 0.17106676, 0.14363566, 0.10100251])
    return avg, std
