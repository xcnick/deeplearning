import os
import gym
import time
import torch
import numpy as np
import numpy.random as rd
import cv2

from copy import deepcopy
from .buffer import ReplayBuffer


def explore_before_training(env, buffer, target_step, reward_scale, gamma) -> int:
    # just for off-policy. Because on-policy don't explore before training.
    if_discrete = env.if_discrete
    action_dim = env.action_dim

    state = env.reset()
    steps = 0

    while steps < target_step:
        action = rd.randint(action_dim) if if_discrete else rd.uniform(-1, 1, size=action_dim)
        next_state, reward, done, _ = env.step(action)
        steps += 1

        scaled_reward = reward * reward_scale
        mask = 0.0 if done else gamma
        other = (scaled_reward, mask, action) if if_discrete else (scaled_reward, mask, *action)
        buffer.append_buffer(state, other)

        state = env.reset() if done else next_state
    return steps


def atari_explore_before_training(env, buffer, target_step, reward_scale, gamma) -> int:
    def preprocess(observation):
        img = np.reshape(observation, [210, 160, 3]).astype(np.float32)
        # RGB转换成灰度图像的一个常用公式是：ray = R*0.299 + G*0.587 + B*0.114
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114  # shape (210,160)
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)  # shape(110,84)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        x_t.astype((np.uint8))
        x_t = np.moveaxis(x_t, 2, 0)  # shape（1，84，84）
        return np.array(x_t).astype(np.float32) / 255.0

    # just for off-policy. Because on-policy don't explore before training.
    if_discrete = env.if_discrete
    action_dim = env.action_dim

    state = env.reset()
    state = preprocess(state)
    state = np.reshape(state, (84, 84))
    state_shadow = np.stack((state, state, state, state), axis=0)
    steps = 0

    while steps < target_step:
        action = rd.randint(action_dim) if if_discrete else rd.uniform(-1, 1, size=action_dim)
        next_state, reward, done, _ = env.step(action)
        next_state = preprocess(next_state)
        next_state_shadow = np.append(next_state, state_shadow[:3, :, :], axis=0)
        steps += 1

        scaled_reward = reward * reward_scale
        mask = 0.0 if done else gamma
        other = (scaled_reward, mask, action) if if_discrete else (scaled_reward, mask, *action)
        buffer.append_buffer(next_state_shadow, other)

        state_shadow = env.reset() if done else next_state_shadow
    return steps


def train_and_evaluate(cfg_args, agent, env):
    if "train_url" not in cfg_args:
        agent_name = agent.__class__.__name__
        cwd = f"./{agent_name}/{env.unwrapped.spec.id}"
    else:
        cwd = cfg_args["train_url"]

    print(f"| cwd: {cwd}")
    os.makedirs(cwd, exist_ok=True)
    torch.set_num_threads(8)
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(0)
    np.random.seed(0)

    """basic arguments"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """training arguments"""
    # net_dim = args.net_dim
    max_memo = cfg_args["max_memo"]
    break_step = cfg_args["break_step"]
    batch_size = cfg_args["batch_size"]
    target_step = cfg_args["target_step"]
    repeat_times = cfg_args["repeat_times"]
    if_break_early = cfg_args["if_allow_break"]
    gamma = cfg_args["gamma"]
    reward_scale = cfg_args["reward_scale"]

    """evaluating arguments"""
    show_gap = cfg_args["show_gap"]
    eval_times1 = cfg_args["eval_times1"]
    eval_times2 = cfg_args["eval_times2"]
    env_eval = deepcopy(env)  # if env_eval is None else deepcopy(env_eval)

    """init: environment"""
    max_step = env.max_step
    state_dim = env.state_dim
    action_dim = env.action_dim
    if_discrete = env.if_discrete
    env_eval = deepcopy(env)  # if env_eval is None else deepcopy(env_eval)

    """init: Agent, ReplayBuffer, Evaluator"""
    if_on_policy = getattr(agent, "if_on_policy", False)

    buffer = ReplayBuffer(
        max_len=max_memo + max_step,
        state_dim=state_dim,
        action_dim=1 if if_discrete else action_dim,
        if_on_policy=if_on_policy,
        device=device,
    )

    evaluator = Evaluator(
        cwd=cwd,
        device=device,
        env=env_eval,
        eval_times1=eval_times1,
        eval_times2=eval_times2,
        show_gap=show_gap,
    )  # build Evaluator

    """prepare for training"""
    agent.state = env.reset()
    if if_on_policy:
        steps = 0
    else:  # explore_before_training for off-policy
        with torch.no_grad():  # update replay buffer
            steps = explore_before_training(env, buffer, target_step, reward_scale, gamma)

        agent.update_net(
            buffer, target_step, batch_size, repeat_times
        )  # pre-training and hard update
        agent.act_target.load_state_dict(agent.act.state_dict()) if getattr(
            agent, "act_target", None
        ) else None
        agent.cri_target.load_state_dict(agent.cri.state_dict()) if getattr(
            agent, "cri_target", None
        ) else None
    total_step = steps

    """start training"""
    if_reach_goal = False
    while not (
        (if_break_early and if_reach_goal)
        or total_step > break_step
        or os.path.exists(f"{cwd}/stop")
    ):
        with torch.no_grad():  # speed up running
            steps = agent.explore_env(env, buffer, target_step, reward_scale, gamma)

        total_step += steps

        obj_a, obj_c = agent.update_net(buffer, target_step, batch_size, repeat_times)

        with torch.no_grad():  # speed up running
            if_reach_goal = evaluator.evaluate_save(agent.actor, steps, obj_a, obj_c)


class Evaluator:
    def __init__(self, cwd, eval_times1, eval_times2, show_gap, env, device):
        self.recorder = [
            (0.0, -np.inf, 0.0, 0.0, 0.0),
        ]  # total_step, r_avg, r_std, obj_a, obj_c
        self.r_max = -np.inf
        self.total_step = 0

        self.cwd = cwd  # constant
        self.device = device
        self.show_gap = show_gap
        self.eva_times1 = eval_times1
        self.eva_times2 = eval_times2
        self.env = env
        self.target_reward = env.target_reward

        self.used_time = None
        self.start_time = time.time()
        self.print_time = time.time()
        print(f"{'Step':>8}  {'MaxR':>8} |{'avgR':>8}  {'stdR':>8}   {'objA':>8}  {'objC':>8}")

    def evaluate_save(self, act, steps, obj_a, obj_c) -> bool:
        reward_list = [
            get_episode_return(self.env, act, self.device) for _ in range(self.eva_times1)
        ]
        r_avg = np.average(reward_list)  # episode return average
        r_std = float(np.std(reward_list))  # episode return std

        if r_avg > self.r_max:  # evaluate actor twice to save CPU Usage and keep precision
            reward_list += [
                get_episode_return(self.env, act, self.device)
                for _ in range(self.eva_times2 - self.eva_times1)
            ]
            r_avg = np.average(reward_list)  # episode return average
            r_std = float(np.std(reward_list))  # episode return std
        if r_avg > self.r_max:  # save checkpoint with highest episode return
            self.r_max = r_avg  # update max reward (episode return)

            """save actor.pth"""
            act_save_path = f"{self.cwd}/actor.pth"
            torch.save(act.state_dict(), act_save_path)
            print(f"{self.total_step:8.2e}  {self.r_max:8.2f} |")

        self.total_step += steps  # update total training steps
        self.recorder.append((self.total_step, r_avg, r_std, obj_a, obj_c))  # update recorder

        if_reach_goal = bool(self.r_max > self.target_reward)  # check if_reach_goal
        if if_reach_goal and self.used_time is None:
            self.used_time = int(time.time() - self.start_time)
            print(
                f"{'Step':>8}  {'TargetR':>8} |"
                f"{'avgR':>8}  {'stdR':>8}   {'UsedTime':>8}  ########\n"
                f"{self.total_step:8.2e}  {self.target_reward:8.2f} |"
                f"{r_avg:8.2f}  {r_std:8.2f}   {self.used_time:>8}  ########"
            )

        if time.time() - self.print_time > self.show_gap:
            self.print_time = time.time()
            print(
                f"{self.total_step:8.2e}  {self.r_max:8.2f} |"
                f"{r_avg:8.2f}  {r_std:8.2f}   {obj_a:8.2f}  {obj_c:8.2f}"
            )
        return if_reach_goal

    def draw_plot(self):
        if len(self.recorder) == 0:
            print("| save_npy_draw_plot() WARNNING: len(self.recorder)==0")
            return None

        """convert to array and save as npy"""
        np.save("%s/recorder.npy" % self.cwd, self.recorder)

        """draw plot and save as png"""
        train_time = int(time.time() - self.start_time)
        total_step = int(self.recorder[-1][0])
        save_title = f"plot_step_time_maxR_{int(total_step)}_{int(train_time)}_{self.r_max:.3f}"

        save_learning_curve(self.recorder, self.cwd, save_title)


def get_episode_return(env, act, device) -> float:
    episode_return = 0.0  # sum of rewards in an episode
    max_step = env.max_step
    if_discrete = env.if_discrete

    state = env.reset()
    for _ in range(max_step):
        s_tensor = torch.as_tensor((state,), device=device)
        a_tensor = act(s_tensor)
        if if_discrete:
            a_tensor = a_tensor.argmax(dim=1)
        # not need detach(), because with torch.no_grad() outside
        action = a_tensor.cpu().numpy()[0]
        state, reward, done, _ = env.step(action)
        episode_return += reward
        if done:
            break
    return env.episode_return if hasattr(env, "episode_return") else episode_return


def get_video_to_watch_gym_render(agent, env):
    from gym import wrappers
    from pyvirtualdisplay import Display

    display = Display(visible=0, size=(400, 300))
    display.start()

    agent_name = agent.__class__.__name__
    cwd = f"./{agent_name}/{env.unwrapped.spec.id}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if agent is not None:
        agent.save_load_model(cwd=cwd, if_save=False)
        rd.seed(194686)
        torch.manual_seed(1942876)

    """initialize evaluete and env.render()"""
    save_frame_dir = os.path.join(cwd, "video")
    env = wrappers.Monitor(env, save_frame_dir, force=True)

    if not os.path.exists(save_frame_dir):
        os.makedirs(save_frame_dir, exist_ok=True)

    state = env.reset()
    episode_return = 0
    step = 0
    episodes = 10
    episode = 0
    with torch.no_grad():
        while episode < episodes:
            if agent is not None:
                s_tensor = torch.as_tensor((state,), dtype=torch.float32, device=device)
                a_tensor = agent.actor(s_tensor)
                if env.if_discrete:
                    a_tensor = a_tensor.argmax(dim=1)
                action = a_tensor.cpu().numpy()[0]
            else:
                action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)

            episode_return += reward
            step += 1

            if done:
                episode += 1
                print(f"{episode:>6}, {step:6.0f}, {episode_return:8.3f}, {reward:8.3f}")
                state = env.reset()
                episode_return = 0
                step = 0
            else:
                state = next_state
        env.close()
