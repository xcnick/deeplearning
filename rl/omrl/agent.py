import os
from copy import deepcopy  # deepcopy target_network

import torch
import numpy as np
import numpy.random as rd


from .builder import AGENTS, build_net, build_optimizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@AGENTS.register_module()
class AgentBase:
    def __init__(
        self,
        actor=None,
        actor_target=None,
        critic=None,
        critic_target=None,
        optimizer=None,
        **kwargs
    ):
        self.soft_update_tau = kwargs.get("soft_update_tau", 5e-3)
        self.state = None  # set for self.update_buffer(), initialize before training

        self.actor = actor
        self.critic = critic
        self.actor_target = actor_target
        self.critic_target = critic_target
        self.actor_optimizer = optimizer
        self.critic_optimizer = optimizer
        self.criterion = None

    def select_action(self, state) -> np.ndarray:
        """
        :array state: state.shape==(state_dim, )
        :return array action: action.shape==(action_dim, ), (action.min(), action.max())==(-1, +1)
        """
        states = torch.as_tensor((state,), dtype=torch.float32, device=device).detach_()
        action = self.act(states)[0]
        return action.cpu().numpy()

    def select_actions(self, states) -> np.ndarray:
        """
        :array states: (state, ) or (state, state, ...) or state.shape==(n, *state_dim)
        :return array action: action.shape==(-1, action_dim), (action.min(), action.max())==(-1, +1)
        """
        states = torch.as_tensor(states, dtype=torch.float32, device=device).detach_()
        actions = self.act(states)
        return actions.cpu().numpy()  # -1 < action < +1

    def explore_env(self, env, buffer, target_step, reward_scale, gamma) -> int:
        """
        :env: RL training environment. env.reset() env.step()
        :buffer: Experience Replay Buffer. buffer.append_buffer() buffer.extend_buffer()
        :int target_step: explored target_step number of step in env
        :float reward_scale: scale reward, 'reward * reward_scale'
        :float gamma: discount factor, 'mask = 0.0 if done else gamma'
        :return int target_step: collected target_step number of step in env
        """
        for _ in range(target_step):
            action = self.select_action(self.state)
            next_s, reward, done, _ = env.step(action)
            other = (reward * reward_scale, 0.0 if done else gamma, *action)
            buffer.append_buffer(self.state, other)
            self.state = env.reset() if done else next_s
        return target_step

    def update_net(self, buffer, target_step, batch_size, repeat_times) -> (float, float):
        """
        :buffer: Experience replay buffer. buffer.append_buffer() buffer.extend_buffer()
        :int target_step: explore target_step number of step in env
        :int batch_size: sample batch_size of data for Stochastic Gradient Descent
        :float repeat_times: the times of sample batch = int(target_step * repeat_times) in off-policy
        :return float obj_a: the objective value of actor
        :return float obj_c: the objective value of critic
        """

    def save_load_model(self, cwd, if_save):
        """
        :str cwd: current working directory, we save model file here
        :bool if_save: save model or load model
        """
        act_save_path = "{}/actor.pth".format(cwd)
        cri_save_path = "{}/critic.pth".format(cwd)

        def load_torch_file(network, save_path):
            network_dict = torch.load(save_path, map_location=lambda storage, loc: storage)
            network.load_state_dict(network_dict)

        if if_save:
            if self.act is not None:
                torch.save(self.act.state_dict(), act_save_path)
            if self.cri is not None:
                torch.save(self.cri.state_dict(), cri_save_path)
        elif (self.act is not None) and os.path.exists(act_save_path):
            load_torch_file(self.act, act_save_path)
            print("Loaded act:", cwd)
        elif (self.cri is not None) and os.path.exists(cri_save_path):
            load_torch_file(self.cri, cri_save_path)
            print("Loaded cri:", cwd)
        else:
            print("FileNotFound when load_model: {}".format(cwd))

    @staticmethod
    def soft_update(target_net, current_net, tau):
        """
        :nn.Module target_net: target network update via a current network, it is more stable
        :nn.Module current_net: current network update via an optimizer
        """
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data.__mul__(tau) + tar.data.__mul__(1 - tau))


@AGENTS.register_module()
class AgentDQN(AgentBase):
    def __init__(
        self,
        actor=None,
        actor_target=None,
        critic=None,
        critic_target=None,
        optimizer=None,
        **kwargs
    ):
        super().__init__(actor, actor_target, critic, critic_target, **kwargs)
        self.explore_rate = kwargs.get("explore_rate", 0.1)
        self.critic = build_net(critic, default_args=kwargs).to(device)
        self.critic_target = deepcopy(self.critic)
        self.actor = self.critic  # to keep the same from Actor-Critic framework
        self.action_dim = self.critic.action_dim

        self.criterion = torch.torch.nn.MSELoss()
        optimizer["params"] = self.critic.parameters()
        self.critic_optimizer = build_optimizer(optimizer)

    def select_action(self, state) -> int:  # for discrete action space
        if rd.rand() < self.explore_rate:  # epsilon-greedy
            a_int = rd.randint(self.action_dim)  # choosing action randomly
        else:
            states = torch.as_tensor((state,), dtype=torch.float32, device=device).detach_()
            action = self.actor(states)[0]
            a_int = action.argmax(dim=0).cpu().numpy()
        return a_int

    def explore_env(self, env, buffer, target_step, reward_scale, gamma) -> int:
        for _ in range(target_step):
            action = self.select_action(self.state)
            next_s, reward, done, _ = env.step(action)

            other = (reward * reward_scale, 0.0 if done else gamma, action)  # action is an int
            buffer.append_buffer(self.state, other)
            self.state = env.reset() if done else next_s
        return target_step

    def update_net(self, buffer, target_step, batch_size, repeat_times) -> (float, float):
        buffer.update_now_len_before_sample()

        next_q = obj_critic = None
        for _ in range(int(target_step * repeat_times)):
            with torch.no_grad():
                reward, mask, action, state, next_s = buffer.sample_batch(batch_size)  # next_state
                next_q = self.critic_target(next_s).max(dim=1, keepdim=True)[0]
                q_label = reward + mask * next_q
            q_eval = self.critic(state).gather(1, action.type(torch.long))
            obj_critic = self.criterion(q_eval, q_label)

            self.critic_optimizer.zero_grad()
            obj_critic.backward()
            self.critic_optimizer.step()
            self.soft_update(self.critic_target, self.critic, self.soft_update_tau)
        return next_q.mean().item(), obj_critic.item()


@AGENTS.register_module()
class AgentDuelingDQN(AgentDQN):
    def __init__(
        self,
        actor=None,
        actor_target=None,
        critic=None,
        critic_target=None,
        optimizer=None,
        **kwargs
    ):
        super().__init__(actor, actor_target, critic, critic_target, **kwargs)
        """
        Advantage function --> Dueling Q value = val_q + adv_q - adv_q.mean()
        """


@AGENTS.register_module()
class AgentDDPG(AgentBase):
    def __init__(
        self,
        actor=None,
        actor_target=None,
        critic=None,
        critic_target=None,
        optimizer=None,
        **kwargs
    ):
        super().__init__(actor, actor_target, critic, critic_target, **kwargs)
        self.explore_noise = kwargs.get("explore_noise", 0.1)

        self.actor = build_net(actor, default_args=kwargs).to(device)
        self.actor_target = deepcopy(self.actor)
        self.critic = build_net(critic, default_args=kwargs).to(device)
        self.critic_target = deepcopy(self.critic)

        self.criterion = torch.nn.MSELoss()
        optimizer["params"] = self.actor.parameters()
        self.actor_optimizer = build_optimizer(optimizer)
        optimizer["params"] = self.critic.parameters()
        self.critic_optimizer = build_optimizer(optimizer)

    def select_action(self, state) -> np.ndarray:
        states = torch.as_tensor((state,), dtype=torch.float32, device=device).detach_()
        action = self.actor(states)[0]
        action = (action + torch.randn_like(action) * self.explore_noise).clamp(-1, 1)
        return action.cpu().numpy()

    def update_net(self, buffer, target_step, batch_size, repeat_times) -> (float, float):
        buffer.update_now_len_before_sample()

        obj_critic = obj_actor = None  # just for print return
        for _ in range(int(target_step * repeat_times)):
            with torch.no_grad():
                reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
                next_q = self.critic_target(next_s, self.actor_target(next_s))
                q_label = reward + mask * next_q
            q_value = self.critic(state, action)
            obj_critic = self.criterion(q_value, q_label)

            self.critic_optimizer.zero_grad()
            obj_critic.backward()
            self.critic_optimizer.step()
            self.soft_update(self.critic_target, self.critic, self.soft_update_tau)

            q_value_pg = self.actor(state)  # policy gradient
            obj_actor = -self.critic_target(state, q_value_pg).mean()

            self.actor_optimizer.zero_grad()
            obj_actor.backward()
            self.actor_optimizer.step()
            self.soft_update(self.actor_target, self.actor, self.soft_update_tau)
        return obj_actor.item(), obj_critic.item()


@AGENTS.register_module()
class AgentTD3(AgentBase):
    def __init__(
        self,
        actor=None,
        actor_target=None,
        critic=None,
        critic_target=None,
        optimizer=None,
        **kwargs
    ):
        super().__init__(actor, actor_target, critic, critic_target, **kwargs)
        self.explore_noise = kwargs.get("explore_noise", 0.1)  # standard deviation of explore noise
        self.policy_noise = kwargs.get("policy_noise", 0.2)  # standard deviation of policy noise
        self.update_freq = kwargs.get("update_freq", 2)  # delay soft update frequency

        self.actor = build_net(actor, default_args=kwargs).to(device)
        self.actor_target = deepcopy(self.actor)
        self.critic = build_net(critic, default_args=kwargs).to(device)
        self.critic_target = deepcopy(self.critic)

        self.criterion = torch.nn.MSELoss()
        optimizer["params"] = self.actor.parameters()
        self.actor_optimizer = build_optimizer(optimizer)
        optimizer["params"] = self.critic.parameters()
        self.critic_optimizer = build_optimizer(optimizer)

    def select_action(self, state) -> np.ndarray:
        states = torch.as_tensor((state,), dtype=torch.float32, device=device).detach_()
        action = self.actor(states)[0]
        action = (action + torch.randn_like(action) * self.explore_noise).clamp(-1, 1)
        return action.cpu().numpy()

    def update_net(self, buffer, target_step, batch_size, repeat_times) -> (float, float):
        buffer.update_now_len_before_sample()

        obj_critic = obj_actor = None
        for i in range(int(target_step * repeat_times)):
            """objective of critic (loss function of critic)"""
            with torch.no_grad():
                reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
                next_a = self.actor_target.get_action(next_s, self.policy_noise)  # policy noise
                next_q = torch.min(*self.critic_target.get_q1_q2(next_s, next_a))  # twin critics
                q_label = reward + mask * next_q
            q1, q2 = self.critic.get_q1_q2(state, action)
            obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)  # twin critics

            self.critic_optimizer.zero_grad()
            obj_critic.backward()
            self.critic_optimizer.step()
            if i % self.update_freq == 0:  # delay update
                self.soft_update(self.critic_target, self.critic, self.soft_update_tau)

            """objective of actor"""
            q_value_pg = self.actor(state)  # policy gradient
            obj_actor = -self.critic_target(state, q_value_pg).mean()

            self.actor_optimizer.zero_grad()
            obj_actor.backward()
            self.actor_optimizer.step()
            if i % self.update_freq == 0:  # delay update
                self.soft_update(self.actor_target, self.actor, self.soft_update_tau)

        return obj_actor.item(), obj_critic.item() / 2
