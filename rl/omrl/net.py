import torch
import torch.nn as nn
import numpy as np


from .builder import NETS


"""Q networks"""


@NETS.register_module()
class QNet(nn.Module):
    """
    Q Network

    """

    def __init__(self, net_dim, **kwargs):
        super().__init__()
        self.state_dim = kwargs.get("state_dim")
        self.action_dim = kwargs.get("action_dim")
        self.net = nn.Sequential(
            nn.Linear(self.state_dim, net_dim),
            nn.ReLU(),
            nn.Linear(net_dim, net_dim),
            nn.ReLU(),
            nn.Linear(net_dim, net_dim),
            nn.ReLU(),
            nn.Linear(net_dim, self.action_dim),
        )

    def forward(self, state):
        return self.net(state)  # Q value


@NETS.register_module()
class QNetDuel(nn.Module):
    """
    Dueling DQN

    """

    def __init__(self, net_dim, **kwargs):
        super().__init__()
        self.state_dim = kwargs.get("state_dim")
        self.action_dim = kwargs.get("action_dim")
        self.net_state = nn.Sequential(
            nn.Linear(self.state_dim, net_dim), nn.ReLU(), nn.Linear(net_dim, net_dim), nn.ReLU()
        )
        self.net_val = nn.Sequential(
            nn.Linear(net_dim, net_dim), nn.ReLU(), nn.Linear(net_dim, 1)
        )  # Q value
        self.net_adv = nn.Sequential(
            nn.Linear(net_dim, net_dim), nn.ReLU(), nn.Linear(net_dim, self.action_dim)
        )  # advantage function value 1

    def forward(self, state):
        t_tmp = self.net_state(state)
        q_val = self.net_val(t_tmp)
        q_adv = self.net_adv(t_tmp)
        return q_val + q_adv - q_adv.mean(dim=1, keepdim=True)  # dueling Q value


@NETS.register_module()
class QNetTwin(nn.Module):
    """
    Double DQN

    """

    def __init__(self, net_dim, **kwargs):
        super().__init__()
        self.state_dim = kwargs.get("state_dim")
        self.action_dim = kwargs.get("action_dim")
        self.net_state = nn.Sequential(
            nn.Linear(state_dim, net_dim), nn.ReLU(), nn.Linear(net_dim, net_dim), nn.ReLU()
        )  # state
        self.net_q1 = nn.Sequential(
            nn.Linear(net_dim, net_dim), nn.ReLU(), nn.Linear(net_dim, action_dim)
        )  # q1 value
        self.net_q2 = nn.Sequential(
            nn.Linear(net_dim, net_dim), nn.ReLU(), nn.Linear(net_dim, action_dim)
        )  # q2 value

    def forward(self, state):
        tmp = self.net_state(state)
        return self.net_q1(tmp)  # one Q value

    def get_q1_q2(self, state):
        tmp = self.net_state(state)
        q1 = self.net_q1(tmp)
        q2 = self.net_q2(tmp)
        return q1, q2  # two Q values


@NETS.register_module()
class QNetTwinDuel(nn.Module):
    """
    Dueling Double DQN

    """

    def __init__(self, net_dim, **kwargs):
        super().__init__()
        self.state_dim = kwargs.get("state_dim")
        self.action_dim = kwargs.get("action_dim")
        self.net_state = nn.Sequential(
            nn.Linear(state_dim, net_dim), nn.ReLU(), nn.Linear(net_dim, net_dim), nn.ReLU()
        )
        self.net_val1 = nn.Sequential(
            nn.Linear(net_dim, net_dim), nn.ReLU(), nn.Linear(net_dim, 1)
        )  # q1 value
        self.net_val2 = nn.Sequential(
            nn.Linear(net_dim, net_dim), nn.ReLU(), nn.Linear(net_dim, 1)
        )  # q2 value
        self.net_adv1 = nn.Sequential(
            nn.Linear(net_dim, net_dim), nn.ReLU(), nn.Linear(net_dim, action_dim)
        )  # advantage function value 1
        self.net_adv2 = nn.Sequential(
            nn.Linear(net_dim, net_dim), nn.ReLU(), nn.Linear(net_dim, action_dim)
        )  # advantage function value 1

    def forward(self, state):
        t_tmp = self.net_state(state)
        q_val = self.net_val1(t_tmp)
        q_adv = self.net_adv1(t_tmp)
        return q_val + q_adv - q_adv.mean(dim=1, keepdim=True)  # one dueling Q value

    def get_q1_q2(self, state):
        tmp = self.net_state(state)

        val1 = self.net_val1(tmp)
        adv1 = self.net_adv1(tmp)
        q1 = val1 + adv1 - adv1.mean(dim=1, keepdim=True)

        val2 = self.net_val2(tmp)
        adv2 = self.net_adv2(tmp)
        q2 = val2 + adv2 - adv2.mean(dim=1, keepdim=True)
        return q1, q2  # two dueling Q values


"""Policy Network Actor"""


@NETS.register_module()
class Actor(nn.Module):
    def __init__(self, net_dim, **kwargs):
        super().__init__()
        self.state_dim = kwargs.get("state_dim")
        self.action_dim = kwargs.get("action_dim")
        self.net = nn.Sequential(
            nn.Linear(self.state_dim, net_dim),
            nn.ReLU(),
            nn.Linear(net_dim, net_dim),
            nn.ReLU(),
            nn.Linear(net_dim, net_dim),
            nn.ReLU(),
            nn.Linear(net_dim, self.action_dim),
        )

    def forward(self, state):
        return self.net(state).tanh()  # action.tanh()

    def get_action(self, state, action_std):
        action = self.net(state).tanh()
        noise = (torch.randn_like(action) * action_std).clamp(-0.5, 0.5)
        return (action + noise).clamp(-1.0, 1.0)


"""Value Network Critic"""


@NETS.register_module()
class Critic(nn.Module):
    def __init__(self, net_dim, **kwargs):
        super().__init__()
        self.state_dim = kwargs.get("state_dim")
        self.action_dim = kwargs.get("action_dim")
        self.net = nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, net_dim),
            nn.ReLU(),
            nn.Linear(net_dim, net_dim),
            nn.ReLU(),
            nn.Linear(net_dim, 1),
        )

    def forward(self, state, action):
        return self.net(torch.cat((state, action), dim=1))  # Q value


@NETS.register_module()
class CriticTwin(nn.Module):
    def __init__(self, net_dim, if_use_dn=False, **kwargs):
        super().__init__()
        self.state_dim = kwargs.get("state_dim")
        self.action_dim = kwargs.get("action_dim")
        self.net_sa = nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, net_dim),
            nn.ReLU(),
            nn.Linear(net_dim, net_dim),
            nn.ReLU(),
        )

        self.net_q1 = nn.Linear(net_dim, 1)
        self.net_q2 = nn.Linear(net_dim, 1)
        # layer_norm(self.net_q1, std=0.1)
        # layer_norm(self.net_q2, std=0.1)

    def forward(self, state, action):
        tmp = self.net_sa(torch.cat((state, action), dim=1))
        return self.net_q1(tmp)  # one Q value

    def get_q1_q2(self, state, action):
        tmp = self.net_sa(torch.cat((state, action), dim=1))
        return self.net_q1(tmp), self.net_q2(tmp)  # two Q values
