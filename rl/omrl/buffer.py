import torch
import numpy as np


class ReplayBuffer:
    def __init__(self, max_len, state_dim, action_dim, if_on_policy, device):
        """Experience Replay Buffer

        save environment transition in a continuous RAM for high performance training
        we save trajectory in order and save state and other (action, reward, mask, ...) separately.

        :int max_len: the maximum capacity of ReplayBuffer. First In First Out
        :int state_dim: the dimension of state
        :int action_dim: the dimension of action (action_dim==1 for discrete action)
        :bool if_on_policy: on-policy or off-policy
        :bool device: create buffer space on CPU RAM or GPU
        """
        self.device = device
        self.max_len = max_len
        self.now_len = 0
        self.next_idx = 0
        self.if_full = False
        self.action_dim = action_dim  # for self.sample_all(
        self.if_on_policy = if_on_policy

        if if_on_policy:
            other_dim = 1 + 1 + action_dim * 2
        else:
            other_dim = 1 + 1 + action_dim

        self.buf_other = torch.empty((max_len, other_dim), dtype=torch.float32, device=self.device)
        self.buf_state = torch.empty((max_len, state_dim), dtype=torch.float32, device=self.device)

    def append_buffer(self, state, other):  # CPU array to CPU array
        state = torch.as_tensor(state, device=self.device)
        other = torch.as_tensor(other, device=self.device)
        self.buf_state[self.next_idx] = state
        self.buf_other[self.next_idx] = other

        self.next_idx += 1
        if self.next_idx >= self.max_len:
            self.if_full = True
            self.next_idx = 0

    # def extend_buffer(self, state, other):  # CPU array to CPU array
    #     state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
    #     other = torch.as_tensor(other, dtype=torch.float32, device=self.device)

    #     size = len(other)
    #     next_idx = self.next_idx + size
    #     if next_idx > self.max_len:
    #         if next_idx > self.max_len:
    #             self.buf_state[self.next_idx:self.max_len] = state[:self.max_len - self.next_idx]
    #             self.buf_other[self.next_idx:self.max_len] = other[:self.max_len - self.next_idx]
    #         self.if_full = True
    #         next_idx = next_idx - self.max_len

    #         self.buf_state[0:next_idx] = state[-next_idx:]
    #         self.buf_other[0:next_idx] = other[-next_idx:]
    #     else:
    #         self.buf_state[self.next_idx:next_idx] = state
    #         self.buf_other[self.next_idx:next_idx] = other
    #     self.next_idx = next_idx

    def sample_batch(self, batch_size) -> tuple:
        """randomly sample a batch of data for training

        :int batch_size: the number of data in a batch for Stochastic Gradient Descent
        :return torch.Tensor reward: reward.shape==(now_len, 1)
        :return torch.Tensor mask:   mask.shape  ==(now_len, 1), mask = 0.0 if done else gamma
        :return torch.Tensor action: action.shape==(now_len, action_dim)
        :return torch.Tensor state:  state.shape ==(now_len, state_dim)
        :return torch.Tensor state:  state.shape ==(now_len, state_dim), next state
        """
        indices = torch.randint(self.now_len - 1, size=(batch_size,), device=self.device)
        r_m_a = self.buf_other[indices]
        return (
            r_m_a[:, 0:1],
            r_m_a[:, 1:2],
            r_m_a[:, 2:],
            self.buf_state[indices],
            self.buf_state[indices + 1],
        )

    # def sample_all(self) -> tuple:
    #     """sample all the data in ReplayBuffer (for on-policy)

    #     :return torch.Tensor reward: reward.shape==(now_len, 1)
    #     :return torch.Tensor mask:   mask.shape  ==(now_len, 1), mask = 0.0 if done else gamma
    #     :return torch.Tensor action: action.shape==(now_len, action_dim)
    #     :return torch.Tensor noise:  noise.shape ==(now_len, action_dim)
    #     :return torch.Tensor state:  state.shape ==(now_len, state_dim)
    #     """
    #     all_other = torch.as_tensor(self.buf_other[:self.now_len], device=self.device)
    #     return (all_other[:, 0],
    #             all_other[:, 1],
    #             all_other[:, 2:2 + self.action_dim],
    #             all_other[:, 2 + self.action_dim:],
    #             self.buf_state[:self.now_len])

    def update_now_len_before_sample(self):
        """update the a pointer `now_len`, which is the current data number of ReplayBuffer
        """
        self.now_len = self.max_len if self.if_full else self.next_idx

    def empty_buffer_before_explore(self):
        """we empty the buffer by set now_len=0. On-policy need to empty buffer before exploration
        """
        self.next_idx = 0
        self.now_len = 0
        self.if_full = False

    def print_state_norm(self, neg_avg=None, div_std=None):  # non-essential
        max_sample_size = 2 ** 14

        """check if pass"""
        state_shape = self.buf_state.shape
        if len(state_shape) > 2 or state_shape[1] > 64:
            print(
                f"| print_state_norm(): state_dim: {state_shape} is too large to print its norm. "
            )
            return None

        """sample state"""
        indices = np.arange(self.now_len)
        rd.shuffle(indices)
        indices = indices[:max_sample_size]  # len(indices) = min(self.now_len, max_sample_size)

        batch_state = self.buf_state[indices]

        """compute state norm"""
        if isinstance(batch_state, torch.Tensor):
            batch_state = batch_state.cpu().data.numpy()
        assert isinstance(batch_state, np.ndarray)

        if batch_state.shape[1] > 64:
            print(
                f"| _print_norm(): state_dim: {batch_state.shape[1]:.0f} is too large to print its norm. "
            )
            return None

        if np.isnan(batch_state).any():  # 2020-12-12
            batch_state = np.nan_to_num(batch_state)  # nan to 0

        ary_avg = batch_state.mean(axis=0)
        ary_std = batch_state.std(axis=0)
        fix_std = ((np.max(batch_state, axis=0) - np.min(batch_state, axis=0)) / 6 + ary_std) / 2

        if neg_avg is not None:  # norm transfer
            ary_avg = ary_avg - neg_avg / div_std
            ary_std = fix_std / div_std

        print(f"| print_norm: state_avg, state_fix_std")
        print(f"| avg = np.{repr(ary_avg).replace('=float32', '=np.float32')}")
        print(f"| std = np.{repr(ary_std).replace('=float32', '=np.float32')}")
