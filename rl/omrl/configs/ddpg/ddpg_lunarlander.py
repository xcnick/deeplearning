_base_ = ["../env/LunarLander.py", "../train_args.py"]
agent = dict(
    type="AgentTD3",
    actor=dict(type="Actor", net_dim=256),
    critic=dict(type="Critic", net_dim=256),
    optimizer=dict(type="Adam", lr=1e-4),
)
train_args = dict(policy_noise=0.2, update_freq=2)
