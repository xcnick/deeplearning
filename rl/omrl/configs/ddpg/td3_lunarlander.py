_base_ = ["../env/LunarLander.py", "../train_args.py"]
agent = dict(
    type="AgentTD3",
    actor=dict(type="Actor", net_dim=256),
    critic=dict(type="CriticTwin", net_dim=256),
    optimizer=dict(type="Adam", lr=1e-4),
)
