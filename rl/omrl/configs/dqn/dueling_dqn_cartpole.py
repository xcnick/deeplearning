_base_ = ["../env/CartPole.py", "../train_args.py"]
agent = dict(
    type="AgentDuelingDQN",
    critic=dict(type="QNetDuel", net_dim=256),
    optimizer=dict(type="Adam", lr=1e-4),
)
train_args = dict(explore_rate=0.25)
