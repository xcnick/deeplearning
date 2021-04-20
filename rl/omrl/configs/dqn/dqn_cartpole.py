_base_ = ["../env/CartPole.py", "../train_args.py"]
agent = dict(
    type="AgentDQN", critic=dict(type="QNet", net_dim=256), optimizer=dict(type="Adam", lr=1e-4)
)
