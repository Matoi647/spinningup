from spinup import vpg_pytorch
import torch
import gym

env_fn = lambda : gym.make('CartPole-v0')

ac_kwargs = dict(hidden_sizes=[64,64], activation=torch.nn.ReLU)

logger_kwargs = dict(output_dir='results/vpg-test', exp_name='vpg-test')

vpg_pytorch(env_fn=env_fn, 
    ac_kwargs=ac_kwargs, 
    seed=0, 
    steps_per_epoch=4000, 
    epochs=50, 
    gamma=0.99, 
    pi_lr=0.0003, 
    vf_lr=0.001, 
    train_v_iters=80, 
    lam=0.97, 
    max_ep_len=1000, 
    logger_kwargs=logger_kwargs, 
    save_freq=10)
