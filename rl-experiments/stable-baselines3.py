import argparse
import os
import time

import gym
import pybullet_envs
import stable_baselines3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

from utils import ParseKwargs


parser = argparse.ArgumentParser(description="RL experiment runner")
parser.add_argument(
    "--algorithm", type=str, default="PPO", help="the name of this experiment"
)
parser.add_argument(
    "--gym-id",
    type=str,
    default="KukaDiverseObjectGrasping-v0",
    help="the id of the gym environment",
)
parser.add_argument(
    "--n-envs", type=int, default=1, help="Number of parallel environments to run"
)
parser.add_argument("--wandb", action="store_true", help="use wandb to log outputs")
parser.add_argument(
    "--wandb-project-name",
    type=str,
    default="rl-experiments",
    help="the wandb's project name",
)
parser.add_argument(
    "--load-model",
    type=str,
    default="",
    help="optionally specify a model name (within the model dir) to load and continue training",
)
parser.add_argument("--algorithm-kwargs", nargs="*", action=ParseKwargs)
args = parser.parse_args()
experiment_name = (
    f"{args.gym_id}__{args.algorithm}__{args.n_envs}_envs__{int(time.time())}"
)

if args.wandb:
    import wandb

    wandb.init(
        project=args.wandb_project_name,
        sync_tensorboard=True,
        config=vars(args),
        name=experiment_name,
        save_code=True,
    )

# Setup directories
log_dir = f"logs/{experiment_name}"
os.makedirs(log_dir, exist_ok=True)
model_dir = f"{log_dir}/models"
os.makedirs(model_dir, exist_ok=True)
tensorboard_dir = f"{log_dir}/tensorboard"
os.makedirs(tensorboard_dir, exist_ok=True)

# Create and wrap the environment
env = make_vec_env(args.gym_id, n_envs=args.n_envs, monitor_dir=log_dir)
env = gym.make(args.gym_id)
env = Monitor(env, log_dir)
env = DummyVecEnv([lambda: env])
env = VecNormalize(env, norm_obs=True, clip_obs=10.0)

# Create the model
def get_algorithm_by_name(name):
    if name == "PPO":
        return stable_baselines3.PPO
    elif name == "SAC":
        return stable_baselines3.SAC
    elif name == "TD3":
        return stable_baselines3.TD3
    elif name == "DDPG":
        return stable_baselines3.DDPG


algorithm = get_algorithm_by_name(args.algorithm)
if args.load_model:
    model = algorithm.load(
        f"{model_dir}/{args.load_model}",
        tensorboard_log=tensorboard_dir,
    )
    model.set_env(env)
else:
    model = algorithm(
        "CnnPolicy",
        env,
        tensorboard_log=tensorboard_dir,
        verbose=1,
        **args.algorithm_kwargs,
    )

# Setup callbacks
callback = CheckpointCallback(save_freq=50000, save_path=model_dir)
callback.n_calls = model.num_timesteps

model.learn(
    total_timesteps=int(1e6),
    callback=callback,
    tb_log_name="run",
    reset_num_timesteps=False,
)
