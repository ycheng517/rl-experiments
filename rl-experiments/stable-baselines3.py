import argparse
import os
import time

import gym
import pybullet_envs
from gym.wrappers import Monitor as BMonitor

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor as SBMonitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

parser = argparse.ArgumentParser(description='PPO agent')
parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                    help='the name of this experiment')
parser.add_argument('--gym-id', type=str, default="KukaDiverseObjectGrasping-v0",
                    help='the id of the gym environment')
parser.add_argument('--wandb', action='store_true',
                    help='use wandb to log outputs')
parser.add_argument('--wandb-project-name', type=str, default="rl-experiments",
                    help="the wandb's project name")
parser.add_argument(
    '--load-model', 
    type=str, 
    default="", 
    help="optionally specify a model name (within the model dir) to load and continue training"
)
args = parser.parse_args()
experiment_name = f"{args.gym_id}__{args.exp_name}__{int(time.time())}"

if args.wandb:
    import wandb
    wandb.init(
        project=args.wandb_project_name,
        sync_tensorboard=True, 
        config=vars(args),
        name=experiment_name, 
        monitor_gym=True, 
        save_code=True
    )

# Setup directories
log_dir = f"logs/{experiment_name}"
model_dir = f"{log_dir}/models"
video_dir = f"{log_dir}/videos"
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
env = gym.make(args.gym_id)
env = BMonitor(env, video_dir)
env = SBMonitor(env, log_dir)
env = DummyVecEnv([lambda: env])
env = VecNormalize(env, norm_obs=True, clip_obs=10.)

# Create the model
if args.load_model:
    env.reset()
    model = PPO.load(
        f"{model_dir}/{args.load_model}",
        tensorboard_log=f"{log_dir}/tensorboard/",
    )
    model.set_env(env)
else:
    model = PPO(
        'CnnPolicy',
        env,
        tensorboard_log=f"{log_dir}/tensorboard/",
        verbose=1)

# Setup callbacks
callback = CheckpointCallback(save_freq=10000, save_path=model_dir)
callback.n_calls = model.num_timesteps

model.learn(
    total_timesteps=int(1e6), 
    callback=callback,
    tb_log_name="run",
    reset_num_timesteps=False,
)
