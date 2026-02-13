import os
from envs.shepherd_env import ShepherdEnv
from agents.rl_agent import train_rl_agent_ppo_mlp,train_rl_agent_td3_mlp
from agents.CNN_QN import train_image_dqn,ImageDQNAgent,N_ACTIONS
import torch
import argparse


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train RL agents for ShepherdEnv.")
parser.add_argument("-a", "--algorithm", type=str,
                    choices=["td3", "dqn", "ppo", "all"], default="ppo",
                    help="Choose the algorithm to train: 'td3', 'a2c', 'ppo', or 'all'. Default is 'ppo'.")

parser.add_argument(
    "-s", "--num_sheep",
    type=int,
    default=1,
    help="Number of sheep in the environment.",
)

parser.add_argument(
    "-m", "--max_steps",
    type=int,
    default=500,
    help="Maximum number of steps per episode.",
)

parser.add_argument(
    "-r", "--obstacle_radius",
    type=float,
    default=0.0,
    help="Radius of obstacles in the environment.",
)

parser.add_argument(
    "-g", "--goal_radius",
    type=float,
    default=0.7,
    help="Radius of the goal area in the environment.",
    )

parser.add_argument(
    "-c", "--checkpoint_dir",
    type=str,
    default=None,
    help="Directory to load a checkpoint from. If not provided, training starts from scratch.",
)

parser.add_argument(
    "-cl", "--criculam_learning",
    action="store_true",
    help="Enable curriculum learning (Half duration + Reset optimizer).",
)

parser.add_argument(
    "--active_sheep",
    action="store_true",
    help="Enable activate sheep (random movement when shepherd is far). Default is False (Sleepy sheep)."
)

args = parser.parse_args()

# Valeurs par défaut (Train from Scratch)
current_timesteps = 2000000
current_dqn_episodes = 1000

if args.criculam_learning:
    print("\n" + "!"*60)
    print("CURRICULUM LEARNING ACTIVÉ")
    
    current_timesteps = 1000000  # 1 Million au lieu de 2
    current_dqn_episodes = 500   # 500 épisodes au lieu de 1000
    
    print(f"   -> Nouveaux Timesteps (PPO/TD3) : {current_timesteps}")
    print(f"   -> Nouveaux Episodes (DQN)      : {current_dqn_episodes}")
    print("!"*60 + "\n")
else:
    print(f"Training Mode: Standard (Full Duration: {current_timesteps} steps / {current_dqn_episodes} eps)")


env = ShepherdEnv(n_sheep=args.num_sheep,
                max_steps=args.max_steps,
                obstacle_radius=args.obstacle_radius,
                goal_radius=args.goal_radius,
                active_sheep=args.active_sheep)
eval_env = ShepherdEnv(n_sheep=args.num_sheep,
                        max_steps=args.max_steps,
                        obstacle_radius=args.obstacle_radius,
                        goal_radius=args.goal_radius,
                        active_sheep=args.active_sheep)

suffix = "active" if args.active_sheep else "sleepy"

if args.algorithm in ["dqn", "all"]:
    try:
        print(f"Training with CNN_QN (#sheep: {env.n_sheep})...")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        dqn_filename = f"dqn_sheep{env.n_sheep}_obst{int(args.obstacle_radius*10)}_{suffix}"

        agent = ImageDQNAgent(
                            n_actions=N_ACTIONS,
                            lr=1e-4,
                            gamma=0.99,
                            device=device
                        )

        if args.checkpoint_dir:
            if os.path.exists(args.checkpoint_dir):
                print(f"Loading DQN model from checkpoint: {args.checkpoint_dir} ...")
                agent.q_net.load_state_dict(torch.load(args.checkpoint_dir, map_location=device))

                if args.criculam_learning:
                    print("Curriculum Learning : Resetting exploration decay (epsilon).")
                    agent.eps_start = 0.5
            else:
                print(f"Warning: Checkpoint directory '{args.checkpoint_dir}' does not exist. Starting DQN training from scratch.")

        model=train_image_dqn(
                        env=env,
                        eval_env=eval_env,
                        agent=agent,
                        episodes=current_dqn_episodes,
                        batch_size=32,
                        target_update=1000,
                        eval_every=5,
                        eval_episodes=5,
                        save_name=dqn_filename
                    )
        
    except Exception as e:
        print(f"CNN_QN training failed: {e}")

if args.algorithm in ["ppo", "all"]:
    try:
        print(f"Training with PPO algorithm (#sheep: {env.n_sheep})...")
        model = train_rl_agent_ppo_mlp(env, eval_env, timesteps=current_timesteps,
                                       checkpoint_dir=args.checkpoint_dir,
                                       criculam_learning=args.criculam_learning)
        model.save(f"models/ppo_sheep{env.n_sheep}_obst{int(args.obstacle_radius*10)}_{suffix}")
    except Exception as e:
        print(f"PPO training failed: {e}")


if args.algorithm in ["td3", "all"]:
    try:
        print(f"Training with TD3 algorithm (#sheep: {env.n_sheep})...")
        model = train_rl_agent_td3_mlp(env, eval_env, timesteps=current_timesteps,
                                       checkpoint_dir=args.checkpoint_dir,
                                       criculam_learning=args.criculam_learning)
        model.save(f"models/td3_sheep{env.n_sheep}_obst{int(args.obstacle_radius*10)}_{suffix}")
    except Exception as e:
        print(f"TD3 training failed: {e}")
