import argparse
import numpy as np
import os
import torch

from stable_baselines3 import PPO, A2C, TD3

from envs.shepherd_env import ShepherdEnv
from agents.rule_based_agent import RuleBasedShepherd, TipsyShepherd, LazyShepherd
from agents.CNN_QN import ImageDQNAgent, N_ACTIONS, render_env_to_rgb, ANGLES, transform


def load_agent(agentType: str, model_name: str, env: ShepherdEnv):
    """Load the selected agent."""
    if agentType == "ruleBase":
        # print("A Rule-Based Shepherd Agent ...")
        return RuleBasedShepherd()
    if agentType == "tipsy":
        # print("A Tipsy Shepherd Agent ...")
        return TipsyShepherd()
    if agentType == "lazy":
        # print("A Lazy Shepherd Agent ...")
        return LazyShepherd()
    if agentType == "DQN" and os.path.exists(model_name):
        print(f"Using DQN Agent from {model_name}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        agent = ImageDQNAgent(
                            n_actions=N_ACTIONS,
                            lr=1e-4,
                            gamma=0.99,
                            device=device
                        )
        agent.q_net.load_state_dict(torch.load(model_name, map_location=device))
        agent.q_net.eval()
        return agent

    if os.path.exists(model_name):
        print(f"Using {agentType} Agent from {model_name}")
        return globals()[agentType].load(
            model_name,
            env=env,
            device="cpu",
        )
    else:
        # Fallback si aucun modèle n'est fourni pour les types non-basiques
        if agentType not in ["ruleBase", "lazy", "tipsy"]:
             raise ValueError(f"Model path does not exist: {model_name}")
        return None


def run_episode(env: ShepherdEnv, agent, model_type: str, display_flag=False):
    """
    Exécute un épisode et retourne :
    - reward_total : la somme des récompenses
    - is_success : True si les moutons sont dans l'enclos à la fin
    - steps : le nombre de pas effectués
    """
    obs = env.reset()
    done = False
    reward_total = 0.0

    while not done:
        if model_type in ["ruleBase", "lazy", "tipsy"]:
            actions = agent.act(obs)
        elif model_type == "DQN":
            state = transform(render_env_to_rgb(env))
            with torch.no_grad():
                action_idx = agent.select_action(state)
                actions = [ANGLES[action_idx]]
        else:
            actions, _ = agent.predict(obs, deterministic=True)

        obs, reward, done, _ = env.step(actions)
        reward_total += reward
        
        if display_flag:
            env.render()

    # --- Vérification du Succès ---
    # On considère un succès si tous les moutons sont dans le rayon du but
    # La méthode _max_sheep_goal_dist() renvoie la distance du mouton le plus loin.
    is_success = env._max_sheep_goal_dist() < env.goal_radius

    return reward_total, is_success, env.steps


def main():
    parser = argparse.ArgumentParser(description="Shepherd Environment Test Runner")

    parser.add_argument(
        "-a", "--agent_dir",
        type=str,
        default="models/",
        help="Path to saved model.",
    )

    parser.add_argument(
        "-t", "--agentType",
        type=str,
        choices=["ruleBase", "lazy", "tipsy", "PPO", "DQN", "TD3"],
        default="ruleBase",
        help="Agent model type.",
    )

    parser.add_argument(
        "-n", "--num_episodes",
        type=int,
        default=1,
        help="Number of episodes to run.",
    )

    parser.add_argument(
        "-s", "--num_sheep",
        type=int,
        default=1,
        help="Number of sheep.",
    )

    parser.add_argument(
        "-m", "--max_steps",
        type=int,
        default=500,
        help="Max steps per episode.",
    )

    parser.add_argument(
        "-r", "--obstacle_radius",
        type=float,
        default=0.0,
        help="Obstacle radius.",
    )

    parser.add_argument(
        "-g", "--goal_radius",
        type=float,
        default=0.7,
        help="Goal radius.",
    )

    # Ajout de l'argument active_sheep pour être cohérent avec le training
    parser.add_argument(
        "--active_sheep",
        action="store_true",
        help="Enable active sheep (random movement).",
    )

    args = parser.parse_args()

    rewards = []
    successes = []
    durations = []

    # Initialisation de l'environnement (une seule fois pour charger le modèle, 
    # mais recréé dans la boucle si besoin ou reset)
    env = ShepherdEnv(n_sheep=args.num_sheep,
                    max_steps=args.max_steps,
                    obstacle_radius=args.obstacle_radius,
                    goal_radius=args.goal_radius,
                    active_sheep=args.active_sheep)
    
    print(f"--- Testing Agent: {args.agentType} ---")
    print(f"Configuration: Sheep={args.num_sheep}, Obstacle={args.obstacle_radius}, Active={args.active_sheep}")

    agent = load_agent(args.agentType, args.agent_dir, env)

    for eps in range(1, args.num_episodes + 1):
        # On reset l'env à chaque épisode (fait automatiquement par run_episode via env.reset())
        # Note: Recréer l'env n'est pas nécessaire sauf si on changeait les paramètres, 
        # mais on garde la même instance pour performance.
        
        r, s, d = run_episode(env, agent, args.agentType, display_flag=(args.num_episodes == 1))

        rewards.append(r)
        successes.append(1 if s else 0)
        durations.append(d)

        if args.num_episodes == 1:
            status = "SUCCESS" if s else "FAIL"
            print(f"Episode {eps}: Reward={r:.2f}, Duration={d}, Status={status}")

    env.close()

    if args.num_episodes > 1:
        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        ci_reward = 1.96 * (std_reward / np.sqrt(len(rewards)))

        success_rate = np.mean(successes) * 100.0
        avg_duration = np.mean(durations)

        print("\n" + "="*40)
        print(f"RESULTS ({args.num_episodes} episodes)")
        print("="*40)
        print(f"Avg Reward    : {avg_reward:.2f} ± {ci_reward:.2f}")
        print(f"Success Rate  : {success_rate:.1f}%")
        print(f"Avg Duration  : {avg_duration:.1f} steps")
        print("="*40 + "\n")

if __name__ == "__main__":
    main()