import numpy as np
import torch
from stable_baselines3 import PPO, A2C, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement

def train_rl_agent_ppo_mlp(env, eval_env, timesteps=2000000, checkpoint_dir=None, criculam_learning=False):
    log_dir = f"./logs/ppo/"
    
    # Choix du device (CPU est souvent plus rapide pour PPO/MLP, mais on force CUDA si demandé)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if checkpoint_dir is not None:
        print(f"Loading checkpoint from {checkpoint_dir}...")
        model = PPO.load(checkpoint_dir,
                env=env,
                device=device,
                n_steps=2048,
                ent_coef=0.003,
                learning_rate=3e-4,
                gamma=0.99,
                verbose=1,
                tensorboard_log=log_dir
                )
        if criculam_learning:
            print("Reinitializing optimizer and PPO internals for curriculum learning...")
            model._setup_model()
            model.learning_rate = 1e-4    
    else:
        print("No checkpoint provided, training from scratch.")
        model = PPO("MlpPolicy",
                    env=env,
                    device=device,
                    n_steps=2048,
                    ent_coef=0.003,
                    learning_rate=3e-4,
                    gamma=0.99,
                    verbose=1,
                    tensorboard_log=log_dir
                    )
    
    # --- CONFIGURATION DE L'ARRÊT AUTOMATIQUE ---
    stop_train_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=5, 
        min_evals=10, 
        verbose=1
    )
    
    # On attache le stop_train_callback à l'intérieur du EvalCallback via 'callback_after_eval'
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=log_dir,
        log_path=log_dir, 
        eval_freq=20000,
        n_eval_episodes=20,
        deterministic=True, 
        render=False,
        callback_after_eval=stop_train_callback # <--- C'est ici que la correction se fait
    )
    
    model.learn(total_timesteps=timesteps, callback=eval_callback)
    return model

def train_rl_agent_td3_mlp(env, eval_env, timesteps=2000000, checkpoint_dir=None, criculam_learning=False):
    log_dir = f"./logs/td3/"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    
    model = TD3("MlpPolicy", env, verbose=1, device=device, tensorboard_log=log_dir, action_noise=action_noise)
    
    # --- CONFIGURATION DE L'ARRÊT AUTOMATIQUE (Même logique pour TD3) ---
    stop_train_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=5, 
        min_evals=10, 
        verbose=1
    )
    
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=log_dir,
        log_path=log_dir, 
        eval_freq=20000,
        n_eval_episodes=20,
        deterministic=True, 
        render=False,
        callback_after_eval=stop_train_callback
    )
    
    model.learn(total_timesteps=timesteps, callback=eval_callback)
    return model