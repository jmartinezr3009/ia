# train_rl.py
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from rcss_gym_env import RcssGymEnv

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_single_agent(home_pos, total_timesteps=200_000):
    env = RcssGymEnv(home_pos=home_pos, max_steps=1000)
    # SB3 requires vector env; for simplicity usamos DummyVecEnv via helper
    from stable_baselines3.common.vec_env import DummyVecEnv
    venv = DummyVecEnv([lambda: env])

    policy_kwargs = dict(activation_fn=None, net_arch=[dict(pi=[128,128], vf=[128,128])])
    model = PPO("MlpPolicy", venv, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log="./tb_logs")

    checkpoint_cb = CheckpointCallback(save_freq=20000, save_path=MODEL_DIR, name_prefix="ppo_rcss")

    model.learn(total_timesteps=total_timesteps, callback=checkpoint_cb)
    model.save(os.path.join(MODEL_DIR, "ppo_rcss_final"))
    env.close()
    return model

if _name_ == "_main_":
    # ejemplo: entrenar un jugador con home en posición de delantero central
    home_pos = (-10.0, 0.0)  # ajústalo según tu conf_file
    model = train_single_agent(home_pos, total_timesteps=150_000)
    print("Entrenamiento finalizado y modelo guardado.")