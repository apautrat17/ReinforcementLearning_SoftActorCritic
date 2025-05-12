import os
import cv2
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import ray
from ray import tune
from ray.rllib.algorithms.sac import SACConfig
from ray.tune.registry import register_env
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch

import torch.nn.init as init
import time


# =========================== Environnement de simulation de base ===========================
class EnvSimulation:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.agent_radius = 10
        self.agent_position = (
            np.random.randint(10, self.width - 10),
            np.random.randint(10, self.height - 10),
        )
        self.target_radius = 10
        self.target_position = (
            np.random.randint(10, self.width - 10),
            np.random.randint(10, self.height - 10),
        )
        self.background_color = (255, 255, 255)
        self.agent_color = (255, 0, 0)
        self.target_color = (0, 0, 255)
        self.window_name = "Simulation"
        self.window = None
        self.done = False
        self.reward = 0
        self.truncated = False
        self.terminated = False
        self.states = [(x, y) for x in range(0, width) for y in range(0, height)]
        self.angle = 0
        self.speed = 5

        self.n_steps = 0

        cv2.namedWindow(self.window_name)

        self.initial_distance = np.sqrt(
            (self.target_position[0] - self.agent_position[0]) ** 2
            + (self.target_position[1] - self.agent_position[1]) ** 2
        )
        self.prev_distance = self.initial_distance

        self.last_action = 0

    def action_space(self):
        # 360 actions discrétisées sur 32 positions
        return [i for i in range(0, 360, 360 // 32)]

    def sample(self):
        return np.random.choice(self.action_space())

    def state_space(self):
        return self.states

    def reset(self):
        self.agent_position = (
            np.random.randint(10, self.width - 10),
            np.random.randint(10, self.height - 10),
        )
        self.target_position = (
            np.random.randint(10, self.width - 10),
            np.random.randint(10, self.height - 10),
        )
        self.done = False
        self.reward = 0
        self.truncated = False
        self.terminated = False
        self.n_steps = 0
        self.initial_distance = np.sqrt(
            (self.target_position[0] - self.agent_position[0]) ** 2
            + (self.target_position[1] - self.agent_position[1]) ** 2
        )
        self.prev_distance = self.initial_distance
        return self.agent_position, {}

    def repartir_reward_autour_de_target(self):
        """Calculer la récompense avec gradient vers la cible
        Avec normalisation entre -1 et 1 sur l'épisode complet"""
        dx = self.target_position[0] - self.agent_position[0]
        dy = self.target_position[1] - self.agent_position[1]
        distance = np.sqrt(dx**2 + dy**2)

        # Sur la cible
        if distance < self.agent_radius + self.target_radius:
            self.done = True
            self.terminated = True
            self.prev_distance = distance
            return 1.0

        reward = (self.prev_distance - distance) / self.initial_distance

        self.prev_distance = distance

        return reward

    def step(self, action):

        # Convertir l'indice de l'action discrète en action réelle (car on utilise SAC)
        action = self.action_space()[action]

        self.n_steps += 1
        self.last_action = action

        # Si le nombre de steps dépasse 500, on termine l'épisode
        if self.n_steps >= 500:
            self.truncated = True
            return (
                self.agent_position,
                self.reward,
                self.terminated,
                self.truncated,
                {},
            )

        angle_rad = np.deg2rad(action)
        dx = int(self.speed * np.cos(angle_rad))
        dy = int(self.speed * np.sin(angle_rad))
        new_x = min(max(self.agent_position[0] + dx, 0), self.width - 1)
        new_y = min(max(self.agent_position[1] + dy, 0), self.height - 1)
        self.agent_position = (new_x, new_y)

        self.reward = self.repartir_reward_autour_de_target()

        return (
            self.agent_position,
            self.reward,
            self.terminated,
            self.truncated,
            {},
        )

    def render(self):
        # Fond blanc pour l'environnement de simulation
        self.window = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255

        # Agent
        cv2.circle(
            self.window, self.agent_position, self.agent_radius, self.agent_color, 5
        )

        # Target
        cv2.circle(
            self.window, self.target_position, self.target_radius, self.target_color, 5
        )

        # Afficher l'environnement
        cv2.imshow(self.window_name, self.window)

        # Attendre une touche ("q" ici) pour fermer la fenêtre
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.done = True
            self.truncated = True


# =========================== EnvSimulation enveloppé dans un environnement gym ===========================
class EnvGym(gym.Env):
    def __init__(self, config=None):
        super().__init__()
        self.env = EnvSimulation()

        self.action_space = Discrete(len(self.env.action_space()))

        self.observation_space = Box(
            low=np.array([0, 0, 0, 0, 0, -1, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 1, np.sqrt(2), 1, 1], dtype=np.float32),
            dtype=np.float32,
        )

    def _get_obs(self):
        x_a = self.env.agent_position[0] / self.env.width
        y_a = self.env.agent_position[1] / self.env.height
        x_t = self.env.target_position[0] / self.env.width
        y_t = self.env.target_position[1] / self.env.height

        dx = x_t - x_a
        dy = y_t - y_a
        distance = np.sqrt(dx**2 + dy**2)
        angle_to_target = np.arctan2(dy, dx) / np.pi  # Normalisé entre -1 et 1

        normalized_last_action = self.env.last_action / self.env.action_space()[-1]

        return np.array(
            [x_a, y_a, x_t, y_t, distance, angle_to_target, normalized_last_action],
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        self.env.reset()
        return self._get_obs(), {}

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        self.env.render()


# =========================== Entraînement de l'agent ===========================


class DiscreteToContinuousWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Convertir l'espace d'action en continu
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

    def action(self, action):
        # Mapper l'action continue [0, 1] à l'espace discret
        discrete_action = int(action[0] * (self.env.action_space.n - 1))
        return discrete_action


# Enregistrement environnement
def create_env(env_config):
    env = EnvGym()
    return DiscreteToContinuousWrapper(env)


# Enregistrement environnement dans Ray
register_env("SACEnv-v0", create_env)


# Enregistrement du modèle
BASE_DIR = os.path.abspath(os.getcwd())
LOG_DIR = os.path.join(BASE_DIR, "ray_results")
os.makedirs(LOG_DIR, exist_ok=True)
writer = SummaryWriter(log_dir=LOG_DIR)


# Initialisation
ray.init(ignore_reinit_error=True)


# Configuration SAC
config = (
    SACConfig()
    .environment("SACEnv-v0")
    .resources(num_gpus=1)
    .env_runners(num_env_runners=4)
    .training(
        gamma=0.95,
        actor_lr=0.0001,
        critic_lr=0.0002,
        train_batch_size_per_learner=64,
    )
)

# Entraînement de l'agent
tuner = tune.Tuner(
    "SAC",
    param_space=config.to_dict(),
    run_config=tune.RunConfig(
        stop={"training_iteration": 1000},
        verbose=2,
        storage_path=f"file://{LOG_DIR}",
        name="SAC_EnvSimulation",
        log_to_file=True,  # Enregistrement des logs, c-a-d résultats
        checkpoint_config=tune.CheckpointConfig(
            checkpoint_frequency=10,
            checkpoint_at_end=True,
            # num_to_keep=5,
        ),
    ),
)

# Lancer l'entraînement

# try:
#     results = tuner.fit()
# except Exception as e:
#     print(f"Training error: {e}")
# finally:
#     ray.shutdown()
# Décommenter ces lignes pour lancer l'entraînement


# # =========================== Test de l'agent ===========================

ray.init(ignore_reinit_error=True)
checkpoint_path = "ray_results/SAC_EnvSimulation/SAC_SACEnv-v0_983fa_00000_0_2025-05-12_11-00-47/checkpoint_000084"

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Modèle qui fonctionne: checkpoint_path = "ray_results/SAC_EnvSimulation/SAC_SACEnv-v0_983fa_00000_0_2025-05-12_11-00-47/checkpoint_000084"
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


algo = config.build()
algo.restore(os.path.abspath(checkpoint_path))

env = EnvGym()
obs, _ = env.reset()


def map_continuous_to_discrete(action, discrete_actions):
    """
    Conversion d'une action continue [-1, 1] en un indice d'action discret.
    """
    angle_01 = (action + 1) / 2  # [-1,1] → [0,1]
    angle_360 = angle_01 * 360  # [0,1] → [0,360]

    # Trouver l'angle discret le plus proche
    differences = [abs((angle_360 - a) % 360) for a in discrete_actions]
    return np.argmin(differences)


nb_success = 0
nb_episodes = 500

for i in range(nb_episodes):

    # Reset l'environnement
    obs, _ = env.reset()

    # Boucle
    done = False
    while not done:
        obs_tensor = {"obs": torch.tensor([obs], dtype=torch.float32)}
        module = algo.get_module("default_policy")
        action_dist_class = module.get_inference_action_dist_cls()

        with torch.no_grad():
            output = module.forward_inference(obs_tensor)
            action_dist = action_dist_class.from_logits(output["action_dist_inputs"])
            action_continuous = action_dist.sample()[0].item()  # Action continue

            # Mapper l'action continue à un indice discret
            action_idx = map_continuous_to_discrete(
                action_continuous, env.env.action_space()
            )

        obs, reward, terminated, truncated, _ = env.step(action_idx)
        env.render()
        if terminated:
            nb_success += 1
        done = terminated or truncated

    env.close()
    time.sleep(0.5)

print(f"Pourcentage de succès: {nb_success / nb_episodes * 100:.2f}%")


# Visualisation des résultats sur TensorBoard (copier ds un autre terminal)
# tensorboard --logdir ray_results --load_fast true
