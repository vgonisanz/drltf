"""
Core of the project to manage all IA stuff
"""
import os
import time
import json
import structlog

import gym
from gym.utils import play

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from huggingface_hub import HfApi
from huggingface_sb3 import package_to_hub

from drltf.utils import timeit
from drltf.window import Window
from drltf.entities import (
    EnvironmentConfig,
    TrainingConfig,
    ModelTrainResult,
    TrainReport,
    EvaluationConfig,
    EvaluationReport,
    ModelEvaluationResult
)

logger = structlog.get_logger(__file__)


class Core():
    """
    Main class to control gym, training, evaluations and all model stuff.
    """
    def __init__(self, models_path="models", gym_name="LunarLander-v2", n_envs=1):
        self._model = None
        self._training_config = None
        self._evaluation_config = None
        self._model_train_report = None
        self._models_path = models_path
        self._model_folder_path = None
        self._gym_name = gym_name
        self._n_envs = n_envs
        self._env = make_vec_env(self._gym_name, n_envs=n_envs)
        if not os.path.isdir(self._models_path):
            logger.warning("model_do_not_exist", path=self._models_path)
        logger.info("created_core", gym=self._gym_name, n_envs=n_envs)

    def load_model(self, model_name):
        self._model_folder_path, model_file_path = self._compose_model_path(self._models_path, model_name)

        if not self._check_if_model_exist(model_file_path):
            return False

        logger.info("loading_model", name=model_name)

        self._model = PPO.load(model_file_path, print_system_info=True)
        return True

    def save_model(self, model_name):
        model_folder = os.path.join(self._models_path, model_name)

        if not os.path.isdir(model_folder):
            logger.debug("creating_model_folder", folder=model_folder)
            os.makedirs(model_folder)

        self._generate_model_file(model_name)
        self._generate_model_train_report_file(model_folder)

    @timeit
    def train(self, config: TrainingConfig, steps: int):
        if self._model:
            logger.warning("overriding_model_on_memory")

        self._env.reset()

        self._training_config = config

        self._model = PPO(
            env=self._env,
            verbose=1,
            **self._training_config.dict()
        )

        logger.info("training_start")
        ts = time.time()
        self._model.learn(total_timesteps=int(steps))
        te = time.time()
        logger.info("training_end")

        self._model_train_report = ModelTrainResult(
            learning_rate=self._model.learning_rate,
            clip_range=self._model.clip_range(-1),
            training_time=round(te-ts, 2)
        )

    @timeit
    def evaluate(self, config: EvaluationConfig):
        if not self._model:
            logger.error("not_model_loaded_aborting")
            return

        logger.info("evaluating_model_reset_environment")
        self._env.reset()

        self._evaluation_config = config

        ts = time.time()
        mean_reward, std_reward = evaluate_policy(
            self._model,
            self._env,
            **self._evaluation_config.dict()
        )
        te = time.time()

        report = EvaluationReport(
            environment_config=EnvironmentConfig(gym_name=self._gym_name, n_envs=self._n_envs),
            evaluation_config=self._evaluation_config,
            result=ModelEvaluationResult(
                mean_reward=round(mean_reward, 2),
                std_reward=round(std_reward, 2),
                evaluation_time=round(te-ts, 2)
            )
        )

        logger.info("evaluation_result", **report.result.dict())

        self._generate_model_evaluation_report_file(self._model_folder_path, report)

    def render(self, max_steps=5000, end_time=1):
        # pylint: disable=unused-variable

        if not self._model:
            logger.error("not_model_loaded")
            return

        obs = self._env.reset()
        with Window():
            logger.info("simulation_start", max_steps=max_steps)
            for i in range(max_steps):
                action, states = self._model.predict(obs, deterministic=True)
                obs, rewards, dones, info = self._env.step(action)
                if dones.all():
                    time.sleep(end_time)
                    break
                self._env.render()
            logger.info("simulation_end")

    @staticmethod
    def whoami():
        return HfApi().whoami()

    @staticmethod
    def backup(models_path, model_name, repo_id):
        api = HfApi()
        files = [
            os.path.join(models_path, model_name, "model.zip"),
            os.path.join(models_path, model_name, "model_eval_report.json"),
            os.path.join(models_path, model_name, "model_train_report.json")
        ]
        for file in files:
            logger.info("upload_file", file=file, repo_id=repo_id)
            api.upload_file(
                path_or_fileobj=file,
                path_in_repo=file,
                repo_id=repo_id
            )

    def publish(self, env_id, model_name, model_architecture, repo_id, commit_message):
        if not self.load_model(model_name):
            logger.error("load_model_error_abort_publish")
            return

        eval_env = DummyVecEnv([lambda: gym.make(env_id)])

        logger.info("package_to_hub", model_name=model_name, 
                                      model_architecture=model_architecture,
                                      env_id=env_id,
                                      repo_id=repo_id,
                                      commit_message=commit_message)

        package_to_hub(model=self._model,
                    model_name=model_name, 
                    model_architecture=model_architecture,
                    env_id=env_id,
                    eval_env=eval_env,
                    repo_id=repo_id,
                    commit_message=commit_message)

    def _generate_model_file(self, model_name):
        model_folder_path, model_file_path = self._compose_model_path(self._models_path, model_name)
        logger.info("saving_model", path=model_folder_path)
        self._model.save(model_file_path)

    def _generate_model_train_report_file(self, model_folder):
        report_path = os.path.join(model_folder, 'model_train_report.json')

        report = TrainReport(
            environment_config=EnvironmentConfig(gym_name=self._gym_name, n_envs=self._n_envs),
            model_config=self._training_config,
            result=self._model_train_report
        )

        logger.info("exporting_model_train_report", path=report_path)
        self._generate_json_from_dict(report_path, report.dict())

    def _generate_model_evaluation_report_file(self, model_folder, report):
        report_path = os.path.join(model_folder, 'model_eval_report.json')

        logger.info("exporting_model_evaluation_report", path=report_path)
        self._generate_json_from_dict(report_path, report.dict())

    @staticmethod
    def _compose_model_path(models_path, model_name):
        model_folder_path = os.path.join(models_path, model_name)
        model_file_path = os.path.join(model_folder_path, "model")
        return model_folder_path, model_file_path

    @staticmethod
    def _check_if_model_exist(model_path):
        if not os.path.isfile(model_path) and not os.path.isfile(model_path + ".zip"):
            logger.error("model_do_not_exist_at", path=model_path)
            return False
        return True

    @staticmethod
    def _generate_json_from_dict(path, content):
        with open(path, 'w') as fp:
            json.dump(content, fp, indent=4, sort_keys=True)

    @staticmethod
    def play(fps, zoom, key_map, print_reward):
        env = gym.make('LunarLander-v2')
        play.play(env, fps=fps, zoom=zoom, keys_to_action=key_map, callback=print_reward)
