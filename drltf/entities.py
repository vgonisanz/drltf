"""
Pydantic entities to help Python obj <--> json conversions and lazy checks
"""
from pydantic import BaseModel


class EnvironmentConfig(BaseModel):
    gym_name: str
    n_envs: int


class TrainingConfig(BaseModel):
    policy: str = ''
    n_steps: int = 0
    batch_size: int = 0
    n_epochs: int = 0
    gamma: float = 0.0
    gae_lambda: float = 0.0
    ent_coef: float = 0.0


class ModelTrainResult(BaseModel):
    learning_rate: float = 0.0
    clip_range: float = 0.0
    training_time: float = 0.0


class TrainReport(BaseModel):
    environment_config: EnvironmentConfig
    model_config: TrainingConfig
    result: ModelTrainResult


class EvaluationConfig(BaseModel):
    n_eval_episodes: int = 10
    deterministic: bool = True


class ModelEvaluationResult(BaseModel):
    mean_reward: float
    std_reward: float
    evaluation_time: float


class EvaluationReport(BaseModel):
    environment_config: EnvironmentConfig
    evaluation_config: EvaluationConfig
    result: ModelEvaluationResult
