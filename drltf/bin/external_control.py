"""
Manual control based on state knowledge though external functions using states.

References:
    - https://gym.openai.com/docs/#observations
    - https://www.gymlibrary.ml/environments/box2d/lunar_lander/
    - https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py
"""
import typer

from drltf.utils import setup_logger
from drltf.core import Core
import gym
from pydantic import BaseModel


class ObservationSpace(BaseModel):
    """
    There are 8 states:
        - the coordinates of the lander in x & y,
        - its linear velocities in x & y,
        - its angle,
        - its angular velocity,
        - two booleans that represent whether each leg is in contact with the ground or not.
    """
    name: str
    iteration: int
    x_pos: float
    y_pos: float
    x_speed: float
    y_speed: float
    angle: float
    angular_speed: float
    left_leg_contact: float
    right_leg_contact: float


def process_observation(data, iteration):
    """
    Try to round decimals vs dont.
    """
    return ObservationSpace(
        name = "LunarLander-v2",
        iteration = iteration,
        x_pos = data[0],
        y_pos = data[1],
        x_speed = data[2],
        y_speed = data[3],
        angle = data[4],
        angular_speed = data[5],
        left_leg_contact = data[6],
        right_leg_contact = data[7],
    )

def play_and_plot_observations(policy=None, random_seed=1):
    env = gym.make("LunarLander-v2")
    env.seed(random_seed)
    observations = []
    current_observation = process_observation(env.reset(), 0)
    observations.append(current_observation)

    for idx in range(1000):
        env.render()
        action = policy(current_observation)
        observation, reward, done, info = env.step(action)
        current_observation = process_observation(observation, idx)
        observations.append(current_observation)
        
        if done:
            break
    env.close()
    #print(observations)
    #plot_observations(observations)


def policy(observation):
    next_action = machine_state(observation)
    print(f"next_action: {next_action}")
    return next_action

def get_machine_state(observation):
    x_speed_threshold = 1e-1
    angular_speed_threshold = 1e-1
    angle_threshold = 1e-1
    
    if abs(observation.x_speed) > x_speed_threshold:
        return 'reduce_x_speed'
    elif abs(observation.angular_speed) > angular_speed_threshold or abs(observation.angle) > angle_threshold:
        return 'stabilize_orientation'
    else:
        return 'hover'

def machine_state(observation):
    state = get_machine_state(observation)
    print(f"state: {state}")
    if state == 'reduce_x_speed':
        action = reduce_x_speed(observation)
        print(f"action: {action}")
        return action
    elif state == 'stabilize orientation':
        return stabilize_orientation(observation)
    elif state == 'hover':
        return hover(observation)
    return 0 

def hover(observation):
    if observation.pos_y < 1.5 and observation.y_speed < 0:
        return 2
    else:
        return 0 

def reduce_x_speed(observation):
    if observation.x_speed > 0:
        if observation.angle > 0:
            return 2
        else:
            return 1 
    else: 
        if observation.angle < 0:
            return 2
        else:
            return 3 

def stabilize_orientation(observation, angular_speed_threshold=1e-2, angle_threshold=1e-2):
    if observation.angular_speed > angular_speed_threshold and observation.angle > angle_threshold:
        return 3
    elif observation.angular_speed < -angular_speed_threshold and observation.angle < -angle_threshold:
        return 1

    if observation.angle > angle_threshold and observation.angular_speed > -angular_speed_threshold:
        return 1
    elif observation.angle < -angle_threshold and observation.angular_speed < angular_speed_threshold:
        return 3 
    return 0

# def plot_observations(observations):
#     observations = np.array(observations)
#     plt.figure(figsize=(30, 5))
#     plt.subplot(1, 3, 1)
#     plt.title('Trajectory')
#     plt.scatter(observations[:, 0], observations[:, 1], c=np.arange(len(observations)))
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.ylim(-0.5, 3.25)
#     plt.xlim(-1, 1)
#     plt.grid()
    
#     plt.subplot(1, 3, 2)
#     plt.title('Speed')
#     plt.plot(observations[:, 2], label='x_speed')
#     plt.plot(observations[:, 3], label='y_speed')
#     plt.plot(observations[:, 5], label='angular speed')
#     plt.legend(loc=0)
#     plt.grid()
    
#     plt.subplot(1, 3, 3)
#     plt.title('Orientation')
#     plt.plot(observations[:, 4])
#     plt.grid()


def main(n_envs: int = 1,
         model_name: str = "vgoni-model",
         models_path: str = "models"):

    typer.echo(f"Running {__file__}")

    # core = Core(n_envs=n_envs, models_path=models_path)
    # core.load_model(model_name)
    # core.render()
    play_and_plot_observations(policy)


if __name__ == "__main__":
    setup_logger()
    typer.run(main)
