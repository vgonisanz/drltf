<div align="center">

# vgonisanz's Deep Reinforcement Learning Training Framework (DRLEF)

vgonisanz's Deep Reinforcement Learning Training Framework to practice with Hugging face
chapter 1 for reinforcement learning course.

You can learn all about the chapter at [this blog entry](https://huggingface.co/blog/deep-rl-intro)

All the code in python notebooks can be found on [this Github repository](https://github.com/huggingface/deep-rl-class)

</div>

## Usage

### Training

To train a model, it would be by default generated at `models/test/model.zip`:

```bash
python drltf/bin/train.py --model-name test --n-envs 4 --steps 50000
```

Take a look to the training report at `models/test/model_train_report.json`

### Evaluating a model

To evaluate it:

```bash
python drltf/bin/eval.py --model-name test --n-envs 4 --no-render
```

Take a look to the training report at `models/test/model_eval_report.json`

### Render

You can analyze how a model works.

```bash
python drltf/bin/render.py --model-name test --n-envs 6
```

If you run a render in a empty x11 windows, you can use ffmpeg to record the screen with:

```bash
ffmpeg -f x11grab -s 1280x720 -i :0.0 -c:v libvpx -quality realtime -cpu-used 0 -b:v 384k -qmin 10 -qmax 42 -maxrate 384k -bufsize 1000k -an -filter:v "crop=in_w:in_h:in_w:in_h" screen.webm -y
```

With i3wm and X11 render window system, dont use full screen elements in order to render properly.
If not, you can get the following error `X connection to :15 broken (explicit kill or server shutdown).`

### Share your models

First time you have to:

1. Create a token at [huggingface account](https://huggingface.co/settings/tokens). You need to create one account before.
1. Create a repository for this project in the website. In example:
  - For backup models you can user: `vgonisanz/lunar-lander-models`
  - or for publish in the Leaderboard `vgonisanz/PPO-LunarLander-v2`.
1. Enable git credential: `git config --global credential.helper store`
1. Login though bash (Credential stored in cache): `huggingface-cli login`

#### Create a Backup of your model

To publish a backup of a model in other of your repositories, just run:

```bash
python drltf/bin/backup.py --model-name test --repo-name "lunar-lander-models"
```

#### Publish in leaderboard

To publish a new model just run:

```bash
python drltf/bin/publish_to_leaderboard.py "My very first model" --env-id LunarLander-v2 --model-architecture PPO --models-path models --model-name test
```

Require a repository created with `{whoami['name']}/{model_architecture}-{env_id}`.

### Configure a telegram bot to send you notifications in real time

You will need to set up the following environment variables:
- `TELEGRAM_NOTIFY_APIKEY`: With your bot conversation
- `TELEGRAM_NOTIFY_CHATID`: The ID of the conversation

Test with:

```
python drltf/bin/send_telegram_notification.py "Hello bot"
```

And you will have a message. If it works, just add use it on Python code:

```
from drltf.utils import notify_telegram
notify_telegram("Hello bot")
```

## Development

To start developing this project, clone this repo and do:

```
sudo apt install python-opengl \
                 ffmpeg \
                 xvfb \
                 libglfw3-dev \
                 swig
make env-create
```

This will create a virtual environment with all the needed dependencies (using [tox](https://tox.readthedocs.io/en/latest/)). You can activate this environment with:

```
$ source ./.tox/drltf/bin/activate
```

Then, you can run `make help`.
Learn more about the different tasks you can perform on this project using [make](https://www.gnu.org/software/make/).

### Upgrade dependencies

From scratch, use the following command to generate `requirements{-dev}.txt` files:

```
make env-compile
```

## Contributing

Please see the [Contributing Guidelines](./CONTRIBUTING.md) section for more details on how you can contribute to this project.

## License

[GNU General Public License v3](./LICENSE)
