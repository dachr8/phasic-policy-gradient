**Status:** Archive (code is provided as-is, no updates expected)

# Phasic Policy Gradient

#### [[Paper]](https://arxiv.org/abs/2009.04416)

This is code for training agents
using [Phasic Policy Gradient](https://arxiv.org/abs/2009.04416) [(citation)](#citation)
.

Supported platforms:

- macOS 10.14 (Mojave)
- Ubuntu 16.04

Supported Pythons:

- 3.7 64-bit

## Install

You can get miniconda from https://docs.conda.io/en/latest/miniconda.html if you don't have it, or install the
dependencies from [`environment.yml`](environment.yml) manually.

```
git clone https://github.com/dachr8/phasic-policy-gradient.git
conda env update --name ppg --file phasic-policy-gradient/environment.yml
conda activate ppg
pip install -e phasic-policy-gradient
```

## Reproduce

PPG with default hyperparameters:

```
nohup mpiexec -np 4 python -m phasic_policy_gradient.train > /tmp/ppg.out &
```

PPO baseline:

```
nohup mpiexec -np 4 python -m phasic_policy_gradient.train --n_epoch_pi 3 --n_epoch_vf 3 --n_aux_epochs 0 --arch shared  --log_dir '/tmp/ppo' > /tmp/ppo.out &
```

PPG, using L_KL instead of L_clip:

```
nohup mpiexec -np 4 python -m phasic_policy_gradient.train --clip_param 0 --kl_penalty 1  --log_dir '/tmp/ppgkl' > /tmp/ppgkl.out &
```

PPG, single network variant:

```
nohup mpiexec -np 4 python -m phasic_policy_gradient.train --arch detach  --log_dir '/tmp/ppg_single_network' > /tmp/ppg_single_network.out &
```

## Visualize

Operating directory: project directory

PPG with default hyperparameters (tmp/ppg-run0):

```
python -m phasic_policy_gradient.graph --experiment_name ppg --save
```

PPO baseline (tmp/ppo-run0):

```
python -m phasic_policy_gradient.graph --experiment_name ppo --save
```

PPG, using L_KL instead of L_clip (tmp/ppgkl-run0):

```
python -m phasic_policy_gradient.graph --experiment_name ppgkl --save
```

PPG, single network variant (tmp/ppgsingle-run0):

```
python -m phasic_policy_gradient.graph --experiment_name ppg_single_network --save
```
