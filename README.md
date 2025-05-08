# Mastering Expression Adaptation to Different Audiences: Learning Multi-Agent Communication via Contrastive Message Graph

Official repository of C2LMG: Mastering Expression Adaptation to Different Audiences: Learning Multi-Agent Communication via Contrastive Message Graph, based on the open-sourced codebases [PyMARL2](https://github.com/hijkzzz/pymarl2), built on the [PyMARL](https://github.com/oxwhirl/pymarl).

## Installation instructions
```
conda create -n c2lmg python=3.8
conda activate c2lmg
pip install -r requirements.txt
```

Install StartCraft II (2.4.10) and SMAC:
```
bash install_sc2.sh
```

## Run an experiment
### SMAC
To run a [SMAC](https://github.com/oxwhirl/smac) experiment, use the following command format:
```
python3 src/main.py --alg-config=c2lmg --env-config=sc2 with env.args.[env_key1]=[value1] env.args.[env_key2]=[value2] ... --cuda_id=[device_id] --manual_seed=[random_seed]
```
For example, to run a 1o_2r_vs_4r SMAC experiment:
```
python3 src/main.py --alg-config=c2lmg --env-config=sc2 with env.args.map_name=1o_2r_vs_4r --cuda_id=0 --manual_seed=1
```
### MPE
To run a [MPE](https://github.com/openai/multiagent-particle-envs) experiment, including Cooperative Navigation and Predator-Prey, use the following command format:
```
python3 src/main.py --alg-config=c2lmg --env-config=mpe/[env_name] --cuda_id=[device_id] --manual_seed=[random_seed]
```
For example, to run a Cooperative Navigation experiment:
```
python3 src/main.py --alg-config=c2lmg --env-config=mpe/somple_spread_0vis --cuda_id=0 --manual_seed=1
```
To run a Predator-Prey experiment:
```
python3 src/main.py --alg-config=c2lmg --env-config=mpe/simple_tag_0vis_colli --cuda_id=0 --manual_seed=1
```