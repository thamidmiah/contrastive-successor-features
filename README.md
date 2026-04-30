# Skill Discovery on Montezuma's Revenge

Training and evaluation code for METRA, CSF, VISR, and DADS on Atari (Montezuma's Revenge, Room 1).

## Requirements

- Python 3.9+
- See `requirements.txt` for full dependency list

## Setup

Create a new environment and install dependencies:

```bash
conda create -n training-env python=3.9
conda activate training-env
pip install -r requirements.txt
```

Install Atari ROMs (required to run the environments):

```bash
pip install gym[atari,accept-rom-license]
```

Or, if you already have ROM files locally:

```bash
ale-import-roms /path/to/your/roms
```

**Apple Silicon note:** the `requirements.txt` pins `tensorflow-macos`. On Linux/Windows, replace it with `tensorflow`.

## Training

Training scripts live in `scripts/Atari Training Scripts/`. Run the script for the method you want:

```bash
bash "scripts/Atari Training Scripts/run_montezuma_metra.sh"
bash "scripts/Atari Training Scripts/run_montezuma_csf.sh"
bash "scripts/Atari Training Scripts/run_montezuma_visr.sh"
bash "scripts/Atari Training Scripts/run_montezuma_dads.sh"
```

There is also `run_mspacman.sh` for Ms. Pac-Man training.

Each script writes outputs to `exp/<env>/<run_id>/`, including periodic checkpoints.

Once experiments are running, they will be logged under the `exp` folder.

## Evaluation

Use `evaluate_skills.py` to produce plots, skill visualisations, and state coverage results from a trained checkpoint.

The evaluator has two modes:

- **deterministic** — fixed seed, fixed start conditions. Tests whether skills are genuinely distinct when everything is controlled.
- **randomised** — multiple seeds, randomised noops. Tests whether skills stay distinct under perturbation.

### Examples

Run both modes on epoch 500:

```bash
python evaluate_skills.py \
    --exp_dir exp/MontezumaRoom1-v2/sd000_1771641141_montezuma_room1_metra \
    --checkpoint_epoch 500 \
    --episodes_per_option 5
```

Deterministic only:

```bash
python evaluate_skills.py \
    --exp_dir exp/MontezumaRoom1-v2/sd000_1771641141_montezuma_room1_metra \
    --checkpoint_epoch 500 \
    --mode deterministic
```

Compare across multiple checkpoints:

```bash
python evaluate_skills.py \
    --exp_dir exp/MontezumaRoom1-v2/sd000_1771641141_montezuma_room1_metra \
    --checkpoint_epoch 100 200 400 600 \
    --mode deterministic \
    --episodes_per_option 3
```


## Acknowledgements

This code repo was built on the original [CSF repo](https://github.com/Princeton-RL/contrastive-successor-features).

