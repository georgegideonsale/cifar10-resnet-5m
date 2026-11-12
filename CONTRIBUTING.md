# Adding a strategy

Anyone on the team can add strategies here, and that includes ones that never got submitted. What we want is
for any teammate (or the course staff) to be able to open one folder and rebuild its `submission.csv` without
having to ask the author.

## Naming your folder

Folders go in `strategies/<id>_<short_slug>/`, using only lowercase letters, digits and underscores. Each of us
has an ID prefix:

- George uses `strat_<N>`, e.g. `strategies/strat_38_sla_sd/`
- Tarun uses `tk_<N>`, e.g. `strategies/tk_1_<slug>/`
- Bhagyashree uses `bp_<N>`, e.g. `strategies/bp_1_<slug>/`

You choose your own N. Don't reuse an ID, not even for a failed run. If you fix or change something, that's a
new ID, and the README should name the old ID as its parent.

Keep every folder directly under `strategies/`, no nesting. Some configs reach into another strategy by relative
path (strat_33 loads its teacher from `../strat_25_rm_400/outputs_s0/model.pt`), and nesting would break that.

## What the folder needs

These are required:

- `README.md`: copy [templates/strategy/README.md](templates/strategy/README.md) and fill in every section.
- The model code, the training script and an inference script (checkpoint in, `submission.csv` out).
- One config file holding every hyperparameter, so there are no magic numbers hidden in the code.
- `logs/`, with the full training log of the run that produced the submission.
- `outputs*/submission.csv`, the exact file that went to Kaggle.
- `outputs*/model.pt`, the checkpoint (state_dict) of the submitted model.

And these are nice to have:

- `outputs*/summary.json` with the parameter count, seed, epochs, GPU, training time and local accuracy.
- `outputs*/metrics.jsonl` with one line per epoch.

You don't have to copy our script layout. A notebook export or your own module structure is fine, as long as
the README gives the exact commands that rebuild your `submission.csv` (a) from the checkpoint and (b) from
scratch. If you train in a notebook, please also add a `.py` export so the code shows up readably in a diff.

Never commit `data/`, `packed/*.npz`, or any file that has test labels in it. The Kaggle data may not be
redistributed.

## Local validation

If you can, use the shared VAL5K split from `tools/pack_data.py` and train on the other 45,000 images. That way
your local number lines up directly with ours. If you use a different split, just say so in the README.

## Before you open a pull request

```bash
python tools/count_params.py strategies/<folder>/<model>.py --config strategies/<folder>/<config>.yaml
python tools/validate_submission.py strategies/<folder>/<outputs>/submission.csv
sha256sum strategies/<folder>/<outputs>/submission.csv
```

`count_params.py` looks for a `build_model(cfg)` function. If your code isn't set up that way, print
`sum(p.numel() for p in model.parameters())` in your training log and copy the total into the README.

After that, go through the checklist in [RULES.md](RULES.md). Every box has to be true.

## Logging Kaggle submissions

For a submitted strategy, add one row per uploaded file to [SUBMISSIONS.md](SUBMISSIONS.md). The row has the
date and time (UTC), your NYU net ID, the file name on Kaggle, the strategy folder, the file's SHA-256 and the
public score.

## Git

```bash
git checkout -b <your-name>/<id>_<slug>
git add strategies/<id>_<slug> SUBMISSIONS.md
git commit -m "<id>: add <short description>"
git push -u origin <your-name>/<id>_<slug>
```

Then open a pull request into `main`. Another member checks it against the checklist and merges it. Small fixes
to your own README can go straight to `main`.

On file size: GitHub refuses anything above 100 MB, and the web upload stops at 25 MB per file. A checkpoint
for a 5M-parameter model is about 17-20 MB, so it fits.
