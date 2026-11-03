# Adding a strategy

Everyone on the team is welcome to add their strategies here, including ones that were never submitted.
The aim is simple: anyone in the team (or the course staff) should be able to take one folder and rebuild its
`submission.csv` without asking the author.

## 1. Pick an ID and a folder name

Folder: `strategies/<id>_<short_slug>/`, lowercase, digits and underscores only.

| Member | ID prefix | Example folder |
|---|---|---|
| George | `strat_<N>` | `strategies/strat_38_sla_sd/` |
| Tarun | `tk_<N>` | `strategies/tk_1_<slug>/` |
| Bhagyashree | `bp_<N>` | `strategies/bp_1_<slug>/` |

Pick your own N. Never reuse an ID, even for a failed run. A fixed or changed version is a new ID; write the
old ID as its parent in the README.

All folders stay directly under `strategies/`. Some configs point to another strategy by relative path
(strat_33 loads its teacher from `../strat_25_rm_400/outputs_s0/model.pt`), so do not nest folders.

## 2. What goes in the folder

Required:

- `README.md`: copy [templates/strategy/README.md](templates/strategy/README.md) and fill in every section.
- The model code, the training script and an inference script (checkpoint in, `submission.csv` out).
- One config file with every hyperparameter. No magic numbers hidden in the code.
- `logs/`: the full training log of the run that made the submission.
- `outputs*/submission.csv`: the exact file that was uploaded to Kaggle.
- `outputs*/model.pt`: the checkpoint of the submitted model (state_dict).

Recommended:

- `outputs*/summary.json` with the parameter count, seed, epochs, GPU, training time and local accuracy.
- `outputs*/metrics.jsonl` with one line per epoch.

You do not have to copy our script layout. A notebook export or your own module structure is fine, as long as
the README lists the exact commands that rebuild your `submission.csv` (a) from the checkpoint and (b) from
scratch. If you keep training in a notebook, also add a `.py` export of it so the code can be read in a diff.

Never commit `data/`, `packed/*.npz` or any file with test labels. The Kaggle data may not be redistributed.

## 3. Local validation

Use the shared VAL5K split from `tools/pack_data.py` if you can (train on the other 45,000 images). Then your
local number can be compared with ours directly. If you use another split, say so in the README.

## 4. Checks before you open a pull request

```bash
python tools/count_params.py strategies/<folder>/<model>.py --config strategies/<folder>/<config>.yaml
python tools/validate_submission.py strategies/<folder>/<outputs>/submission.csv
sha256sum strategies/<folder>/<outputs>/submission.csv
```

`count_params.py` expects a `build_model(cfg)` function. If your code is structured differently, print the
total of `sum(p.numel() for p in model.parameters())` in your training log and copy it into the README.

Then go through the checklist in [RULES.md](RULES.md). Every box must be true.

## 5. Record your Kaggle submissions

If the strategy was submitted, add one row per submitted file to [SUBMISSIONS.md](SUBMISSIONS.md):
date and time (UTC), your NYU net ID, the file name on Kaggle, the strategy folder, the SHA-256 of the
file and the public score.

## 6. Git workflow

```bash
git checkout -b <your-name>/<id>_<slug>
git add strategies/<id>_<slug> SUBMISSIONS.md
git commit -m "<id>: add <short description>"
git push -u origin <your-name>/<id>_<slug>
```

Open a pull request into `main`. One other member reads it against the checklist and merges it.
Small fixes to your own README can go straight to `main`.

File size: GitHub refuses files above 100 MB, and the web upload stops at 25 MB per file. A checkpoint of a
5M-parameter model is about 17-20 MB, so it fits.
