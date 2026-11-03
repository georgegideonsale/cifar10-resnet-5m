# Kaggle submissions

One row per file uploaded to Kaggle. Add your rows at the bottom of the table, oldest first.
The SHA-256 lets anyone check that the file in the repo is the file that was scored:
`sha256sum strategies/<folder>/<outputs>/submission.csv`.

Public LB = accuracy on the public half of the test set (1,000 images). The private score is shown only after
the deadline.

| # | Date (UTC) | Member | File name on Kaggle | Kaggle ID | Strategy folder and file | SHA-256 | Public LB |
|---:|---|---|---|---:|---|---|---:|
| 1 | 2026-10-31 03:04 | georgesale | submission_strat_1.csv | 57824717 | [strat_1](strategies/strat_1_wrn16x5_base) `outputs/submission.csv` | `84393e62b486a369f754d8873e4a5a27453921927a5ad3be33a952c398c54cda` | 0.866 |
| 2 | 2026-11-01 02:19 | georgesale | submission_strat_9.csv | 57857631 | [strat_9](strategies/strat_9_s6_tta6) `outputs/submission.csv` | `16924aaa45512d59b7144b1e023d0abdc9c568cd56f7827fdbbcc79ad6075fa9` | 0.876 |
| 3 | 2026-11-01 06:17 | georgesale | submission_strat_11.csv | 57862406 | [strat_11](strategies/strat_11_s3_recipe_400ep) `outputs/submission.csv` | `441e8fa46c33065db821f64b3c85af3f45e895bf45201f17f45d6e21125c121b` | 0.882 |
| 4 | 2026-11-02 23:48 | georgesale | strat_38_s1_agg_flip2.csv | 57924465 | [strat_38](strategies/strat_38_sla_sd) `outputs_s1/submission.csv` | `add91534ef523369ceef92c1288879dde0f0362b9f95496822d172a37c1f4c27` | 0.903 |
| 5 | 2026-11-02 23:51 | georgesale | 1_strat_25_s0_flip.csv | 57924507 | [strat_25](strategies/strat_25_rm_400) `outputs_s0/submission.csv` | `9dab187cd269e4363c985dad3d9465c4cad40c6d10e7b7379a73b2f803fd4b66` | 0.893 |
| 6 | 2026-11-02 23:52 | georgesale | 3_strat_26_s0_flip.csv | 57924531 | [strat_26](strategies/strat_26_rm_sam_deep_400) `outputs_s0/submission.csv` | `7b593bc216441d6ea9905a157a034f5c0f8323af6a1638cd3c91699897df7ce4` | 0.898 |
| 7 | 2026-11-02 23:52 | georgesale | 4_strat_33_s0_flip.csv | 57924537 | [strat_33](strategies/strat_33_rm400_ba) `outputs_s0/submission.csv` | `157105e3dfe0973586b7232fe61f1c583620cf1aae5c8c92e9e35d76811f03bd` | 0.896 |

Rows 1-7 were checked on 2026-11-03: each file was downloaded back from Kaggle and is byte-identical to the
file in the repo.

## Final selections

Kaggle ranks each team by the best private score among the (up to 2) selected submissions. Fill this in
before the deadline (2026-11-07 04:59 UTC).

| Selection | Strategy folder | Kaggle ID | Why |
|---|---|---:|---|
| 1 | | | |
| 2 | | | |
