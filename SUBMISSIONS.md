# Kaggle submissions

One row per file uploaded to Kaggle. Add your rows at the bottom of the table, oldest first.
The SHA-256 lets anyone check that the file in the repo is the file that was scored:
`sha256sum strategies/<folder>/<outputs>/submission.csv`.

Public LB = accuracy on the public half of the test set (1,000 images). The private score is shown only after
the deadline.

| # | Date (UTC) | Member | File name on Kaggle | Kaggle ID | Strategy folder and file | SHA-256 | Public LB |
|---:|---|---|---|---:|---|---|---:|
| 1 | 2026-10-31 03:04 | gs4602 | submission_strat_1.csv | 57824717 | [strat_1](strategies/strat_1_wrn16x5_base) `outputs/submission.csv` | `84393e62b486a369f754d8873e4a5a27453921927a5ad3be33a952c398c54cda` | 0.866 |
| 2 | 2026-11-01 02:19 | gs4602 | submission_strat_9.csv | 57857631 | [strat_9](strategies/strat_9_s6_tta6) `outputs/submission.csv` | `16924aaa45512d59b7144b1e023d0abdc9c568cd56f7827fdbbcc79ad6075fa9` | 0.876 |
| 3 | 2026-11-01 06:17 | gs4602 | submission_strat_11.csv | 57862406 | [strat_11](strategies/strat_11_s3_recipe_400ep) `outputs/submission.csv` | `441e8fa46c33065db821f64b3c85af3f45e895bf45201f17f45d6e21125c121b` | 0.882 |
| 4 | 2026-11-02 23:48 | gs4602 | strat_38_s1_agg_flip2.csv | 57924465 | [strat_38](strategies/strat_38_sla_sd) `outputs_s1/submission.csv` | `add91534ef523369ceef92c1288879dde0f0362b9f95496822d172a37c1f4c27` | 0.903 |
| 5 | 2026-11-02 23:51 | gs4602 | 1_strat_25_s0_flip.csv | 57924507 | [strat_25](strategies/strat_25_rm_400) `outputs_s0/submission.csv` | `9dab187cd269e4363c985dad3d9465c4cad40c6d10e7b7379a73b2f803fd4b66` | 0.893 |
| 6 | 2026-11-02 23:52 | gs4602 | 3_strat_26_s0_flip.csv | 57924531 | [strat_26](strategies/strat_26_rm_sam_deep_400) `outputs_s0/submission.csv` | `7b593bc216441d6ea9905a157a034f5c0f8323af6a1638cd3c91699897df7ce4` | 0.898 |
| 7 | 2026-11-02 23:52 | gs4602 | 4_strat_33_s0_flip.csv | 57924537 | [strat_33](strategies/strat_33_rm400_ba) `outputs_s0/submission.csv` | `157105e3dfe0973586b7232fe61f1c583620cf1aae5c8c92e9e35d76811f03bd` | 0.896 |
| 8 | 2026-11-03 21:44 | gs4602 | strat_40_s0_flip2.csv | 57961146 | [strat_40](strategies/strat_40_rm400_sla) `outputs_s0_flip2/submission.csv` | `4c1e8cfc6f75b61d9dff2c12d01c3d2775db156e1b4bc24c3bab7139ed509c4b` | 0.905 |
| 9 | 2026-11-05 22:40 | gs4602 | strat_40_s0_agg_flip2.csv | 58036904 | [strat_40](strategies/strat_40_rm400_sla) `outputs_s0/submission.csv` | `c296223618e79fc32508c76cd60853f1551e24fb5040b8693b30fb8f7cd0ce5c` | 0.905 |
| 10 | 2026-11-05 23:52 | gs4602 | strat_40_s0_agg_flip2.csv | 58038132 | [strat_40](strategies/strat_40_rm400_sla) `outputs_s0/submission.csv` (same file as row 9, uploaded twice by mistake) | `c296223618e79fc32508c76cd60853f1551e24fb5040b8693b30fb8f7cd0ce5c` | 0.905 |
| 11 | 2026-11-07 00:14 | gs4602 | strat_42_s0_fullfit_tta6.csv | 58073314 | [strat_42](strategies/strat_42_rm400_sla_full) `outputs_s0/submission.csv` | `642a6a786d549d0f60be95b658d8cda4677a557af59b9fd66bbba2522561a88c` | 0.910 |
| 12 | 2026-11-07 04:27 | gs4602 | strat_43_s0_cutout_tta6.csv | 58079001 | [strat_43](strategies/strat_43_rm400_sla_cutout) `outputs_s0/submission.csv` | `0933478b055f17f54d425e077ab022cf05dff3f01744da531ce548b0ba7df24c` | 0.897 |
| 13 | 2026-11-07 04:27 | gs4602 | strat_50_cutout_full_tta6.csv | 58079004 | [strat_50](strategies/strat_50_cutout_full) `outputs_s0/submission.csv` | `89e4647dc4998d6fd600671cf54808ba0107650ba8e8819de15168e561786eb0` | 0.894 |
| 14 | 2026-11-07 04:28 | gs4602 | strat_51_wide84_full_tta6.csv | 58079007 | [strat_51](strategies/strat_51_wide84_full) `outputs_s0/submission.csv` | `6826ef666d41cb9baf3251f24bb147b786b5928dde4dccf4e43cc176bfa9da75` | 0.897 |

Rows 1-7 were checked on 2026-11-03 and rows 9-14 on 2026-11-07: each file was downloaded back from Kaggle and
is byte-identical to the file in the repo.

## Final selections

Kaggle ranks each team by the best private score among the (up to 2) selected submissions. Fill this in
before the deadline (2026-11-07 04:59 UTC).

| Selection | Strategy folder | Kaggle ID | Why |
|---|---|---:|---|
| 1 | [strat_42](strategies/strat_42_rm400_sla_full) `outputs_s0/submission.csv` (SHA-256 `642a6a78…`) | 58073314 | Team's best public score (0.910). Fully reproducible from the code, log and checkpoint in this repo. |
| 2 | Bhagyashree Patil (`bap9622`): `submission.csv` on branch `bap9622` ([pull request #1](https://github.com/georgegideonsale/cifar10-resnet-5m/pull/1)), SHA-256 `45b4031d2a4b57f8f76c4f3a5a2886bf3a0619d65995dd29653c7e19b2cc9742` | 57886359 | Second-best public score (0.909), from a different model (PreAct ResNet 4-4-3, 4,706,674 parameters), so the two picks do not depend on one recipe. The file downloaded back from Kaggle is byte-identical to the file on the branch. |

Selections chosen by the team leader on 2026-11-07, before the deadline.
