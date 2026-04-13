# Experiment Section Draft

This document is a handoff draft for writing Section IV, **Experiments**. It summarizes what the current repository can already support, what data are already available, and what still needs to be filled by the paper-writing side.

The draft is intentionally paper-facing. It does not replace `docs/v6.md`, which is a system specification document.

---

## IV. Experiments

### A. Experimental Setup

#### 1. Simulation Environment

The experiments are conducted in a dataset-backed PyBullet simulation environment with tactile rendering. Each episode corresponds to a single-step in-contact grasp pose refinement problem.

For each episode, the environment samples one initial grasp candidate from the dataset. The policy observes the current grasp state and outputs one normalized 6-DoF corrective action. The action is decoded into a 3D translational delta and a 3D rotational delta, then applied to the initial grasp pose. The simulator reconstructs the refined grasp, collects the post-refinement observation, and finally executes a release trial to determine whether the grasp remains successful.

The task is therefore not a long-horizon manipulation policy or a multi-step regrasp planner. It evaluates whether a single corrective pose update can improve grasp robustness under tactile and stability feedback.

The policy observation used by the main Full model contains:

- a visual-tactile latent feature;
- tactile contact semantics, denoted as `t_cover` and `t_edge`.

The contact semantic signals are defined as:

- `t_cover`: the ratio of active tactile contact pixels among valid tactile pixels;
- `t_edge`: the mean normalized distance from active tactile pixels to the tactile sensor boundary.

The reward used by the Full model contains three components:

- a drop/release outcome reward;
- an uncertainty-aware calibrated stability improvement reward;
- a tactile contact semantic reward.

The calibrated stability path follows:

```text
raw stability logit -> online logistic calibration -> calibrated probability -> posterior uncertainty
```

The default Full training setup uses PPO with asynchronous multi-worker rollout collection and validates checkpoints using held-out validation objects. The final formal test uses the selected `best.pt` checkpoint.

TODO for paper side:

- Decide whether to describe the simulator as PyBullet + TACTO-style tactile rendering or only as a PyBullet-based tactile simulation environment.
- Fill exact training hardware if required by the venue.

#### 2. Unseen Formal Test Protocol

The formal evaluation uses unseen held-out objects and does not reuse training objects. The current formal test protocol is implemented by:

```text
scripts/evaluate_best_checkpoints.py
configs/eval/manifest_template.yaml
```

The current formal output root is:

```text
/rl-grasp-refine/outputs/unseen_test_formal
```

Each evaluated method writes:

```text
summary.csv
per_run_summary.csv
per_object_summary.csv
metadata.json
```

The current formal protocol uses:

- 13 unseen test objects: object ids `75` to `87`;
- 3 test seeds: `1001`, `1002`, `1003`;
- 100 target episodes per object;
- one output row per `(method, test_seed, object_id)` in `per_object_summary.csv`;
- one output row per `(method, test_seed)` in `per_run_summary.csv`;
- one experiment-level row per method in `summary.csv`.

The formal evaluation is checkpoint-based. For learned policies, the evaluator restores:

```text
<experiment_dir>/checkpoints/best.pt
<experiment_dir>/configs/
```

For fixed-action baselines, the evaluator still uses the same environment/config/calibration context, but the actor output is ignored.

Policy modes supported by the formal evaluator:

- `learned_best`: deterministic actor output from the restored best checkpoint;
- `zero_action`: normalized zero action `[0, 0, 0, 0, 0, 0]`;
- `random_uniform`: normalized random action sampled from `Uniform(-1, 1)^6`.

TODO for paper side:

- Confirm whether to explicitly state the object id range `75-87`, or simply say 13 unseen objects.
- Confirm whether `100 episodes per object` should be stated in the main paper or only in supplementary details.

#### 3. No-Action-Adjusted Metrics

The main paper should not primarily interpret raw reference-based scores. Instead, all main plots and planned tables use a no-action-adjusted protocol.

The no-action baseline is an evaluation-time retest floor. It executes the same formal evaluation pipeline with zero pose refinement. This baseline estimates policy-independent retest effects under the same live evaluation protocol.

For any metric `M`, the adjusted value for method `m` is:

```text
AdjustedM_m(test_seed, object_id)
=
M_m(test_seed, object_id)
-
M_no-action(test_seed, object_id)
```

The current plot scripts compute this subtraction using `per_object_summary.csv`, paired by:

```text
(test_seed, object_id)
```

The main adjusted metrics are:

- **Success Gain over No-Action**
  ```text
  success_lift_vs_dataset(method) - success_lift_vs_dataset(no-action)
  ```
- **Excess Degradation over No-Action**
  ```text
  positive_drop_rate(method) - positive_drop_rate(no-action)
  ```
- **Excess Recovery over No-Action**
  ```text
  negative_hold_rate(method) - negative_hold_rate(no-action)
  ```
- **Excess T-Cover Delta over No-Action**
  ```text
  t_cover_delta_mean(method) - t_cover_delta_mean(no-action)
  ```
- **Excess T-Edge Delta over No-Action**
  ```text
  t_edge_delta_mean(method) - t_edge_delta_mean(no-action)
  ```
- **Excess Probability Delta over No-Action**
  ```text
  prob_delta_mean(method) - prob_delta_mean(no-action)
  ```

Interpretation:

- adjusted success gain measures the net success improvement beyond the no-action retest floor;
- excess degradation measures additional damage to reference-positive samples beyond no-action;
- excess recovery measures additional recovery of reference-negative samples beyond no-action;
- excess tactile/probability deltas measure mechanism-level changes beyond the no-action retest floor.

Important wording boundary:

```text
These adjusted metrics are object-run-level paired differences, not episode-level live-live causal estimates.
```

Recommended paper wording:

> To factor out policy-independent retest effects, we report no-action-adjusted metrics. For each method, we subtract the no-action result matched by test seed and object id before aggregating over objects.

#### 4. Confidence Intervals

All adjusted plots use object-level paired bootstrap confidence intervals.

The current procedure is:

1. compute method minus no-action at `(test_seed, object_id)` level;
2. average the adjusted values over the three test seeds for each object;
3. obtain one adjusted value per object;
4. bootstrap over objects;
5. report 95% percentile confidence intervals.

Thus, the CI reflects uncertainty across unseen objects. It does not require episode-level pairing.

Recommended paper wording:

> Confidence intervals are computed by paired object-level bootstrap. We first compute no-action-adjusted values matched by test seed and object id, average them over the three test seeds for each object, and bootstrap over objects to obtain 95% confidence intervals.

TODO for paper side:

- Decide whether to mention the bootstrap iteration count `10000`.

#### 5. Compared Methods for Main Results

The main comparison should use the following methods:

| Paper Name | Current Label | Description |
| --- | --- | --- |
| No Action | `no-action` | Zero normalized pose refinement under the same formal retest pipeline. Used as the adjustment baseline. |
| Random | `rand-action` | Uniform random normalized 6D action, `Uniform(-1, 1)^6`. |
| Vanilla | `drop-only-latent-only-128-epi` | Minimal learned policy using latent perception and drop reward only. |
| Full | `full-latefus-128-epi` | Full learned policy with late-fusion policy observation, stability reward, tactile reward, and online calibration. |

In the paper, the primary comparison should emphasize Random, Vanilla, and Full. No Action should be included as the adjustment baseline and may also appear as a zero row in adjusted tables/figures.

TODO for paper side:

- Confirm whether No Action appears as a method row in Table 1 or is only described as the baseline. Current plots include it with adjusted value zero.

---

### B. Main Results

#### 1. Table 1: Main Quantitative Comparison

Planned table:

```text
Table 1. Main no-action-adjusted quantitative comparison.
```

Recommended columns:

- Method;
- Success Gain over No-Action;
- Excess Degradation over No-Action;
- Excess Recovery over No-Action;
- optionally observed reference-based lift/degradation/recovery in secondary columns.

Current data source:

```text
/rl-grasp-refine/outputs/unseen_test_formal/<label>/per_object_summary.csv
```

Current plot/data extraction support:

```bash
./plot_scripts/plot_group.sh main --print-data-format csv
```

This writes:

```text
plot_scripts/generated/main/plot_data_main.txt
```

TODO for paper side:

- Fill Table 1 numerical values from `plot_data_main.txt`.
- Decide whether to include No Action as an explicit zero row.
- Decide decimal precision.

#### 2. Fig. 4: Success Gain over No-Action

Current script:

```text
plot_scripts/fig01_main_overall_performance.py
```

Metric:

```text
Success Gain over No-Action
```

Data column before adjustment:

```text
success_lift_vs_dataset
```

Aggregation:

- method minus no-action at `(test_seed, object_id)`;
- average over 3 test seeds per object;
- mean and 95% object-bootstrap CI over objects.

Recommended caption idea:

> Success gain over the no-action retest baseline on unseen objects. Error bars denote 95% object-level paired bootstrap confidence intervals.

TODO for paper side:

- Assign final figure number and insert generated PNG.

#### 3. Fig. 5: Excess Degradation / Excess Recovery over No-Action

Current script:

```text
plot_scripts/fig02_main_risk_return.py
```

Metrics:

```text
Excess Degradation over No-Action
Excess Recovery over No-Action
```

Data columns before adjustment:

```text
positive_drop_rate
negative_hold_rate
```

Interpretation:

- lower Excess Degradation is better;
- higher Excess Recovery is better;
- zero indicates the no-action retest floor.

Recommended caption idea:

> Risk-return comparison relative to the no-action baseline. Excess degradation measures additional loss on reference-positive grasps, whereas excess recovery measures additional success on reference-negative grasps.

TODO for paper side:

- Decide whether to use grouped bars or scatter as the main risk-return figure.

#### 4. Fig. 6: 3-Seed Training Curves of Full Policy

Planned figure:

```text
3-seed training curves of the Full policy
```

Current status:

- seed 7 Full run exists;
- seed 8 and seed 9 configs have been created:
  - `configs/experiment/exp_debug_stb5x_latefus_128_epi_seed8.yaml`
  - `configs/experiment/exp_debug_stb5x_latefus_128_epi_seed9.yaml`
- seed 8 / seed 9 training completion status is not documented here.

Suggested plotted training metric:

- `validation/outcome/success_lift_vs_dataset`;
- optionally also `outcome/success_lift_vs_dataset`.

Recommended caption idea:

> Three-seed training curves of the Full policy. The curve shows the mean across training seeds, and the shaded region indicates across-seed variability.

TODO for paper side:

- Confirm seed 8 and seed 9 runs are completed.
- Decide whether the shaded region is standard deviation, standard error, or min-max.
- Implement or reuse a training-curve plotting script if not already available.

---

### C. Ablation Study

#### 1. Ablation Variants

The planned ablation group is:

| Paper Name | Current Label | Description |
| --- | --- | --- |
| Vanilla | `drop-only-latent-only-128-epi` | Minimal latent-only policy with drop reward only. |
| w/o Online Calibration | `wo-onl-cal_latefus_128-epi` | Removes online calibration mechanism. |
| w/o Stability Reward | `wo-stb-rwd_latefus_128-epi` | Removes stability reward contribution. |
| w/o Tactile Reward | `wo-tac-rwd_latefus_128-epi` | Removes tactile contact reward while retaining tactile semantic observation unless otherwise specified by config. |
| w/o Tactile Semantics + Reward | `wo-tac-sem-n-rwd_latefus_128-epi` | Removes tactile semantic observation and tactile reward. |
| Full | `full-latefus-128-epi` | Full proposed method. |

Current formal outputs exist for these labels under:

```text
/rl-grasp-refine/outputs/unseen_test_formal
```

Current plot config group:

```text
ablation
```

#### 2. Table 2: Ablation Results

Planned table:

```text
Table 2. Ablation study under the no-action-adjusted protocol.
```

Recommended columns:

- Method;
- Success Gain over No-Action;
- Excess Degradation over No-Action;
- Excess Recovery over No-Action;
- optionally Excess Probability Delta over No-Action;
- optionally Excess T-Cover / T-Edge Delta over No-Action.

Current data extraction command:

```bash
./plot_scripts/plot_group.sh ablation --print-data-format csv
```

This writes:

```text
plot_scripts/generated/ablation/plot_data_ablation.txt
```

TODO for paper side:

- Fill Table 2 values from `plot_data_ablation.txt`.
- Decide whether mechanism metrics stay in Table 2 or move to a separate mechanism figure.

#### 3. Optional Mechanism Figure

Current script:

```text
plot_scripts/fig04_mechanism_triplet.py
```

Metrics:

- Excess T-Cover Delta over No-Action;
- Excess T-Edge Delta over No-Action;
- Excess Probability Delta over No-Action.

Data columns before adjustment:

```text
t_cover_delta_mean
t_edge_delta_mean
prob_delta_mean
```

Recommended interpretation:

> Mechanism metrics are interpreted as excess changes beyond the no-action retest floor, not as raw absolute before-after changes.

Recommended caption idea:

> Mechanism-level ablation results. Tactile and calibrated stability signals are reported as excess changes over the no-action retest baseline. Error bars indicate 95% object-level paired bootstrap confidence intervals.

TODO for paper side:

- Decide whether this figure is included in the main paper or supplementary material.

---

### D. Complexity Analysis

#### 1. Table 3: Training and Inference Complexity

Planned table:

```text
Table 3. Training and inference complexity.
```

Possible columns:

- Method;
- policy observation components;
- reward components;
- training episodes per iteration;
- number of PPO workers;
- training wall-clock time;
- validation wall-clock time;
- formal evaluation wall-clock time;
- inference latency per episode or policy forward time;
- model parameter count.

Currently available sources:

- training logs: `outputs/exp_debug/<experiment>/run.log`;
- training metrics: `outputs/exp_debug/<experiment>/metrics.jsonl`;
- formal metadata: `outputs/unseen_test_formal/<label>/metadata.json`;
- formal summaries: `outputs/unseen_test_formal/<label>/summary.csv`;
- per-attempt policy forward timing is logged in attempt summaries when available;
- aggregate timing fields exist in training logs and formal metadata.

Known available timing-like signals:

- `timing/collect_wall_s`;
- `timing/update_wall_s`;
- `timing/validation_wall_s`;
- `timing/iteration_wall_s`;
- `evaluation_wall_minutes` in formal metadata / summary.

TODO for paper side:

- Decide exact complexity table columns.
- Extract training wall-clock statistics for Full and major baselines.
- Extract or compute model parameter counts.
- Decide whether inference complexity means neural network forward latency only, or full live refinement episode latency.

#### 2. Deployment Feasibility Discussion

Suggested discussion points:

- The learned policy outputs only one 6-DoF corrective action, so decision-time policy inference is lightweight.
- The heavier cost lies in perception, tactile rendering/sensing, and simulation or robot execution, not PPO policy forward itself.
- The method is single-step and therefore avoids long-horizon planning latency.
- Online calibration is low-dimensional and updates only a small logistic calibration state.
- The formal evaluation uses live retest rollouts, but deployment would execute only one refinement action per grasp candidate.

TODO for paper side:

- Add measured inference latency if available.
- Add hardware details.
- Avoid overclaiming real-robot deployment unless validated.

---

## Current Commands

Generate all main figures and their data dump:

```bash
./plot_scripts/plot_group.sh main --print-data-format csv
```

Generate all ablation figures and their data dump:

```bash
./plot_scripts/plot_group.sh ablation --print-data-format csv
```

Outputs:

```text
plot_scripts/generated/main/*.png
plot_scripts/generated/main/plot_data_main.txt
plot_scripts/generated/ablation/*.png
plot_scripts/generated/ablation/plot_data_ablation.txt
```

Run formal evaluation:

```bash
python scripts/evaluate_best_checkpoints.py configs/eval/manifest_template.yaml
```

Train additional Full seeds:

```bash
python scripts/train.py --experiment configs/experiment/exp_debug_stb5x_latefus_128_epi_seed8.yaml
python scripts/train.py --experiment configs/experiment/exp_debug_stb5x_latefus_128_epi_seed9.yaml
```

---

## Handoff Notes

The paper-writing side should treat this document as a structured source of experiment-section content, not as final prose.

Most important choices still needed:

- Whether No Action is shown as an explicit zero row in tables;
- Whether observed reference-based metrics are included as secondary columns;
- Decimal precision and table formatting;
- Whether mechanism figure is main paper or supplement;
- Exact complexity metrics for Table 3;
- Final figure numbering.
