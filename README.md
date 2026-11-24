# LogitCast

Code for **Reading the Odds: Efficient Event Forecasting via Logit Readout**.

LogitCast forecasts real-world events by reading a probability for every outcome from the logits of a decision
head, in a single forward pass and without generating text. It starts from
[JEV-9B](https://huggingface.co/autotrust/AutoTrust-Decider-Mini-9B) (a frozen Qwen3.5-9B backbone with a LoRA
adapter and a decision head), and tunes the adapter and head with the cross-entropy of resolved outcomes. A
temperature fitted on validation questions calibrates each setting.

The repository also builds **LogitBench**, the benchmark of the paper: binary and multiple-choice questions from
Polymarket, Kalshi, Manifold, ForecastBench, FutureX, OpenForesight and BTF-3, each frozen at a forecast time `t`
and given only information public at `t`. Market questions are evaluated with recent prices (market-informed) and
without them (market-blind).

## Layout

| Folder | Contents |
|---|---|
| `pipeline/` | Builds every dataset: fetch markets, prices and dated news headlines, apply the leakage rules (`pipeline/candidates.py`), construct the OpenForesight options, verify the result |
| `logitcast/` | The model (`jev.py`), training (`train_forecast.py`), evaluation of binary (`eval_forecast.py`) and multiple-choice (`eval_choice.py`) questions, and the ForecastBench recalibration (`calibrate_unified.py`) |
| `baselines/` | Reasoning baselines served with vLLM (OpenForecaster-8B, Qwen3.5-9B-Think, gpt-oss), price-only baselines and the text regression |
| `rl/` | The controlled comparison of verbalized and logit-read forecasts under the same Brier objective, with and without reasoning |
| `analysis/` | Scoring helpers, including the paired bootstrap over events |
| `scripts/` | Launchers for training LogitCast and running the reasoning baselines |

Every Python file is a module run from the repository root, e.g. `python -m logitcast.train_forecast --help`.
Datasets are written to `data/`, checkpoints to `runs/` and predictions to `results/`. None of these are tracked.

## Setup

```bash
pip install -r requirements.txt
```

The reasoning baselines need vLLM 0.11.0, and `rl/` needs TRL with vLLM. Install these in separate environments,
since their pinned versions of torch differ. Models are downloaded from Hugging Face on first use. JEV-9B is pinned
to the revision used in the paper (`logitcast/jev.py`).

## Building the data

The Polymarket binary set is built in stages:

| Step | Module | Output |
|---|---|---|
| Download every closed market | `pipeline.fetch_markets` | `data/raw/markets/` |
| Select candidates and apply the leakage rules | `pipeline.candidates` | `data/candidates.parquet` |
| Hourly prices up to the forecast time | `pipeline.fetch_prices` | `data/raw/prices.jsonl` |
| Price filters and context text | `pipeline.build_dataset` | `data/forecast.parquet` |
| Dated headlines published before the forecast time | `pipeline.fetch_news`, `pipeline.add_news` | `data/forecast.parquet` |
| Structured contexts | `pipeline.add_json` | `data/forecast.parquet` |
| Checks of the hard rules | `pipeline.verify_dataset` | stdout |

The other sets have their own builders: `build_choice` (Polymarket multiple choice), `build_manifold`,
`build_manifold_choice`, `build_kalshi`, `build_forecastbench`, `build_futurex`, `build_btf3`, and the
OpenForesight builders (`build_openforesight*`, `split_openforesight`, `answer_judge`). The leakage rules are listed
in the `pipeline/candidates.py` docstring.

## Training and evaluating LogitCast

```bash
bash scripts/train_logitcast.sh GPU jev 1     # LogitCast, from JEV-9B
bash scripts/train_logitcast.sh GPU base 1    # the Qwen-3.5-9B-LogitCast control, from the undistilled backbone
```

The script trains one model on all four settings and scores it on every test set, each read through the
temperature of its own setting. Set `PYTHON` to choose the interpreter.

## Baselines

```bash
bash scripts/run_reasoning_baselines.sh GPU openai/gpt-oss-20b 1
python -m baselines.baselines_price      # logistic regression and gradient boosting on prices alone
python -m baselines.text_regression     # logistic regression on question text, with and without prices
```

## Verbalized and logit-read forecasts

`rl/run_brier.sh` tunes the logit-read model with the Brier score, and `rl/run_stated.sh` tunes the verbalized
model with GRPO on the same questions and rewards. `rl/run_reason.sh` and `rl/run_think.sh` repeat the comparison
with a reasoning trace before the forecast.

## Citation

```bibtex
@article{tire2026reading,
  title  = {Reading the Odds: Efficient Event Forecasting via Logit Readout},
  author = {Tire, Kutay and Taga, Ege Onur and Oymak, Samet},
  year   = {2026}
}
```
