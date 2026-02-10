"""Command-line entry point for reproducible experiment runs."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from diffusers import DDPMScheduler
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from ..TimeGrad import TimeGrad
from ..basic_models import HistoricalSimulation, VarCov
from ..stat_tests import generate_report
from ..utils import (
    compute_individual_returns,
    compute_portfolio_returns,
    estimate_var_es_torch,
    estimate_var_es_torch_multivariate,
    seed_everything,
)

DIMENSIONS = ("univariate", "multivariate")
METHODS = ("timegrad", "timegrad_tuned", "historical", "variance_covariance")
LEVELS = (0.01, 0.05)
DEFAULT_CONTEXT = 90
DEFAULT_TRAIN_SAMPLES = 3000
DEFAULT_BATCH_SIZE = 128
DEFAULT_N_SAMPLES = 500
DEFAULT_DEVICE = "cpu"
DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_DATA_PATH = Path("data/complete_stocks.csv")
DEFAULT_PORTFOLIO = 10
DEFAULT_SEED = 12
DEFAULT_CUTOFF = "2022-06-01"

BASE_TIMEGRAD = {
    "num_train_timesteps": 30,
    "beta_end": 0.05,
    "num_inference_steps": 30,
    "lr": 1e-3,
    "n_epochs": 50,
}

TUNED_TIMEGRAD: Dict[str, Dict[float, Dict[str, float]]] = {
    "univariate": {
        0.05: {
            "num_train_timesteps": 46,
            "beta_end": 0.35878774019831244,
            "num_inference_steps": 46,
            "lr": 8.985916117829954e-04,
            "n_epochs": 31,
        },
        0.01: {
            "num_train_timesteps": 18,
            "beta_end": 0.2901444414255984,
            "num_inference_steps": 18,
            "lr": 0.001898390960977473,
            "n_epochs": 88,
        },
    },
    "multivariate": {
        0.05: {
            "num_train_timesteps": 12,
            "beta_end": 0.022429104089340533,
            "num_inference_steps": 12,
            "lr": 0.0030588361272846074,
            "n_epochs": 96,
        },
        0.01: {
            "num_train_timesteps": 47,
            "beta_end": 0.13159191527736805,
            "num_inference_steps": 47,
            "lr": 0.003457518172345782,
            "n_epochs": 15,
        },
    },
}


@dataclass
class PortfolioData:
    train_tensor: torch.Tensor
    full_tensor: torch.Tensor
    test_context: torch.Tensor
    test_target_scaled: torch.Tensor
    test_target_real: torch.Tensor
    scaler: Optional[StandardScaler]
    feature_dim: int
    test_size: int
    tickers: List[str]
    description: str


class ExperimentLogger:
    def __init__(self, base_dir: Path, slug: str) -> None:
        self.base_dir = base_dir
        self.slug = slug
        self.log_dir = base_dir / "logs" / slug
        self.ckpt_dir = base_dir / "checkpoints" / slug
        self.result_dir = base_dir / "results" / slug
        for directory in (self.log_dir, self.ckpt_dir, self.result_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def checkpoint_path(self, repeat_idx: int) -> Path:
        return self.ckpt_dir / f"repeat_{repeat_idx}.pt"

    def log_path(self, repeat_idx: int) -> Path:
        return self.log_dir / f"repeat_{repeat_idx}.json"

    def summary_path(self) -> Path:
        return self.result_dir / "summary.json"

    def write_log(self, repeat_idx: int, payload: Dict) -> None:
        with self.log_path(repeat_idx).open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    def write_summary(self, payload: Dict) -> None:
        with self.summary_path().open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)


def alpha_tag(alpha: float) -> str:
    pct = int(round(alpha * 100))
    return f"{pct}pct"


def slugify(dimension: str, method: str, alpha: float) -> str:
    return f"{dimension}_{method}_{alpha_tag(alpha)}"


def load_market_data(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def prepare_returns(df: pd.DataFrame, tickers: Iterable[str]) -> pd.DataFrame:
    df_copy = df.loc[df["Ticker"].isin(tickers)].copy(deep=True)
    df_returns = compute_individual_returns(df_copy)
    df_returns = compute_portfolio_returns(df_returns)
    df_returns = df_returns.sort_values("Date").reset_index(drop=True)
    df_returns = df_returns.iloc[1:].reset_index(drop=True)
    return df_returns


def build_timegrad_loader(series: torch.Tensor, context_size: int, num_samples: int, batch_size: int, seed: int) -> DataLoader:
    idx_space = np.arange(context_size, series.shape[0])
    if idx_space.size == 0:
        raise ValueError("Not enough observations to build training windows.")
    sample_size = min(num_samples, idx_space.size)
    rng = np.random.default_rng(seed)
    selected = rng.choice(idx_space, size=sample_size, replace=False)

    contexts = torch.zeros(sample_size, context_size, series.shape[1])
    targets = torch.zeros(sample_size, 1, series.shape[1])
    for row, idx in enumerate(selected):
        contexts[row] = series[idx - context_size : idx]
        targets[row] = series[idx]
    dataset = TensorDataset(contexts, targets)
    effective_batch = min(batch_size, sample_size)
    return DataLoader(dataset, batch_size=effective_batch, shuffle=False)


def build_test_tensors(full_series: torch.Tensor, start_idx: int, context_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    test_size = full_series.shape[0] - start_idx
    if test_size <= 0:
        raise ValueError("No test observations found. Check the split date.")
    contexts = torch.zeros(test_size, context_size, full_series.shape[1])
    targets = torch.zeros(test_size, 1, full_series.shape[1])
    for i in range(test_size):
        idx = start_idx + i
        contexts[i] = full_series[idx - context_size : idx]
        targets[i] = full_series[idx]
    return contexts, targets


def prepare_portfolio(
    df: pd.DataFrame,
    tickers: List[str],
    dimension: str,
    context_size: int,
    train_cutoff: pd.Timestamp,
) -> PortfolioData:
    returns = prepare_returns(df, tickers)
    mask = returns["Date"] <= train_cutoff
    train_size = int(mask.sum())
    if train_size <= context_size:
        raise ValueError("Training window must be larger than the context size.")

    if dimension == "univariate":
        features = returns[["Return_Target"]].values
        scaler = StandardScaler().fit(features[:train_size])
        train_tensor = torch.tensor(scaler.transform(features[:train_size]), dtype=torch.float32)
        full_tensor = torch.tensor(scaler.transform(features), dtype=torch.float32)
    else:
        return_cols = sorted([
            col
            for col in returns.columns
            if col.startswith("Return_") and col != "Return_Target"
        ])
        features = returns[return_cols].values
        scaler = StandardScaler().fit(features[:train_size])
        train_tensor = torch.tensor(scaler.transform(features[:train_size]), dtype=torch.float32)
        full_tensor = torch.tensor(scaler.transform(features), dtype=torch.float32)

    test_context, test_scaled = build_test_tensors(full_tensor, train_size, context_size)
    test_real = torch.tensor(returns[["Return_Target"]].values[train_size:], dtype=torch.float32)

    desc = " + ".join([f"{1/len(tickers):.2f}*{ticker}" for ticker in tickers])
    return PortfolioData(
        train_tensor=train_tensor,
        full_tensor=full_tensor,
        test_context=test_context,
        test_target_scaled=test_scaled.flatten(),
        test_target_real=test_real.flatten(),
        scaler=scaler,
        feature_dim=train_tensor.shape[1],
        test_size=test_context.shape[0],
        tickers=tickers,
        description=desc,
    )


def select_tickers(df: pd.DataFrame, size: int, seed: int) -> List[str]:
    unique = df["Ticker"].unique()
    if len(unique) < size:
        raise ValueError("Not enough tickers in the dataset to sample the portfolio.")
    rng = np.random.default_rng(seed)
    choice = rng.choice(unique, size=size, replace=False)
    return sorted(choice.tolist())


def timegrad_config(dimension: str, method: str, alpha: float) -> Dict[str, float]:
    if method == "timegrad_tuned":
        return {**BASE_TIMEGRAD, **TUNED_TIMEGRAD[dimension].get(alpha, {})}
    return BASE_TIMEGRAD


def evaluate_timegrad(
    portfolio: PortfolioData,
    dimension: str,
    alpha: float,
    device: str,
    seed: int,
    method: str,
) -> Tuple[Dict[str, float], List[float], TimeGrad]:
    params = timegrad_config(dimension, method, alpha)
    loader = build_timegrad_loader(
        portfolio.train_tensor,
        context_size=DEFAULT_CONTEXT,
        num_samples=DEFAULT_TRAIN_SAMPLES,
        batch_size=DEFAULT_BATCH_SIZE,
        seed=seed,
    )
    scheduler = DDPMScheduler(
        num_train_timesteps=int(params["num_train_timesteps"]),
        beta_end=float(params["beta_end"]),
        clip_sample=False,
    )
    model = TimeGrad(
        target_dim=portfolio.feature_dim if dimension == "multivariate" else 1,
        input_size=portfolio.feature_dim if dimension == "multivariate" else 1,
        scheduler=scheduler,
        hidden_size=50,
        num_layers=2,
        num_inference_steps=int(params["num_inference_steps"]),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(params["lr"]))
    model.to(device)
    losses = model.fit(loader, optimizer, int(params["n_epochs"]), device)

    VaR = torch.zeros(portfolio.test_size)
    ES = torch.zeros(portfolio.test_size)
    for idx in range(portfolio.test_size):
        context = portfolio.test_context[idx : idx + 1]
        if dimension == "univariate":
            var_i, es_i = estimate_var_es_torch(
                model,
                context,
                alpha=alpha,
                n_samples=DEFAULT_N_SAMPLES,
                device=device,
            )
        else:
            corr = torch.corrcoef(torch.squeeze(context).T).to(torch.double)
            var_i, es_i = estimate_var_es_torch_multivariate(
                model,
                context,
                scaler=portfolio.scaler,
                R=corr,
                alpha=alpha,
                n_samples=DEFAULT_N_SAMPLES,
                device=device,
            )
        VaR[idx], ES[idx] = var_i, es_i

    metric_target = (
        portfolio.test_target_scaled if dimension == "univariate" else portfolio.test_target_real
    )
    loss_curve = [float(x) for x in losses]
    metrics = {k: float(v) for k, v in generate_report(metric_target, VaR, ES, alpha).items()}
    return metrics, loss_curve, model


def evaluate_baseline(
    estimator,
    portfolio: PortfolioData,
    dimension: str,
    alpha: float,
) -> Dict[str, float]:
    VaR = torch.zeros(portfolio.test_size)
    ES = torch.zeros(portfolio.test_size)
    for idx in range(portfolio.test_size):
        context = portfolio.test_context[idx : idx + 1]
        kwargs = {}
        if dimension == "multivariate":
            kwargs["scaler"] = portfolio.scaler
        VaR[idx], ES[idx] = estimator.predict(context, **kwargs)
    metric_target = (
        portfolio.test_target_scaled if dimension == "univariate" else portfolio.test_target_real
    )
    return {k: float(v) for k, v in generate_report(metric_target, VaR, ES, alpha).items()}


def run_single_replicate(
    df: pd.DataFrame,
    dimension: str,
    method: str,
    alpha: float,
    device: str,
    repeat_seed: int,
    portfolio_size: int,
    train_cutoff: pd.Timestamp,
) -> Tuple[Dict[str, float], Dict, Optional[Dict[str, torch.Tensor]]]:
    tickers = select_tickers(df, portfolio_size, repeat_seed)
    seed_everything(repeat_seed)
    portfolio = prepare_portfolio(df, tickers, dimension, DEFAULT_CONTEXT, train_cutoff)

    if method in ("timegrad", "timegrad_tuned"):
        metrics, loss_curve, model = evaluate_timegrad(
            portfolio, dimension, alpha, device, repeat_seed, method
        )
        return (
            metrics,
            {
                "tickers": tickers,
                "portfolio": portfolio.description,
                "seed": repeat_seed,
                "metrics": metrics,
                "timegrad": {"loss_curve": loss_curve},
            },
            model.state_dict(),
        )

    if method == "historical":
        estimator = HistoricalSimulation(alpha=alpha)
    else:
        estimator = VarCov(alpha=alpha)

    metrics = evaluate_baseline(estimator, portfolio, dimension, alpha)
    return (
        metrics,
        {
            "tickers": tickers,
            "portfolio": portfolio.description,
            "seed": repeat_seed,
            "metrics": metrics,
        },
        None,
    )


def summarize(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    summary = {}
    for key in metrics_list[0].keys():
        summary[key] = float(np.mean([metrics[key] for metrics in metrics_list]))
    return summary


def run_experiment(
    dimension: str,
    method: str,
    alpha: float,
    *,
    data_path: Path = DEFAULT_DATA_PATH,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    device: str = DEFAULT_DEVICE,
    n_repeats: int = 5,
    portfolio_size: int = DEFAULT_PORTFOLIO,
    seed: int = DEFAULT_SEED,
    train_cutoff: str = DEFAULT_CUTOFF,
) -> Dict[str, float]:
    if dimension not in DIMENSIONS:
        raise ValueError(f"Unsupported dimension: {dimension}")
    if method not in METHODS:
        raise ValueError(f"Unsupported method: {method}")
    if alpha not in LEVELS:
        raise ValueError(f"Unsupported alpha level: {alpha}")

    df = load_market_data(data_path)
    cutoff = pd.Timestamp(train_cutoff)
    slug = slugify(dimension, method, alpha)
    logger = ExperimentLogger(results_dir, slug)

    metrics_per_repeat: List[Dict[str, float]] = []
    for repeat_idx in range(n_repeats):
        repeat_seed = seed + repeat_idx
        metrics, log_payload, state_dict = run_single_replicate(
            df,
            dimension,
            method,
            alpha,
            device,
            repeat_seed,
            portfolio_size,
            cutoff,
        )
        metrics_per_repeat.append(metrics)
        record = {
            **log_payload,
            "repeat": repeat_idx,
            "alpha": alpha,
            "dimension": dimension,
            "method": method,
        }
        if state_dict is not None:
            ckpt_path = logger.checkpoint_path(repeat_idx)
            torch.save(state_dict, ckpt_path)
            record["checkpoint"] = str(ckpt_path)
        logger.write_log(repeat_idx, record)

    summary = summarize(metrics_per_repeat)
    logger.write_summary(
        {
            "dimension": dimension,
            "method": method,
            "alpha": alpha,
            "n_repeats": n_repeats,
            "portfolio_size": portfolio_size,
            "seed": seed,
            "summary_metrics": summary,
        }
    )
    return summary


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run thesis experiments from the CLI.")
    parser.add_argument("--dimension", choices=DIMENSIONS, help="univariate or multivariate")
    parser.add_argument(
        "--method",
        choices=list(METHODS) + ["all"],
        help="Which estimator to run. Use 'all' to sweep every method for the chosen dimension.",
    )
    parser.add_argument(
        "--level",
        type=float,
        choices=LEVELS,
        default=0.05,
        help="VaR level (alpha).",
    )
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Torch device to use.")
    parser.add_argument("--n-repeats", type=int, default=5, help="Number of random portfolios per experiment.")
    parser.add_argument("--portfolio-size", type=int, default=DEFAULT_PORTFOLIO)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-cutoff", default=DEFAULT_CUTOFF, help="Training data end date (YYYY-MM-DD).")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Ignore dimension/method flags and run the full 2x4x2 grid.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    if not args.run_all and args.dimension is None:
        raise SystemExit("Set --dimension or pass --run-all to execute every experiment.")
    if not args.run_all and args.method is None:
        raise SystemExit("Set --method (use 'all' for every estimator) or pass --run-all.")
    combos: List[Tuple[str, str, float]]
    if args.run_all:
        combos = [
            (dimension, method, level)
            for dimension in DIMENSIONS
            for method in METHODS
            for level in LEVELS
        ]
    elif args.method == "all":
        combos = [(args.dimension, method, args.level) for method in METHODS]
    else:
        combos = [(args.dimension, args.method, args.level)]

    for dimension, method, level in combos:
        print(f"Running {dimension} / {method} / alpha={level:.2%}")
        summary = run_experiment(
            dimension,
            method,
            level,
            data_path=args.data_path,
            results_dir=args.results_dir,
            device=args.device,
            n_repeats=args.n_repeats,
            portfolio_size=args.portfolio_size,
            seed=args.seed,
            train_cutoff=args.train_cutoff,
        )
        pretty = ", ".join([f"{k}: {v:.4f}" for k, v in summary.items()])
        print(f"Summary -> {pretty}\n")


if __name__ == "__main__":
    main()
