"""Train causal classifiers, export INT8 weights, and evaluate clean held-out inputs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch import nn
from torch.nn import functional as F

from .config import CONFIG, DATASETS, DILATIONS, KERNEL, RECEPTIVE_FIELD, RESULTS, SEEDS, WIDTHS, WINDOW, run_dir
from .data import DEFAULT_DATA, LABELS, SHA256, load_segments, subject_split, training_stats, windows
from .quant import INPUT_SCALE, Int8Model, dense_inference, export_layer, quantize_input

SUPERVISED_TIMESTAMPS = np.arange(RECEPTIVE_FIELD - 1, WINDOW, 8)
BATCH = 128


class TCN(nn.Module):
    def __init__(self, classes: int):
        super().__init__()
        widths = (3, *WIDTHS)
        self.blocks = nn.ModuleList(
            nn.Conv1d(widths[i], widths[i + 1], KERNEL, dilation=dilation)
            for i, dilation in enumerate(DILATIONS)
        )
        self.head = nn.Conv1d(widths[-1], classes, 1)

    def forward(self, x, intermediate=False):
        states = []
        for block in self.blocks:
            left = (block.kernel_size[0] - 1) * block.dilation[0]
            x = F.relu(block(F.pad(x, (left, 0))))
            states.append(x)
        x = self.head(x)
        return (*states, x) if intermediate else x


def _loaders(dataset, data_dir):
    segments = load_segments(dataset, data_dir)
    splits = subject_split(dataset, segments)
    mean, std = training_stats(segments, splits["train"])
    arrays = {
        split: windows(segments, subjects, CONFIG[dataset][f"{split}_cap"], mean, std)
        for split, subjects in splits.items()
    }
    return splits, mean, std, arrays


def _batches(x, y, order=None):
    order = np.arange(len(x)) if order is None else order
    for begin in range(0, len(x), BATCH):
        ids = order[begin:begin + BATCH]
        yield torch.from_numpy(x[ids]).transpose(1, 2).contiguous(), torch.from_numpy(y[ids])


def _validate(model, x, y):
    model.eval()
    predicted = []
    with torch.no_grad():
        for xb, _ in _batches(x, y):
            logits = model(xb)[:, :, SUPERVISED_TIMESTAMPS]
            predicted.append(logits.argmax(dim=1).cpu().numpy())
    predictions = np.concatenate(predicted).ravel()
    truth = np.repeat(y, len(SUPERVISED_TIMESTAMPS))
    return _scores(truth, predictions, model.head.out_channels)


def _scores(truth, predictions, classes):
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_f1": float(f1_score(truth, predictions, labels=np.arange(classes), average="macro", zero_division=0)),
    }


def _calibrate(model, train_x, mean, std):
    indices = np.linspace(0, len(train_x) - 1, min(768, len(train_x)), dtype=int)
    calibration = torch.from_numpy(train_x[indices]).transpose(1, 2).contiguous()
    with torch.no_grad():
        states = model(calibration, intermediate=True)
    scales = [
        max(float(np.quantile(np.abs(state.numpy()[:, :, RECEPTIVE_FIELD - 1:]), 0.999)), 1e-6) / 127
        for state in states
    ]
    layers = tuple(
        export_layer(conv, input_scale, output_scale, index < len(model.blocks))
        for index, (conv, input_scale, output_scale) in enumerate(
            zip((*model.blocks, model.head), (INPUT_SCALE, *scales[:-1]), scales)
        )
    )
    return Int8Model(layers, mean, std)


def train_one(dataset: str, seed: int, data_dir: Path, results: Path, threads: int = 1):
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(seed)
    splits, mean, std, arrays = _loaders(dataset, data_dir)
    train_x, train_y = arrays["train"]
    validation_x, validation_y = arrays["validation"]
    test_x, test_y = arrays["test"]
    model = TCN(len(LABELS[dataset]))
    output = run_dir(results, dataset, seed)
    output.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    epochs = CONFIG[dataset]["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0002)
    count = np.bincount(train_y, minlength=model.head.out_channels)
    class_weight = (count.mean() / count) ** 0.5
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weight, dtype=torch.float32))
    best = (-1.0, -1)
    best_weights = None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for xb, yb in _batches(train_x, train_y, rng.permutation(len(train_x))):
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)[:, :, SUPERVISED_TIMESTAMPS]
            loss = criterion(logits, yb[:, None].expand(-1, len(SUPERVISED_TIMESTAMPS)))
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(xb)
        scheduler.step()
        metrics = _validate(model, validation_x, validation_y)
        history.append({"epoch": epoch, "train_loss": loss_sum / len(train_x), **metrics})
        if metrics["macro_f1"] > best[0]:
            best = (metrics["macro_f1"], epoch)
            best_weights = {name: value.detach().clone() for name, value in model.state_dict().items()}
        if epoch in (1, epochs) or epoch % 5 == 0:
            print(f"{dataset}/{seed} epoch {epoch:>2}/{epochs}: val macro-F1={metrics['macro_f1']:.4f}", flush=True)
    model.load_state_dict(best_weights)
    np.savez_compressed(output / "fp32.npz", **{
        name: value.cpu().numpy() for name, value in best_weights.items()
    })
    model.eval()
    quantized = _calibrate(model, train_x, mean, std)
    quantized.save(output / "int8.npz")
    quant_predictions = []
    for begin in range(0, len(test_x), BATCH):
        x = quantize_input(test_x[begin:begin + BATCH])
        quant_predictions.append(dense_inference(quantized, x)[:, SUPERVISED_TIMESTAMPS].argmax(axis=-1))
    int8 = _scores(np.repeat(test_y, len(SUPERVISED_TIMESTAMPS)),
                   np.concatenate(quant_predictions).ravel(), quantized.classes)
    fp32 = _validate(model, test_x, test_y)
    metadata = {
        "dataset": dataset, "seed": seed, "classes": list(LABELS[dataset]),
        "sample_hz": CONFIG[dataset]["sample_hz"], "archive_sha256": SHA256[dataset],
        "subjects": splits, "windows": {name: len(pair[0]) for name, pair in arrays.items()},
        "scored_test_timestamps": len(test_y) * len(SUPERVISED_TIMESTAMPS), "receptive_field": RECEPTIVE_FIELD,
        "widths": WIDTHS, "dilations": DILATIONS, "selected_epoch": best[1],
        "fp32": fp32, "int8": int8, "history": history, "torch_version": str(torch.__version__),
    }
    (output / "metrics.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Completed {dataset}/{seed}: FP32 {fp32['accuracy']:.3f}, INT8 {int8['accuracy']:.3f}", flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--reuse", action="store_true", help="Skip model exports with completed metrics.")
    args = parser.parse_args()
    jobs = []
    for dataset in (args.dataset,) if args.dataset else DATASETS:
        for seed in (args.seed,) if args.seed is not None else SEEDS:
            directory = run_dir(args.results, dataset, seed)
            if args.reuse and all((directory / name).exists() for name in ("int8.npz", "fp32.npz", "metrics.json")):
                continue
            jobs.append((dataset, seed, args.data_dir, args.results, args.threads))
    if args.workers < 1 or args.threads < 1:
        parser.error("workers and threads must be positive")
    if args.workers == 1:
        for job in jobs:
            train_one(*job)
    elif jobs:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(train_one, *job) for job in jobs]
            for future in futures:
                future.result()


if __name__ == "__main__":
    main()
