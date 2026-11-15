"""Train wider, causal INT8 classifiers on all outer folds and three seeds."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import DATASETS, RESULTS, SEEDS
from .data import DEFAULT_DATA, LABELS, SHA256, load_segments, training_stats, windows
from .folds import FOLDS, model_dir, splits
from .quant import INPUT_SCALE, Int8Model, dense_inference, export_layer, quantize_input
from .train import _batches, _scores

WIDTHS = (16, 24, 24)
SETTINGS = {
    "wisdm": {"dilations": (1, 4, 16), "epochs": 26, "caps": (18, 8, 14)},
    "mhealth": {"dilations": (1, 5, 24), "epochs": 34, "caps": (46, 22, 24)},
}
FOLD_RESULTS = RESULTS / "v2"


class CausalClassifier(nn.Module):
    def __init__(self, classes, dilations):
        super().__init__()
        widths = (3,) + WIDTHS
        self.layers = nn.ModuleList([
            nn.Conv1d(widths[i], WIDTHS[i], 3, dilation=d) for i, d in enumerate(dilations)
        ] + [nn.Conv1d(WIDTHS[-1], classes, 1)])
        self.padding = tuple(2 * d for d in dilations)

    def forward(self, inputs, intermediates=False):
        state = inputs
        hidden = []
        for layer, pad in zip(self.layers[:-1], self.padding, strict=True):
            state = F.relu(layer(F.pad(state, (pad, 0))))
            hidden.append(state)
        scores = self.layers[-1](state)
        return (*hidden, scores) if intermediates else scores


def _evaluate(model, x, y, times):
    model.eval()
    with torch.no_grad():
        pred = np.concatenate([model(xb)[:, :, times].argmax(1).numpy()
                               for xb, _ in _batches(x, y)]).ravel()
    return _scores(np.repeat(y, len(times)), pred, model.layers[-1].out_channels)


def _export(model, x, mean, std, first_valid):
    model.eval()
    indices = np.linspace(0, len(x) - 1, min(1000, len(x)), dtype=int)
    with torch.no_grad():
        values = model(torch.from_numpy(x[indices]).transpose(1, 2).contiguous(), intermediates=True)
    scales = [max(float(np.quantile(np.abs(a[:, :, first_valid:].numpy()), 0.9995)), 1e-6) / 127
              for a in values]
    layers = tuple(export_layer(conv, a, b, i < 3) for i, (conv, a, b) in enumerate(
        zip(model.layers, (INPUT_SCALE, *scales[:-1]), scales, strict=True)))
    return Int8Model(layers, mean, std)


def fit(dataset, fold, seed, data_dir, results):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(seed)
    settings = SETTINGS[dataset]
    dilations = settings["dilations"]
    first_valid = 2 * sum(dilations)
    times = np.arange(first_valid, 96, 6)
    segments = load_segments(dataset, data_dir)
    partition = splits(dataset, segments, fold)
    mean, std = training_stats(segments, partition["train"])
    data = {name: windows(segments, partition[name], cap, mean, std)
            for name, cap in zip(("train", "validation", "test"), settings["caps"], strict=True)}
    x_train, y_train = data["train"]
    v_x, v_y = data["validation"]
    x_test, y_test = data["test"]
    classes = len(LABELS[dataset])
    model = CausalClassifier(classes, dilations)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings["epochs"], eta_min=0.00015)
    bin_count = np.bincount(y_train, minlength=classes)
    loss = nn.CrossEntropyLoss(weight=torch.tensor((bin_count.mean() / bin_count)**0.45,
                                                   dtype=torch.float32), label_smoothing=0.02)
    history = []
    best = -1.0
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        tloss = 0.0
        for xb, yb in _batches(x_train, y_train, rng.permutation(len(x_train))):
            if epoch < settings["epochs"] * 0.8:
                scale = 1.0 + 0.045 * torch.randn((len(xb), 3, 1))
                bias = 0.035 * torch.randn((len(xb), 3, 1))
                xb = (xb * scale + bias).clamp(-6, 6)
            optimizer.zero_grad(set_to_none=True)
            value = loss(model(xb)[:, :, times], yb[:, None].expand(-1, len(times)))
            value.backward()
            optimizer.step()
            tloss += value.item() * len(xb)
        scheduler.step()
        score = _evaluate(model, v_x, v_y, times)
        history.append({"epoch": epoch, "train_loss": tloss / len(x_train), **score})
        if score["macro_f1"] > best:
            best = score["macro_f1"]
            best_epoch = epoch
            selected = {name: value.detach().clone() for name, value in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0 or epoch == settings["epochs"]:
            print(f"{dataset} fold={fold} seed={seed} epoch={epoch}: validation F1={score['macro_f1']:.3f}", flush=True)
    model.load_state_dict(selected)
    fixed = _export(model, x_train, mean, std, first_valid)
    yp = np.concatenate([dense_inference(fixed, quantize_input(x_test[i:i+128]))[:, times].argmax(-1)
                         for i in range(0, len(x_test), 128)]).ravel()
    int8 = _scores(np.repeat(y_test, len(times)), yp, classes)
    fp32 = _evaluate(model, x_test, y_test, times)
    output = model_dir(results, dataset, fold, seed)
    output.mkdir(parents=True, exist_ok=True)
    fixed.save(output / "int8.npz")
    np.savez_compressed(output / "fp32.npz", **{name: v.numpy() for name, v in selected.items()})
    record = {
        "dataset": dataset, "fold": fold, "seed": seed, "subjects": partition,
        "labels": list(LABELS[dataset]), "sha256": SHA256[dataset], "widths": WIDTHS, "dilations": dilations,
        "best_epoch": best_epoch, "windows": {name: len(pair[1]) for name, pair in data.items()},
        "test_predictions": len(y_test) * len(times), "int8": int8, "fp32": fp32, "history": history,
        "torch_version": str(torch.__version__),
    }
    (output / "metrics.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Finished {dataset}/{fold}/{seed} held-out INT8 accuracy {int8['accuracy']:.3f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--results", type=Path, default=FOLD_RESULTS)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--fold", type=int, choices=FOLDS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    jobs = [(dataset, fold, seed, args.data_dir, args.results)
            for dataset in ((args.dataset,) if args.dataset else DATASETS)
            for fold in ((args.fold,) if args.fold is not None else FOLDS)
            for seed in ((args.seed,) if args.seed is not None else SEEDS)
            if not (args.reuse and all((model_dir(args.results, dataset, fold, seed) / f).is_file()
                                       for f in ("int8.npz", "fp32.npz", "metrics.json")))]
    if args.workers == 1:
        for job in jobs:
            fit(*job)
    elif jobs:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for future in [pool.submit(fit, *job) for job in jobs]:
                future.result()


if __name__ == "__main__":
    main()
