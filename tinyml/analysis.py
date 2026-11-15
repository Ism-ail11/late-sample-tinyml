"""Aggregate paired observations, make tables, and produce publication figures."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import CONFIG, DATASETS, RESULTS, ROOT, SEEDS, STUDY, run_dir
from .protocol import HORIZON, MODES

COLORS = {
    "replay": "#516176", "span": "#a15c8a", "tap": "#bd6725",
    "tap_equal": "#2459a6", "span_equal": "#7662a8", "adaptive": "#087d74",
    "serial": "#bc4c43", "adaptive_no_equal": "#8e742f",
}
SHORT = {
    "replay": "Replay", "span": "Span", "tap": "Tap mask", "tap_equal": "Tap + equality",
    "span_equal": "Span + equality", "adaptive": "Adaptive + equality",
    "serial": "Serial", "adaptive_no_equal": "Adaptive, no equality",
}
DISPLAY = {dataset: CONFIG[dataset]["display_name"] for dataset in DATASETS}
KEY = ["dataset", "seed", "horizon", "case_id"]
SEED_GROUP = ["dataset", "seed"]


def read_results(results: Path = RESULTS):
    frames = {"native": [], "renode": []}
    training = []
    for dataset in DATASETS:
        for seed in SEEDS:
            directory = run_dir(results, dataset, seed)
            for backend in frames:
                frame = pd.read_csv(directory / f"{backend}_events.csv.gz")
                if set(frame["mode"]) != set(MODES) or frame["mismatches"].sum() != 0:
                    raise ValueError(f"Incomplete or incorrect {backend} data in {directory}")
                if frame.duplicated(KEY + ["mode"]).any():
                    raise ValueError(f"Duplicate trial rows in {directory}/{backend}")
                if frame.groupby(KEY, sort=False).size().ne(len(MODES)).any():
                    raise ValueError(f"Unpaired trial rows in {directory}/{backend}")
                frames[backend].append(frame)
            training.append(json.loads((directory / "metrics.json").read_text(encoding="utf-8")))
    native, renode = (pd.concat(frame, ignore_index=True) for frame in frames.values())
    keys = KEY + ["mode"]
    joint = renode.merge(native[keys + ["revised_timestamps", "status"]],
                         on=keys, how="left", validate="one_to_one", suffixes=("", "_native"))
    if not (joint["revised_timestamps"] == joint["revised_timestamps_native"]).all():
        raise ValueError("Host and Renode disagree about the number of revised timestamps.")
    if not (joint["status"] == joint["status_native"]).all():
        raise ValueError("Host and Renode disagree about the repair status.")
    checksum_spread = renode.groupby(KEY, sort=False)["checksum"].nunique()
    if checksum_spread.max() != 1:
        raise ValueError("Checksum disagreement between Renode modes.")
    return native, renode, training


def principal(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["horizon"].eq(HORIZON) & frame["status"].eq("ok")].copy()


def paired(frame: pd.DataFrame, value: str, candidate: str, reference: str = "replay") -> pd.DataFrame:
    target = frame.loc[frame["mode"].eq(candidate)].copy()
    baseline = frame.loc[frame["mode"].eq(reference), KEY + [value]]
    joined = target.merge(baseline, on=KEY, validate="one_to_one", suffixes=("", "_reference"))
    joined = joined.loc[joined[f"{value}_reference"].gt(0)].copy()
    joined["reduction_pct"] = 100 * (1 - joined[value] / joined[f"{value}_reference"])
    return joined


def _distribution(result: pd.DataFrame):
    return result.groupby(SEED_GROUP, sort=False)["reduction_pct"].agg(
        median="median", tail=lambda x: x.quantile(0.05), win=lambda x: x.gt(0).mean() * 100,
    )


def _table(path: Path, columns: list[str], rows: list[list[str]], caption: str):
    divider = ["---"] + ["---:"] * (len(columns) - 1)
    def line(items):
        return "| " + " | ".join(str(item) for item in items) + " |"

    md = [caption, "", line(columns), line(divider), *(line(row) for row in rows), ""]
    path.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    escaped = {"%": r"\%", "_": r"\_", "&": r"\&", "±": r"$\pm$", "—": "--", "×": r"$\times$", "Δ": r"$\Delta$", "–": "--"}
    def tex(value):
        return "".join(escaped.get(char, char) for char in str(value))

    layout = "l" + "r" * (len(columns) - 1)
    table = [f"\\begin{{tabular}}{{{layout}}}", r"\toprule",
             " & ".join(map(tex, columns)) + r" \\", r"\midrule"]
    table.extend(" & ".join(map(tex, row)) + r" \\" for row in rows)
    table += [r"\bottomrule", r"\end{tabular}", ""]
    path.with_suffix(".tex").write_text("\n".join(table), encoding="utf-8")


def _mean_sd(mean, sd, decimals=1):
    if abs(mean) < 0.5 * 10 ** -decimals:
        mean = 0.0
    return f"{mean:.{decimals}f} ± {sd:.{decimals}f}"


def model_table(training: list[dict], output: Path):
    rows = []
    for model in training:
        rows.append({
            "dataset": model["dataset"], "seed": model["seed"],
            "train_subjects": len(model["subjects"]["train"]),
            "validation_subjects": len(model["subjects"]["validation"]),
            "test_subjects": len(model["subjects"]["test"]),
            "train_windows": model["windows"]["train"], "validation_windows": model["windows"]["validation"],
            "test_windows": model["windows"]["test"], "test_timestamps": model["scored_test_timestamps"],
            "selected_epoch": model["selected_epoch"],
            **{f"{precision}_{metric}_pct": 100 * score
               for precision in ("fp32", "int8") for metric, score in model[precision].items()},
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "table_01_models.csv", index=False, float_format="%.6f")
    display = []
    for dataset in DATASETS:
        part = frame.loc[frame["dataset"].eq(dataset)]
        record = part.iloc[0]
        scores = [_mean_sd(part[key].mean(), part[key].std()) for key in
                  ("fp32_accuracy_pct", "int8_accuracy_pct", "int8_macro_f1_pct")]
        display.append([DISPLAY[dataset], f"{int(record.train_subjects)}/{int(record.validation_subjects)}/{int(record.test_subjects)}",
                        f"{int(record.test_windows):,}", *scores])
    _table(output / "table_01_models",
           ["Dataset", "Train/val/test subjects", "Test windows", "FP32 accuracy (%)",
            "INT8 accuracy (%)", "INT8 macro-F1 (%)"], display,
           "Clean held-out classification. Scores are mean ± sample SD over three training seeds on fixed subject splits.")


def benchmark_table(renode: pd.DataFrame, output: Path) -> pd.DataFrame:
    frame = principal(renode)
    all_seed = []
    display = []
    for mode in MODES:
        compared = paired(frame, "instructions", mode)
        savings = _distribution(compared)
        rates = frame.loc[frame["mode"].eq(mode)].groupby(SEED_GROUP, sort=False).agg(
            cases=("instructions", "size"), median_instructions=("instructions", "median"),
            p95_instructions=("instructions", lambda x: x.quantile(0.95)),
            max_sram_bytes=("application_sram_bytes", "max"), mismatches=("mismatches", "sum"),
        ).join(savings).reset_index()
        rates["mode"] = mode
        rates = rates.rename(columns={"median": "median_savings_pct", "tail": "p05_savings_pct", "win": "win_pct"})
        all_seed.append(rates)
        row = [SHORT[mode]]
        for dataset in DATASETS:
            part = rates.loc[rates["dataset"].eq(dataset)]
            row.extend([
                f"{part['median_instructions'].mean()/1000:.1f}",
                _mean_sd(part["median_savings_pct"].mean(), part["median_savings_pct"].std()),
                f"{part['p95_instructions'].mean()/1000:.1f}", f"{int(part.max_sram_bytes.max()):,}",
            ])
        display.append(row)
    full = pd.concat(all_seed, ignore_index=True)
    full.to_csv(output / "table_02_renode.csv", index=False, float_format="%.6f")
    columns = ["Method"]
    for dataset in DATASETS:
        tag = DISPLAY[dataset]
        columns.extend([f"{tag} median (k)", f"{tag} saved (%)", f"{tag} p95 (k)", f"{tag} SRAM (B)"])
    _table(output / "table_02_renode", columns, display,
           "Cortex-M4 emulation at history 48. Instruction quantiles are computed within each seed and averaged across seeds. "
           "Saved is the mean ± sample SD of seed medians of matched per-case percentage savings versus replay. "
           "SRAM is the maximum observed live method state plus repair stack. All compared scores agreed with the oracle.")
    return full


def ablation_table(renode: pd.DataFrame, native: pd.DataFrame, output: Path) -> pd.DataFrame:
    frame = principal(renode)
    macs = principal(native)
    bursts = frame.loc[frame["burst_size"].gt(1)]
    native_bursts = macs.loc[macs["burst_size"].gt(1)]
    measures = [
        ("Tap dependencies without equality", frame, macs, "tap", "span"),
        ("Tap dependencies with equality", frame, macs, "tap_equal", "span_equal"),
        ("INT8 equality with tap masks", frame, macs, "tap_equal", "tap"),
        ("INT8 equality with dense spans", frame, macs, "span_equal", "span"),
        ("INT8 equality with adaptive masks", frame, macs, "adaptive", "adaptive_no_equal"),
        ("Density switching versus tap masks", frame, macs, "adaptive", "tap_equal"),
        ("Batching on burst arrivals", bursts, native_bursts, "adaptive", "serial"),
    ]
    rows = []
    display = []
    for label, arm, host, candidate, reference in measures:
        measured = _distribution(paired(arm, "instructions", candidate, reference))
        estimated = _distribution(paired(host, "kernel_macs", candidate, reference))
        results = measured.join(estimated, lsuffix="_Arm", rsuffix="_mac").reset_index()
        results.insert(0, "comparison", label)
        results["candidate"] = candidate
        results["reference"] = reference
        rows.append(results)
        row = [label]
        for dataset in DATASETS:
            part = results.loc[results["dataset"].eq(dataset)]
            row.extend([
                _mean_sd(part.median_Arm.mean(), part.median_Arm.std()),
                f"{part.win_Arm.mean():.1f}",
                _mean_sd(part.median_mac.mean(), part.median_mac.std()),
            ])
        display.append(row)
    # Placeholder sensitivity uses the same clip and age at the shared conditions.
    pick = frame.loc[frame["mode"].eq("adaptive") & frame["pattern"].eq("single")]
    idx = ["dataset", "seed", "clip_id", "delay"]
    joined = pick.loc[pick["placeholder"].eq("hold")].merge(
        pick.loc[pick["placeholder"].eq("zero"), idx + ["instructions"]], on=idx,
        suffixes=("", "_reference"), validate="one_to_one",
    )
    joined["reduction_pct"] = 100 * (1 - joined["instructions"] / joined["instructions_reference"])
    results = _distribution(joined).add_suffix("_Arm").reset_index()
    label = "Last observation versus zero placeholder"
    results.insert(0, "comparison", label)
    results["candidate"], results["reference"] = "hold", "zero"
    rows.append(results)
    row = [label]
    for dataset in DATASETS:
        part = results.loc[results["dataset"].eq(dataset)]
        row.extend([_mean_sd(part.median_Arm.mean(), part.median_Arm.std()), f"{part.win_Arm.mean():.1f}", "—"])
    display.append(row)
    full = pd.concat(rows, ignore_index=True)
    full.to_csv(output / "table_03_ablations.csv", index=False, float_format="%.6f")
    _table(output / "table_03_ablations",
           ["Change", "WISDM inst. saved (%)", "WISDM wins (%)", "WISDM MACs saved (%)",
            "MHEALTH inst. saved (%)", "MHEALTH wins (%)", "MHEALTH MACs saved (%)"],
           display, "Paired ablations. Positive savings favor the named change. The instruction and MAC subsets have "
                    "different sizes. The placeholder experiment uses common emulated delays only.")
    return full


def memory_table(renode: pd.DataFrame, native: pd.DataFrame, output: Path) -> pd.DataFrame:
    eligible = native.loc[(native["mode"].eq("adaptive")) & native["pattern"].isin(["single", "burst4"])
                          & native["placeholder"].eq("hold") & native["delay"].isin([8, 24, 48])]
    keys = ["dataset", "horizon"]
    counts = eligible.groupby(keys, sort=False).agg(
        native_cases=("status", "size"), accepted=("status", lambda x: x.eq("ok").sum()),
    )
    arm = renode.loc[renode["mode"].eq("adaptive")].groupby(keys, sort=False).agg(
        ring_bytes=("state_bytes", "max"), stack_bytes=("stack_bytes", "max"),
        application_sram_bytes=("application_sram_bytes", "max"),
    )
    out = arm.join(counts).reset_index().sort_values(keys)
    out["coverage_pct"] = 100 * out.accepted / out.native_cases
    out.to_csv(output / "table_04_memory.csv", index=False, float_format="%.6f")
    display = [[DISPLAY[row.dataset], row.horizon, f"{row.ring_bytes:,}", row.stack_bytes,
                f"{row.application_sram_bytes:,}", f"{row.coverage_pct:.1f}"]
               for row in out.itertuples(index=False)]
    _table(output / "table_04_memory",
           ["Dataset", "Max age", "State (B)", "Stack (B)", "SRAM (B)", "Accepted (%)"], display,
           "Memory ablation. Coverage uses only paired single-sample and four-sample cases with ages 8, 24, and 48 on the native corpus. "
           "Stack is the maximum observed Cortex-M4 watermark. If an update is too old, the entire transaction is rejected.")
    return out


def burst_table(renode: pd.DataFrame, native: pd.DataFrame, output: Path):
    rows = []
    for backend, dataframe, value in (("renode", renode, "instructions"), ("native", native, "kernel_macs")):
        frame = principal(dataframe)
        comparison = paired(frame.loc[frame["burst_size"].gt(1)], value, "adaptive", "serial")
        by_seed = comparison.groupby(["dataset", "seed", "pattern", "burst_size"])["reduction_pct"].agg(
            cases="size", median_savings_pct="median", win_pct=lambda values: 100 * values.gt(0).mean(),
        ).reset_index()
        by_seed["backend"] = backend
        rows.append(by_seed)
    output_frame = pd.concat(rows, ignore_index=True)
    output_frame.to_csv(output / "table_05_bursts.csv", index=False, float_format="%.6f")
    display = []
    step = output_frame.loc[output_frame["backend"].eq("renode")]
    for dataset in DATASETS:
        for size in STUDY["burst_sizes"]:
            part = step.loc[step["dataset"].eq(dataset) & step["burst_size"].eq(size)]
            display.append([DISPLAY[dataset], size, int(part.cases.sum()),
                            _mean_sd(part.median_savings_pct.mean(), part.median_savings_pct.std()),
                            f"{part.win_pct.mean():.1f}"])
    _table(output / "table_05_bursts", ["Dataset", "Burst size", "Cases", "Instructions saved (%)", "Wins (%)"],
           display, "Emulated burst batching versus applying the same updates one at a time.")


def main_table(training: list[dict], measured: pd.DataFrame, output: Path):
    rows = []
    for dataset in DATASETS:
        model = [record["int8"]["accuracy"] for record in training if record["dataset"] == dataset]
        cohort = measured.loc[measured["dataset"].eq(dataset) & measured["mode"].eq("adaptive")]
        rows.append({
            "dataset": DISPLAY[dataset], "cases_per_seed": int(cohort.cases.iloc[0]),
            "median_savings_pct": cohort.median_savings_pct.mean(),
            "savings_sd_pp": cohort.median_savings_pct.std(),
            "faster_pct": cohort.win_pct.mean(), "sram_bytes": int(cohort.max_sram_bytes.max()),
            "int8_accuracy_pct": 100 * np.mean(model),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "table_06_letter.csv", index=False, float_format="%.6f")
    display = [[row.dataset, _mean_sd(row.median_savings_pct, row.savings_sd_pp),
                f"{row.faster_pct:.1f}", f"{row.sram_bytes:,}", f"{row.int8_accuracy_pct:.1f}"]
               for row in frame.itertuples(index=False)]
    _table(output / "table_06_letter",
           ["Dataset", "Saved (%)", "Faster (%)", "SRAM (B)", "Acc. (%)"], display,
           "Proposed correction versus replay at a 48-sample horizon. Saved is the mean ± sample "
           "SD of three within-seed medians. Faster is the share of paired cases with lower instruction count. "
           "Accuracy describes the clean held-out INT8 classifier.")


def _setup():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "text.color": "#132039",
        "axes.labelcolor": "#293b54", "axes.titlecolor": "#132039", "axes.titlesize": 11,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#9ba8bb",
        "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True, "grid.color": "#dce4ed", "grid.linewidth": 0.7,
        "xtick.color": "#334761", "ytick.color": "#334761", "legend.frameon": False,
        "figure.facecolor": "#ffffff", "savefig.facecolor": "#ffffff",
        "svg.fonttype": "none", "pdf.fonttype": 42,
    })


def _finish(fig, output: Path, label: str):
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output / f"{label}.{extension}", dpi=220, bbox_inches="tight", pad_inches=0.19)
    plt.close(fig)


def figure_delay(renode: pd.DataFrame, output: Path):
    frame = principal(renode).query("pattern == 'single' and placeholder == 'hold'")
    modes = ("replay", "span", "tap", "adaptive")
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), sharey=True)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.17, top=0.73, wspace=0.14)
    for panel, (axis, dataset) in enumerate(zip(axes, DATASETS)):
        current = frame.loc[frame["dataset"].eq(dataset)]
        for mode, marker in zip(modes, ("o", "s", "^", "D")):
            seed = current.loc[current["mode"].eq(mode)].groupby(["delay", "seed"])["instructions"].median() / 1000
            stats = seed.groupby(level="delay").agg(["min", "median", "max"])
            axis.fill_between(stats.index, stats["min"], stats["max"], alpha=0.14, color=COLORS[mode], linewidth=0)
            axis.plot(stats.index, stats["median"], color=COLORS[mode], marker=marker, markersize=4,
                      linewidth=1.8, label=SHORT[mode])
        axis.set(title=f"{'AB'[panel]}  {DISPLAY[dataset]}",
                 xlabel=f"Oldest sample age (samples at {CONFIG[dataset]['sample_hz']} Hz)")
        axis.set_xticks([1, 8, 16, 32, 48])
        axis.grid(axis="y")
    axes[0].set_ylabel("Instructions per correction (thousands)")
    fig.suptitle("Correction cost as samples arrive later", x=0.10, y=0.985, ha="left",
                 fontsize=16, fontweight="bold")
    handles, legend = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend, loc="upper left", bbox_to_anchor=(0.082, 0.927),
               ncols=4, fontsize=9, columnspacing=1.5)
    fig.text(0.10, 0.02, "Single late sample, last-observation fill. Bands show the range of per-seed medians.", fontsize=8)
    _finish(fig, output, "fig_01_delay")


def figure_ablation(table: pd.DataFrame, output: Path):
    names = list(dict.fromkeys(table["comparison"]))
    labels = [
        "Tap dependencies · equality off", "Tap dependencies · equality on",
        "INT8 equality · tap mask", "INT8 equality · dense span", "INT8 equality · adaptive",
        "Adaptive · versus tap mask", "Batching · versus serial", "Hold · versus zero placeholder",
    ]
    fig, axis = plt.subplots(figsize=(9.5, 5.1))
    fig.subplots_adjust(left=0.35, right=0.96, bottom=0.16, top=0.78)
    axis.axvline(0, color="#586b82", linewidth=1)
    y = np.arange(len(names))
    for dataset, offset, marker, color in (("wisdm", -0.14, "o", "#2459a6"), ("mhealth", 0.14, "s", "#087d74")):
        stats = table.loc[table["dataset"].eq(dataset)].groupby("comparison")["median_Arm"].agg(["mean", "min", "max"]).loc[names]
        error = np.array([stats["mean"] - stats["min"], stats["max"] - stats["mean"]])
        axis.errorbar(stats["mean"], y + offset, xerr=np.maximum(error, 0), label=DISPLAY[dataset],
                      fmt=marker, color=color, markersize=5, capsize=2.5, elinewidth=1.2, linewidth=0)
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlabel("Paired instruction savings (%)")
    axis.grid(axis="y", visible=False)
    axis.grid(True, axis="x")
    handles, names_legend = axis.get_legend_handles_labels()
    fig.legend(handles, names_legend, loc="upper right", bbox_to_anchor=(0.97, 0.84), ncols=2, fontsize=9)
    fig.suptitle("What changes the cost of correction?", x=0.10, y=0.97, ha="left", fontsize=16, fontweight="bold")
    fig.text(0.10, 0.892, "Each point averages three seed medians. Whiskers span the seed medians.",
             fontsize=9, color="#425772")
    fig.text(0.35, 0.045, "Positive values favor the first method in each comparison.", fontsize=8, color="#425772")
    _finish(fig, output, "fig_02_ablations")


def figure_burst_memory(renode: pd.DataFrame, memory: pd.DataFrame, output: Path):
    frame = principal(renode).query("burst_size > 1")
    comparison = paired(frame, "instructions", "adaptive", "serial")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.7))
    fig.subplots_adjust(left=0.09, right=0.89, bottom=0.25, top=0.70, wspace=0.36)
    for dataset, marker, color in (("wisdm", "o", "#2459a6"), ("mhealth", "s", "#087d74")):
        subset = comparison.loc[comparison["dataset"].eq(dataset)]
        grouped = subset.groupby(["burst_size", "seed"])["reduction_pct"].median()
        stats = grouped.groupby(level=0).agg(["median", "min", "max"])
        axes[0].plot(stats.index, stats["median"], color=color, marker=marker,
                     linewidth=1.9, label=DISPLAY[dataset])
        axes[0].fill_between(stats.index, stats["min"], stats["max"], color=color, alpha=0.15, linewidth=0)
        part = memory.loc[memory["dataset"].eq(dataset)].sort_values("horizon")
        axes[1].plot(part["horizon"], part["application_sram_bytes"] / 1024, color=color, marker=marker, linewidth=1.9)
    axes[0].set(title="A  Batch repair versus serial repair", xlabel="Samples in a contiguous burst",
                ylabel="Paired instruction savings (%)", xticks=list(STUDY["burst_sizes"]))
    axes[0].axhline(0, color="#8292a7", linewidth=0.8)
    axes[1].set(title="B  Retention at three SRAM budgets", xlabel="Maximum correctable age (samples)",
                ylabel="Measured method SRAM (KiB)", xticks=[8, 24, 48])
    coverage = axes[1].twinx()
    coverage.grid(False)
    coverage.spines["right"].set_visible(True)
    part = memory.loc[memory["dataset"].eq("wisdm")].sort_values("horizon")
    coverage.plot(part["horizon"], part["coverage_pct"], color="#805d25", marker="^",
                  linestyle="--", linewidth=1.4, label="Accepted")
    coverage.set(ylim=(0, 112), ylabel="Transactions accepted (%)", yticks=[0, 33.3, 66.7, 100])
    handles, labels = axes[0].get_legend_handles_labels()
    extra, extra_labels = coverage.get_legend_handles_labels()
    fig.legend(handles + extra, labels + extra_labels, loc="upper left", bbox_to_anchor=(0.075, 0.86), ncols=3)
    fig.suptitle("Burst batching and the cost of retaining history", x=0.09, y=0.968, ha="left",
                 fontsize=16, fontweight="bold")
    fig.text(0.09, 0.065, "Panel B acceptance uses matched ages 8, 24, and 48 and is identical for both datasets.",
             fontsize=8, color="#425772")
    _finish(fig, output, "fig_03_bursts_memory")


def figure_distribution(renode: pd.DataFrame, output: Path):
    frame = principal(renode)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), sharey=True)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.18, top=0.68, wspace=0.14)
    for panel, (axis, dataset) in enumerate(zip(axes, DATASETS)):
        subset = frame.loc[frame["dataset"].eq(dataset)]
        for mode in ("tap", "tap_equal", "adaptive", "serial"):
            values = np.sort(paired(subset, "instructions", mode)["reduction_pct"].to_numpy())
            line = {"tap": "-", "tap_equal": "--", "adaptive": "-", "serial": "-."}[mode]
            axis.step(values, 100 * np.arange(1, len(values) + 1) / len(values), where="post",
                      color=COLORS[mode], label=SHORT[mode], linewidth=1.7, linestyle=line)
        axis.axvline(0, linestyle=":", color="#516176", linewidth=1)
        axis.set(title=f"{'AB'[panel]}  {DISPLAY[dataset]}", xlabel="Per-case instruction savings versus replay (%)")
    axes[0].set_ylabel("Cumulative share of cases (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.08, 0.895), ncols=4, fontsize=9)
    fig.suptitle("Per-case savings expose the slow tail", x=0.10, y=0.982, ha="left",
                 fontsize=16, fontweight="bold")
    fig.text(0.10, 0.025, "Points left of zero took more instructions than replay. The curves pool the three fixed-split runs.",
             fontsize=8, color="#425772")
    _finish(fig, output, "fig_04_distribution")


def run(results: Path = RESULTS, figures: Path = ROOT / "figures"):
    native, renode, training = read_results(results)
    tables = results / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    model_table(training, tables)
    rule = benchmark_table(renode, tables)
    ablations = ablation_table(renode, native, tables)
    memory = memory_table(renode, native, tables)
    burst_table(renode, native, tables)
    main_table(training, rule, tables)
    _setup()
    figure_delay(renode, figures)
    figure_ablation(ablations, figures)
    figure_burst_memory(renode, memory, figures)
    figure_distribution(renode, figures)
    summary = {
        "training_seeds": list(SEEDS), "datasets": list(DATASETS), "modes": len(MODES),
        "implementation_ablations": 7, "additional_sensitivity_analyses": 2,
        "native_method_runs": len(native), "renode_method_runs": len(renode),
        "native_score_pairs_checked": int(native.score_pairs_checked.sum()),
        "renode_score_pairs_checked": int(renode.score_pairs_checked.sum()),
        "native_mismatches": int(native.mismatches.sum()), "renode_mismatches": int(renode.mismatches.sum()),
        "primary": {},
    }
    for dataset in DATASETS:
        group = rule.loc[rule["dataset"].eq(dataset) & rule["mode"].eq("adaptive")]
        subset = principal(renode).query("dataset == @dataset")
        paired_rows = paired(subset, "instructions", "adaptive")
        summary["primary"][dataset] = {
            "renode_cases_per_seed": int(group.cases.iloc[0]),
            "adaptive_savings_pct_seed_mean": float(group.median_savings_pct.mean()),
            "adaptive_savings_pct_seed_sd": float(group.median_savings_pct.std()),
            "adaptive_win_pct_seed_mean": float(group.win_pct.mean()),
            "slowest_adaptive_savings_pct": float(paired_rows.reduction_pct.min()),
            "max_observed_adaptive_sram_bytes": int(group.max_sram_bytes.max()),
        }
    (results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--figures", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    print(json.dumps(run(args.results, args.figures), indent=2))


if __name__ == "__main__":
    main()
