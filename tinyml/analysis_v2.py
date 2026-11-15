"""Verify and aggregate the fold study and generate paper-ready evidence."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .analysis import _finish, _mean_sd, _setup, _table
from .config import CONFIG, DATASETS, ROOT, SEEDS
from .folds import CLIP_DIR, FOLDS, model_dir
from .study_v2 import V2, jobs_for

PAIR = ["dataset", "fold", "seed", "subject", "case_id"]
ANNOT = {"wisdm": "WISDM", "mhealth": "MHEALTH"}
COLORS = {"wisdm": "#2865a3", "mhealth": "#098576"}
PURPLE = "#7655a0"
TEAL = "#0c7d72"
GRAY = "#516377"


def load_archive(root=V2, clips=CLIP_DIR):
    _, wanted = jobs_for(SimpleNamespace(
        cohorts=None, dataset=None, fold=None, seed=None, clips=clips, results=root,
        build=ROOT / "build/v2", reuse=True))
    missing = [x["path"] for x in wanted if not (root / x["path"]).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} benchmark parts are missing, beginning with {missing[0]}")
    out = []
    for item in wanted:
        data = pd.read_csv(root / item["path"])
        if len(data) != item["cases"] * len(item["methods"]) or set(data["mode"]) != set(item["methods"]):
            raise ValueError(f"Incomplete comparison set in {item['path']}")
        if data.duplicated(["case_id", "mode"]).any() or data["mismatches"].ne(0).any():
            raise ValueError(f"Unpaired or incorrect data in {item['path']}")
        counts = data.groupby("case_id")["mode"].nunique()
        if len(counts) != item["cases"] or not counts.eq(len(item["methods"])).all():
            raise ValueError(f"Incomplete or unpaired trials in {item['path']}")
        if not data.groupby("case_id")["checksum"].nunique().eq(1).all():
            raise ValueError(f"Emulated method outputs disagree in {item['path']}")
        out.append(data)
    full = pd.concat(out, ignore_index=True)
    if full.duplicated(["cohort", *PAIR, "mode"]).any():
        raise ValueError("The archive contains repeated experimental keys.")
    if (full[["instructions", "append_instructions", "score_pairs"]] <= 0).any().any():
        raise ValueError("The archive contains an invalid work or correctness count.")
    return full


def compare(archive, cohort, method, reference, reference_cohort=None):
    lhs = archive.loc[archive["cohort"].eq(cohort) & archive["mode"].eq(method)].copy()
    rhs = archive.loc[archive["cohort"].eq(reference_cohort or cohort) & archive["mode"].eq(reference)]
    joined = lhs.merge(rhs[PAIR + ["instructions", "append_instructions", "checksum", "accepted"]],
                       on=PAIR, suffixes=("", "_ref"), validate="one_to_one")
    if joined.empty or not joined["checksum"].eq(joined["checksum_ref"]).all():
        raise ValueError(f"No compatible matched trials for {cohort}/{method} versus {reference_cohort}/{reference}")
    joined = joined.loc[joined["accepted"] & joined["accepted_ref"]].copy()
    joined["saved"] = 100 * (1 - joined["instructions"] / joined["instructions_ref"])
    joined["append_added"] = joined["append_instructions"] - joined["append_instructions_ref"]
    joined["repair_saved"] = joined["instructions_ref"] - joined["instructions"]
    return joined


def strata(frame):
    columns = ["dataset", "fold", "seed"]
    if frame.empty:
        raise ValueError("No paired trials for this statistic")
    return frame.groupby(columns).agg(
        trials=("saved", "size"), median_pct=("saved", "median"),
        p05_pct=("saved", lambda x: x.quantile(.05)),
        win_pct=("saved", lambda x: 100 * x.gt(0).mean()),
        mean_repair_saved=("repair_saved", "mean"), mean_append_added=("append_added", "mean"),
    ).reset_index()


def subject_bootstrap(frame, draws=1600):
    """Resample participants within fold, retaining each sampled participant's paired trials."""
    rng = np.random.default_rng(945711)
    clusters = []
    for _, fold in frame.groupby("fold"):
        clusters.append([subject["saved"].to_numpy() for _, subject in fold.groupby("subject")])
    values = np.empty(draws)
    for draw in range(draws):
        selected = [cluster[i] for cluster in clusters for i in rng.integers(len(cluster), size=len(cluster))]
        values[draw] = np.median(np.concatenate(selected))
    lo, hi = np.quantile(values, (.025, .975))
    return float(lo), float(hi)


def _panels(frame):
    recent = frame["pattern"].eq("single") & frame["delay"].le(7) & frame["placeholder"].eq("hold")
    old = frame["pattern"].eq("single") & frame["delay"].ge(8) & frame["placeholder"].eq("hold")
    return {"All scenarios": frame, "Single, ages 0-7": frame[recent], "Single, ages 8-48": frame[old],
            "Contiguous bursts": frame[frame["pattern"].str.startswith("burst")],
            "Scattered bursts": frame[frame["pattern"].str.startswith("scatter")]}


def primary_table(frame, folder):
    rows = []
    comparisons = ("replay", "layer_replay", "tap_equal", "v1_adaptive", "age_equal", "recent_replay")
    for reference in comparisons:
        select = compare(frame, "main", "accumulator", reference)
        for label, group in _panels(select).items():
            for dataset, sample in group.groupby("dataset", sort=False):
                blocked = strata(sample)
                low, high = subject_bootstrap(sample)
                rows.append({
                    "dataset": dataset, "scenario": label, "reference": reference, "pairs": len(sample),
                    "clips": sample.clip_id.nunique(), "subjects": sample.subject.nunique(),
                    "mean_fold_seed_median_pct": blocked.median_pct.mean(),
                    "sd_fold_seed_median_pp": blocked.median_pct.std(),
                    "min_fold_seed_median_pct": blocked.median_pct.min(),
                    "max_fold_seed_median_pct": blocked.median_pct.max(), "mean_win_pct": blocked.win_pct.mean(),
                    "pooled_median_pct": sample.saved.median(), "subject_bootstrap_low_pct": low,
                    "subject_bootstrap_high_pct": high, "max_method_sram": int(sample.method_sram.max()),
                })
    out = pd.DataFrame(rows)
    out.to_csv(folder / "table_01_main.csv", index=False, float_format="%.5f")
    brief = out.loc[out["reference"].isin(("replay", "tap_equal", "recent_replay")) &
                    out["scenario"].isin(("All scenarios", "Single, ages 0-7"))]
    rendered = [[ANNOT[x.dataset], x.scenario, x.reference, f"{x.mean_fold_seed_median_pct:.1f}",
                 f"{x.mean_win_pct:.1f}", f"{x.pairs:,}"] for x in brief.itertuples(index=False)]
    _table(folder / "table_01_main", ["Dataset", "Scenario", "Reference", "Saved (%)", "Faster (%)", "Pairs"],
           rendered, "Matched Unicorn Cortex-M4 repair costs. Saved is the average of nine fold/seed medians. "
                     "Recent cases are last-observation-filled isolated samples.")
    return out


def ablation_table(frame, folder):
    setup = [
        ("Age-indexed masks", "age_equal", "tap_equal", "all", "main"),
        ("Recent single-update entry", "recent_replay", "age_equal", "recent", "ablation"),
        ("Accumulator reuse", "accumulator", "no_delta", "recent", "ablation"),
        ("Recent accumulator dispatch", "accumulator", "generic", "recent", "ablation"),
        ("Full replay for three changed taps", "accumulator", "all_taps", "bursts", "ablation"),
        ("Exact activation equality", "accumulator", "no_equal", "bursts", "ablation"),
        ("Batch application", "recent_replay", "serial", "bursts", "ablation"),
        ("Cached fallback on older samples", "accumulator", "no_delta", "old", "ablation"),
    ]
    rows = []
    for label, method, reference, rule, cohort in setup:
        pairs = compare(frame, cohort, method, reference)
        if rule == "recent":
            pairs = pairs[pairs["pattern"].eq("single") & pairs["delay"].le(7)]
        elif rule == "old":
            pairs = pairs[pairs["pattern"].eq("single") & pairs["delay"].ge(8)]
        elif rule == "bursts":
            pairs = pairs[pairs["batch"].gt(1)]
        table = strata(pairs)
        table["comparison"] = label
        table["candidate"], table["reference"], table["cohort"] = method, reference, cohort
        rows.append(table)
    output = pd.concat(rows, ignore_index=True)
    output.to_csv(folder / "table_02_ablation.csv", index=False, float_format="%.5f")
    results = []
    for method, block in output.groupby("comparison", sort=False):
        line = [method]
        for dataset in DATASETS:
            score = block.loc[block["dataset"].eq(dataset)]
            line.extend([f"{score.median_pct.mean():.2f}", f"{score.win_pct.mean():.1f}"])
        results.append(line)
    _table(folder / "table_02_ablation", ["Change", "WISDM saved (%)", "WISDM wins (%)",
                                         "MHEALTH saved (%)", "MHEALTH wins (%)"], results,
           "Paired implementation ablations on the listed workload subsets. The CSV includes all fold and seed rows.")
    return output


def resource_table(frame, folder):
    pairs = compare(frame, "main", "accumulator", "recent_replay", "no_cache")
    pairs = pairs[pairs["pattern"].eq("single") & pairs["placeholder"].eq("hold") & pairs["delay"].between(1, 7)]
    controlled = compare(frame, "append8", "accumulator", "recent_replay", "append0")
    summaries = []
    for dataset, subset in pairs.groupby("dataset", sort=False):
        fixed = controlled.loc[controlled.dataset.eq(dataset)]
        saved = fixed["repair_saved"].mean()
        extra = fixed["append_added"].mean()
        summaries.append({
            "dataset": dataset, "pairs": len(subset), "controlled_pairs": len(fixed),
            "cached_instructions_per_append": subset.append_instructions.mean(),
            "uncached_instructions_per_append": subset.append_instructions_ref.mean(),
            "optimized_build_append_change_pct": 100 * subset.append_added.mean() / subset.append_instructions_ref.mean(),
            "controlled_append_extra_instructions": extra,
            "controlled_append_overhead_pct": 100 * extra / fixed.append_instructions_ref.mean(),
            "controlled_mean_instructions_saved_per_repair": saved,
            "controlled_break_even_repairs_per_100_appends": max(0.0, 100 * extra / saved) if saved > 0 else np.nan,
            "max_method_sram": int(frame.query("dataset == @dataset and cohort == 'main' and mode == 'accumulator'").method_sram.max()),
            "max_uncached_sram": int(frame.query("dataset == @dataset and cohort == 'no_cache' and mode == 'recent_replay'").method_sram.max()),
        })
    out = pd.DataFrame(summaries)
    out.to_csv(folder / "table_03_cost.csv", index=False, float_format="%.6f")
    md = [[ANNOT[r.dataset], f"{r.optimized_build_append_change_pct:+.2f}",
           f"{r.controlled_append_overhead_pct:+.2f}",
           f"{r.controlled_break_even_repairs_per_100_appends:.2f}", f"{r.max_method_sram:,}", f"{r.max_uncached_sram:,}"]
          for r in out.itertuples(index=False)]
    _table(folder / "table_03_cost", ["Dataset", "Optimized append Δ (%)", "Controlled append Δ (%)",
                                      "Controlled break-even (%)", "Cached (B)", "Uncached (B)"], md,
           "The optimized builds expose the combined effect of the cache and GCC code generation. "
           "The controlled build prevents convolution kernel inlining and cloning. "
           "Its amortized break-even uses the mean extra instructions per append divided by mean savings "
           "per recent repair. SRAM is live state plus the maximum correction stack in the optimized build.")
    return out


def sensitivity_table(frame, folder):
    rows = []
    for cohort in ("zero_cache", "cache4", "cache8", "cache16", "scalar", "m33", "horizon8", "horizon24", "append8"):
        method = "recent_replay" if cohort == "zero_cache" else "accumulator"
        reference = "tap_equal" if cohort == "zero_cache" else "recent_replay"
        compare_rows = compare(frame, cohort, method, reference)
        recent = compare_rows[compare_rows["pattern"].eq("single") & compare_rows["delay"].le(7)]
        for dataset, group in recent.groupby("dataset", sort=False):
            profile = frame.loc[frame["cohort"].eq(cohort) & frame["dataset"].eq(dataset) & frame["mode"].eq(method)]
            rows.append({
                "dataset": dataset, "cohort": cohort, "candidate": method, "reference": reference,
                "folds": group.fold.nunique(), "subjects": group.subject.nunique(), "recent_pairs": len(group),
                "recent_saved_pct": strata(group).median_pct.mean(),
                "coverage_pct": 100 * profile.accepted.mean(),
                "method_sram": int(profile.method_sram.max()), "append_instructions": profile.append_instructions.mean(),
            })
    result = pd.DataFrame(rows)
    result.to_csv(folder / "table_04_sensitivity.csv", index=False, float_format="%.6f")
    if not result.empty:
        display = [[ANNOT[r.dataset], r.cohort, f"{r.recent_saved_pct:.2f}", f"{r.coverage_pct:.1f}", f"{r.method_sram:,}"]
                   for r in result.itertuples(index=False)]
        _table(folder / "table_04_sensitivity", ["Dataset", "Profile", "Recent saved (%)", "Accepted (%)", "SRAM (B)"],
               display, "Sensitivity uses one training seed per fold and at most one clip per participant. "
                        "The zero-cache row compares the cache-free recent path with tap-mask equality.")
    return result


def policy_table(frame, folder):
    # Measured variants cross the same fixture, seed, horizon, and mode.
    baseline = frame.loc[frame.cohort.eq("zero_cache") & frame["mode"].eq("recent_replay")]
    configs = []
    for cohort in ("cache4", "cache8", "cache16"):
        cases = compare(frame, cohort, "accumulator", "recent_replay", "zero_cache")
        cases = cases[cases["pattern"].eq("single") & cases["delay"].between(1, 7)
                      & cases["placeholder"].eq("hold")]
        for dataset, group in cases.groupby("dataset", sort=False):
            memory = frame.loc[frame.cohort.eq(cohort) & frame.dataset.eq(dataset)
                               & frame["mode"].eq("accumulator")].method_sram.max()
            extra = group.append_added.mean()
            saved = group.repair_saved.mean()
            configs.append({
                "dataset": dataset, "profile": cohort, "paired_recent_repairs": len(group),
                "append_change_instructions": extra, "repair_instructions_saved": saved,
                "break_even_repairs_per_100_appends": max(0.0, 100 * extra / saved) if saved > 0 else np.nan,
                "max_method_sram": int(memory),
                "max_cache_free_sram": int(baseline.loc[baseline.dataset.eq(dataset)].method_sram.max()),
            })
    table = pd.DataFrame(configs)
    table.to_csv(folder / "table_08_cache_policy.csv", index=False, float_format="%.6f")
    rows = [[ANNOT[x.dataset], x.profile, f"{x.append_change_instructions:+.1f}",
             f"{x.break_even_repairs_per_100_appends:.2f}", f"{x.max_method_sram:,}"]
            for x in table.itertuples(index=False)]
    _table(folder / "table_08_cache_policy", ["Dataset", "Cache", "Append change (inst.)", "Break-even (%)", "SRAM (B)"],
           rows, "Amortized cost across separately optimized cache builds. Appends include any compiler code-generation differences.")
    return table


def training_table(root, folder):
    runs = []
    for dataset in DATASETS:
        for fold in FOLDS:
            for seed in SEEDS:
                data = json.loads((model_dir(root, dataset, fold, seed) / "metrics.json").read_text())
                runs.append({"dataset": dataset, "fold": fold, "seed": seed,
                             "test_subjects": len(data["subjects"]["test"]), "test_windows": data["windows"]["test"],
                             "fp32_accuracy_pct": 100 * data["fp32"]["accuracy"],
                             "int8_accuracy_pct": 100 * data["int8"]["accuracy"],
                             "int8_macro_f1_pct": 100 * data["int8"]["macro_f1"]})
    table = pd.DataFrame(runs)
    table.to_csv(folder / "table_05_models.csv", index=False, float_format="%.6f")
    render = [[ANNOT[r.dataset], r.fold, r.seed, r.test_subjects, f"{r.int8_accuracy_pct:.1f}", f"{r.int8_macro_f1_pct:.1f}"]
              for r in table.itertuples(index=False)]
    _table(folder / "table_05_models", ["Dataset", "Fold", "Seed", "Test subjects", "Accuracy (%)", "Macro F1 (%)"],
           render, "Clean out-of-fold INT8 activity recognition for each trained model.")
    return table


def native_table(root, folder):
    x = pd.read_csv(root / "streaming_native.csv.gz")
    if x["mismatches"].ne(0).any():
        raise ValueError("Native asynchronous replay contains a mismatch.")
    outcome = x.groupby(["dataset", "workload"]).agg(
        streams=("ticks", "size"), score_pairs=("score_pairs", "sum"),
        accepted=("accepted_transactions", "sum"), expired=("rejected_transactions", "sum"),
        labels_revised=("label_changed_timestamps", "sum"),
    ).reset_index()
    outcome.to_csv(folder / "table_06_streaming.csv", index=False)
    rows = [[ANNOT[r.dataset], r.workload, f"{r.streams:,}", f"{r.accepted:,}", f"{r.expired:,}", f"{r.score_pairs:,}"]
            for r in outcome.itertuples(index=False)]
    _table(folder / "table_06_streaming", ["Dataset", "Profile", "Streams", "Accepted", "Expired", "Scores"],
           rows, "Asynchronous native verification with repeated and overlapping arrivals. All checked INT8 scores matched.")
    return x


def agreement_table(frame, folder):
    methods = ("replay", "tap_equal", "recent_replay", "accumulator")
    core = frame.loc[frame["cohort"].eq("main") & frame["mode"].isin(methods)]
    other = frame.loc[frame["cohort"].eq("renode_check")]
    keys = PAIR + ["mode"]
    joined = other.merge(core[keys + ["instructions", "append_instructions", "state_bytes",
                                      "stack_bytes", "checksum", "mismatches"]],
                         on=keys, validate="one_to_one", suffixes=("_renode", "_unicorn"))
    if len(joined) != len(other):
        raise ValueError("A Renode confirmation row is missing in the Unicorn archive.")
    for key in ("checksum", "state_bytes", "stack_bytes", "mismatches"):
        if not joined[f"{key}_renode"].eq(joined[f"{key}_unicorn"]).all():
            raise ValueError(f"Renode and Unicorn disagree on {key}")
    for key in ("instructions", "append_instructions"):
        joined[f"{key}_relative_difference_pct"] = 100 * (
            joined[f"{key}_unicorn"] / joined[f"{key}_renode"] - 1
        ).abs()
    ranges = joined.groupby(["dataset", "mode"], sort=False).agg(
        matched=("case_id", "size"),
        median_difference_pct=("instructions_relative_difference_pct", "median"),
        largest_difference_pct=("instructions_relative_difference_pct", "max"),
        median_append_difference_pct=("append_instructions_relative_difference_pct", "median"),
    ).reset_index()
    ranges.to_csv(folder / "table_07_backends.csv", index=False, float_format="%.7f")
    view = [[ANNOT[r.dataset], r.mode, f"{r.matched:,}", f"{r.median_difference_pct:.3f}",
             f"{r.largest_difference_pct:.3f}"] for r in ranges.itertuples(index=False)]
    _table(folder / "table_07_backends", ["Dataset", "Method", "Matched", "Median difference (%)", "Max difference (%)"],
           view, "Identical ARM firmware in Unicorn and Renode. All score checks and SRAM watermarks agree. "
                 "Instruction-count differences are relative to Renode. Both experiments retain their original counts.")
    return ranges, joined


def figure_delay(frame, output):
    fig, axes = plt.subplots(2, 2, figsize=(10.1, 6.6), sharex="col", layout="constrained")
    symbols = {"replay": "o", "tap_equal": "s", "accumulator": "D"}
    colors = {"replay": GRAY, "tap_equal": PURPLE, "accumulator": TEAL}
    source = frame.query("cohort == 'main' and pattern == 'single' and placeholder == 'hold'")
    beats = compare(frame, "main", "accumulator", "tap_equal").query("pattern == 'single' and placeholder == 'hold'")
    for index, dataset in enumerate(DATASETS):
        top, bottom = axes[:, index]
        filt = source.loc[source["dataset"].eq(dataset)]
        for mode in symbols:
            stat = filt[filt["mode"].eq(mode)].groupby(["delay", "fold", "seed"]).instructions.median().groupby("delay")
            mean, lower, upper = stat.median() / 1000, stat.min() / 1000, stat.max() / 1000
            name = {"replay": "Temporal replay", "tap_equal": "Tap mask + equality",
                    "accumulator": "Accumulator reuse"}[mode]
            top.fill_between(mean.index, lower, upper, alpha=.14, color=colors[mode])
            top.plot(mean.index, mean, label=name, marker=symbols[mode], color=colors[mode], lw=1.7, ms=4)
        top.set_title(f"{ANNOT[dataset]}  ·  {CONFIG[dataset]['sample_hz']} Hz", fontweight="bold")
        pair = beats[beats["dataset"].eq(dataset)].groupby(["delay", "fold", "seed"]).saved.median().groupby("delay")
        stat = pair.agg(["median", "min", "max"])
        bottom.fill_between(stat.index, stat["min"], stat["max"], color=TEAL, alpha=.17)
        bottom.plot(stat.index, stat["median"], color=TEAL, marker="D", ms=4, lw=2)
        bottom.axhline(0, color=GRAY, lw=.8)
        for ax in (top, bottom):
            ax.axvspan(0, 7.0, color="#d8eee9", alpha=.5, zorder=0)
        bottom.set_xlabel("Sample age at correction (ticks)")
        bottom.set_xticks([0, 4, 7, 12, 24, 32, 48])
    axes[0, 0].set_ylabel("Instructions per repair (thousands)")
    axes[1, 0].set_ylabel("Savings versus tap mask (%)")
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Recent-sample correction on Cortex-M4", fontsize=16, fontweight="bold")
    _finish(fig, output, "fig_01_delay")


def figure_ablation(table, output):
    labels = list(dict.fromkeys(table.comparison))
    position = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9.1, 4.9))
    fig.subplots_adjust(left=.37, bottom=.14, top=.83, right=.98)
    ax.axvline(0, lw=.9, color=GRAY)
    for n, dataset in enumerate(DATASETS):
        data = table[table.dataset.eq(dataset)].groupby("comparison").median_pct.agg(["mean", "min", "max"]).loc[labels]
        err = np.maximum(0, np.vstack([data["mean"] - data["min"], data["max"] - data["mean"]]))
        ax.errorbar(data["mean"], position + (-.15 if n == 0 else .15), xerr=err, capsize=3,
                    fmt="o" if n == 0 else "s", color=COLORS[dataset], label=ANNOT[dataset])
    ax.set_yticks(position, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Paired instructions saved (%)")
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.legend(loc="lower right")
    fig.suptitle("Instruction cost of each implementation change", x=.04, ha="left", fontsize=16, fontweight="bold")
    fig.text(.04, .883, "Markers average the fold/seed medians. Whiskers span all observed fold/seed medians.",
             color=GRAY, fontsize=9)
    _finish(fig, output, "fig_02_ablation")


def figure_resources(resources, sensitivity, output):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), layout="constrained")
    probability = np.linspace(0, 20, 100)
    for dataset in DATASETS:
        info = resources.loc[resources.dataset.eq(dataset)].iloc[0]
        y = probability / 100 * info.controlled_mean_instructions_saved_per_repair - info.controlled_append_extra_instructions
        color = COLORS[dataset]
        axes[0].plot(probability, y, color=color, lw=2, label=ANNOT[dataset])
        if 0 <= info.controlled_break_even_repairs_per_100_appends <= 20:
            axes[0].plot(info.controlled_break_even_repairs_per_100_appends, 0, "o", color=color, ms=5)
        subset = sensitivity.loc[sensitivity.dataset.eq(dataset) & sensitivity.cohort.isin(("cache4", "cache8", "cache16"))]
        axes[1].plot(subset.method_sram / 1024, subset.recent_saved_pct, "o-", color=color, lw=1.8)
        for row in subset.itertuples():
            axes[1].annotate(row.cohort.replace("cache", "") + " frames", (row.method_sram / 1024, row.recent_saved_pct),
                             xytext=(4, -12 if dataset == "wisdm" else 6), textcoords="offset points", fontsize=7, color=color)
    axes[0].axhline(0, color=GRAY, lw=.8)
    axes[0].set(xlabel="Recent repairs per 100 appends", ylabel="Net instructions saved per append",
                title="A  Amortized cost with the controlled kernel")
    axes[1].set(xlabel="Observed SRAM (KiB)", ylabel="Recent repair savings (%)", title="B  Cache-size trade-off")
    axes[0].legend()
    fig.suptitle("The accumulator cache trades SRAM for correction work", fontsize=15, fontweight="bold")
    _finish(fig, output, "fig_03_resources")


def figure_portability(frame, output):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), sharey=True, layout="constrained")
    areas = ("cache8", "scalar", "m33")
    names = ("Cortex-M4 · DSP", "Cortex-M4 · scalar", "Cortex-M33")
    for ax, dataset in zip(axes, DATASETS, strict=True):
        for cohort, name, color, marker in zip(areas, names, (TEAL, GRAY, PURPLE), ("o", "s", "D")):
            x = compare(frame, cohort, "accumulator", "recent_replay")
            x = x.loc[x["dataset"].eq(dataset) & x["pattern"].eq("single") & x["placeholder"].eq("hold")]
            stat = x.groupby(["delay", "fold"]).saved.median().groupby("delay").median()
            ax.plot(stat.index, stat, lw=1.7, marker=marker, color=color, label=name, ms=4)
        ax.set(title=ANNOT[dataset], xlabel="Sample age (ticks)")
        ax.axhline(0, color=GRAY, lw=.8)
    axes[0].set_ylabel("Savings versus recomputation (%)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Recent correction across ARM profiles and kernels", fontsize=15, fontweight="bold")
    _finish(fig, output, "fig_04_portability")


def run(results=V2, output=ROOT / "figures/v2"):
    folder = results / "tables"
    folder.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    all_rows = load_archive(results)
    main = primary_table(all_rows, folder)
    ablations = ablation_table(all_rows, folder)
    resources = resource_table(all_rows, folder)
    sensitivity = sensitivity_table(all_rows, folder)
    policy = policy_table(all_rows, folder)
    training = training_table(results, folder)
    native = native_table(results, folder)
    _setup()
    figure_delay(all_rows, output)
    figure_ablation(ablations, output)
    figure_resources(resources, sensitivity, output)
    figure_portability(all_rows, output)
    _, matched = agreement_table(all_rows, folder)
    core = all_rows.query("cohort == 'main' and mode == 'accumulator'")
    renode = all_rows.query("backend == 'renode'")
    unicorn = all_rows.query("backend == 'unicorn'")
    report = {
        "models": len(training), "folds_per_dataset": 3, "seeds_per_fold": 3,
        "arm_runs": len(all_rows), "arm_score_pairs": int(all_rows.score_pairs.sum()),
        "arm_mismatches": int(all_rows.mismatches.sum()),
        "renode_runs": len(renode), "renode_score_pairs": int(renode.score_pairs.sum()),
        "unicorn_runs": len(unicorn), "unicorn_score_pairs": int(unicorn.score_pairs.sum()),
        "emulator_matched_runs": len(matched),
        "emulator_median_instruction_difference_pct": float(matched.instructions_relative_difference_pct.median()),
        "native_streams": len(native), "native_score_pairs": int(native.score_pairs.sum()),
        "native_mismatches": int(native.mismatches.sum()), "methods": int(all_rows["mode"].nunique()),
        "ablation_comparisons": int(ablations.comparison.nunique()),
        "per_dataset": {},
    }
    for dataset in DATASETS:
        part = core[core.dataset.eq(dataset)]
        stats = main[main.dataset.eq(dataset)].set_index(["reference", "scenario"])
        def saving(reference, cohort):
            return float(stats.loc[(reference, cohort), "mean_fold_seed_median_pct"])
        resource = resources.loc[resources.dataset.eq(dataset)].iloc[0]
        report["per_dataset"][dataset] = {
            "clips": part.clip_id.nunique(), "subjects": part.subject.nunique(),
            "all_vs_temporal_pct": saving("replay", "All scenarios"),
            "all_vs_tap_pct": saving("tap_equal", "All scenarios"),
            "recent_vs_tap_pct": saving("tap_equal", "Single, ages 0-7"),
            "renode_recent_vs_tap_pct": float(strata(compare(all_rows, "renode_check", "accumulator", "tap_equal")
                .query("dataset == @dataset and pattern == 'single' and delay <= 7")).median_pct.mean()),
            "recent_vs_recompute_pct": saving("recent_replay", "Single, ages 0-7"),
            "recent_vs_layer_replay_pct": saving("layer_replay", "Single, ages 0-7"),
            "recent_vs_tap_bootstrap_low_pct": float(stats.loc[("tap_equal", "Single, ages 0-7"), "subject_bootstrap_low_pct"]),
            "recent_vs_tap_bootstrap_high_pct": float(stats.loc[("tap_equal", "Single, ages 0-7"), "subject_bootstrap_high_pct"]),
            "optimized_build_append_change_pct": float(resource.optimized_build_append_change_pct),
            "controlled_append_overhead_pct": float(resource.controlled_append_overhead_pct),
            "controlled_break_even_repairs_per_100_appends": float(resource.controlled_break_even_repairs_per_100_appends),
            "max_cache_sram_bytes": int(resource.max_method_sram),
            "max_no_cache_sram_bytes": int(resource.max_uncached_sram),
            "cache_policy_break_even_pct": {
                row.profile: float(row.break_even_repairs_per_100_appends)
                for row in policy.loc[policy.dataset.eq(dataset)].itertuples(index=False)
            },
            "int8_accuracy_pct": float(training.loc[training.dataset.eq(dataset), "int8_accuracy_pct"].mean()),
        }
    letter = []
    for dataset, item in report["per_dataset"].items():
        one = main[(main.dataset.eq(dataset)) & main.reference.eq("tap_equal")
                   & main.scenario.eq("Single, ages 0-7")].iloc[0]
        letter.append({
            "dataset": ANNOT[dataset], "participants": item["subjects"],
            "recent_savings_pct": item["recent_vs_tap_pct"],
            "recent_seed_fold_sd": one.sd_fold_seed_median_pp,
            "all_savings_pct": item["all_vs_tap_pct"],
            "sram_kib": item["max_cache_sram_bytes"] / 1024,
        })
    table = pd.DataFrame(letter)
    table.to_csv(folder / "table_09_letter.csv", index=False, float_format="%.6f")
    view = [[r.dataset, r.participants, _mean_sd(r.recent_savings_pct, r.recent_seed_fold_sd),
             f"{r.all_savings_pct:.1f}", f"{r.sram_kib:.2f}"] for r in table.itertuples(index=False)]
    _table(folder / "table_09_letter", ["Dataset", "N", "Recent (%)", "All (%)", "SRAM (KiB)"],
           view, "Accumulator correction versus tap-mask equality in the Cortex-M4 fold study. "
                 "Recent uses ages 0-7. The percentage savings use nine fold/seed medians.")
    (results / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=V2)
    p.add_argument("--figures", type=Path, default=ROOT / "figures/v2")
    args = p.parse_args()
    print(json.dumps(run(args.results, args.figures), indent=2))


if __name__ == "__main__":
    main()
