#!/usr/bin/env python3
"""Compare two bench_transform JSON files of the R-4 track and give a verdict.

    python3 tools/r4_compare.py BASE.json NEW.json [--trace BASE_trace.txt NEW_trace.txt]
    python3 tools/r4_compare.py NEW.json                 # one file: only the table against cuFFT

Why this exists: docs/R4_PLAN.md accepts or rejects every experiment by one
rule, and a rule applied by eye gets bent. A change counts only if the median
moves by more than the noise floor: 3 %, or three times the larger coefficient
of variation of the two runs, whichever is greater. The exit status is 0 in
every case that produced a table; the verdict is in the text.

Standard library only, so it runs on any machine that has the JSON.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys

MIN_EFFECT = 0.03  # a change below 3 % is never called a result
CV_FACTOR = 3.0  # ... nor one below three coefficients of variation
MAX_CV = 0.02  # a run noisier than this should be repeated, not interpreted


def load(path: str) -> dict[str, dict[str, float]]:
    """run_name -> {'median': µs, 'cv': fraction}; plain iterations stand in for a missing median."""
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as error:
        sys.exit(f"{path}: not valid JSON ({error}). A truncated file usually means the run crashed; do not use it.")
    rows: dict[str, dict[str, float]] = {}
    plain: dict[str, list[float]] = {}
    for bench in data.get("benchmarks", []):
        name = bench.get("run_name", bench["name"])
        unit = {"ns": 1e-3, "us": 1.0, "ms": 1e3, "s": 1e6}[bench.get("time_unit", "ns")]
        if bench.get("run_type") == "aggregate":
            if bench.get("aggregate_name") == "median":
                rows.setdefault(name, {})["median"] = bench["real_time"] * unit
            elif bench.get("aggregate_name") == "cv":
                rows.setdefault(name, {})["cv"] = bench["real_time"]  # a ratio, not a time
        else:
            plain.setdefault(name, []).append(bench["real_time"] * unit)
    for name, times in plain.items():
        if "median" not in rows.get(name, {}):
            rows.setdefault(name, {})["median"] = statistics.median(times)
            rows[name].setdefault("cv", statistics.pstdev(times) / statistics.mean(times) if len(times) > 1 else 0.0)
    for row in rows.values():
        row.setdefault("cv", 0.0)
    return {name: row for name, row in rows.items() if "median" in row}


def fmt(us: float) -> str:
    return f"{us / 1e3:9.3f} ms" if us >= 1e3 else f"{us:9.1f} µs"


# Like against like: the true transform against cuFFT plus its conversion pass,
# the separable one (no Bracewell pass) against the bare R2C transform.
OPPONENT = {"FactoredResident": "cuFFT", "FactoredResidentSeparable": "cuFFT_R2C"}
OPPONENT_LABEL = {"cuFFT": "cuFFT + преобразование в Хартли", "cuFFT_R2C": "cuFFT R2C без преобразования"}


def against_cufft(rows: dict[str, dict[str, float]], reference: dict[str, dict[str, float]]) -> None:
    known = {**reference, **rows}  # the new file wins where both have the case
    lines = []
    for name, row in rows.items():
        family, _, case = name.partition("/")
        opponent = OPPONENT.get(family)
        if opponent is None or f"{opponent}/{case}" not in known:
            continue
        ratio = row["median"] / known[f"{opponent}/{case}"]["median"]
        word = "быстрее" if ratio < 1.0 else "медленнее"
        lines.append(f"  {name:<48} {fmt(row['median'])}   {ratio:5.2f}×  {word}, чем {OPPONENT_LABEL[opponent]}")
    if lines:
        print("\nПротив cuFFT (тот же размер и точность):")
        print("\n".join(lines))


def compare(base: dict[str, dict[str, float]], new: dict[str, dict[str, float]]) -> None:
    common = [name for name in new if name in base]
    if not common:
        sys.exit("The two files share no benchmark names; nothing to compare.")
    print(f"{'benchmark':<48} {'база':>12} {'новое':>12} {'изменение':>10}  вердикт")
    noisy = False
    for name in common:
        b, n = base[name], new[name]
        cv = max(b["cv"], n["cv"])
        threshold = max(MIN_EFFECT, CV_FACTOR * cv)
        change = n["median"] / b["median"] - 1.0
        if cv > MAX_CV:
            verdict = f"ШУМ: cv {cv * 100:.1f} % > {MAX_CV * 100:.0f} % — повторить замер"
            noisy = True
        elif change <= -threshold:
            verdict = "быстрее"
        elif change >= threshold:
            verdict = "МЕДЛЕННЕЕ"
        else:
            verdict = f"в пределах шума (порог {threshold * 100:.1f} %)"
        print(f"{name:<48} {fmt(b['median'])} {fmt(n['median'])} {change * 100:+9.1f} %  {verdict}")
    if noisy:
        print("\nЕсть зашумлённые строки: закрыть всё, что использует GPU, и повторить оба прогона.")


TRACE = re.compile(r"(axis-[xyz])\([a-z]+\) ([0-9.]+) ms|(bracewell) ([0-9.]+) ms|(total) ([0-9.]+) ms")


SIZE = re.compile(r"factored\s+(\d+)\^3")


def load_trace(path: str) -> dict[str, dict[str, float]]:
    """'n=512' -> kernel -> median ms over the lines of the file.

    r4_measure.sh prefixes every line with n=<size>. Older traces (one size per
    file, no prefix) get the size from profile_gpu3d's summary line, so both
    kinds compare against each other.
    """
    samples: dict[str, dict[str, list[float]]] = {}
    unlabelled: dict[str, list[float]] = {}
    size = ""
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            found = SIZE.search(line)
            if found and "[RapiDHT trace]" not in line and not line.startswith("n="):
                size = f"n={found.group(1)}"
            if "[RapiDHT trace]" not in line:
                continue
            key = line.split("[RapiDHT trace]")[0].strip()
            for match in TRACE.finditer(line):
                kernel = match.group(1) or match.group(3) or match.group(5)
                value = float(match.group(2) or match.group(4) or match.group(6))
                target = samples.setdefault(key, {}) if key else unlabelled
                target.setdefault(kernel, []).append(value)
    if unlabelled:
        samples.setdefault(size, {}).update(unlabelled)
    return {key: {k: statistics.median(v) for k, v in kernels.items()} for key, kernels in samples.items()}


def compare_traces(base_path: str, new_path: str) -> None:
    base, new = load_trace(base_path), load_trace(new_path)
    print("\nПо ядрам, медиана по строкам трассы, мс:")
    shared = [key for key in new if key in base]
    if not shared:
        print("  у трасс нет общего размера — сравнивать нечего")
    for key in shared:
        print(f"  {key or '(без метки размера)'}")
        for kernel, value in new[key].items():
            if kernel in base[key]:
                old = base[key][kernel]
                print(f"    {kernel:<10} {old:8.3f} -> {value:8.3f}  ({(value / old - 1) * 100:+6.1f} %)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", help="BASE.json NEW.json, or a single NEW.json")
    parser.add_argument("--trace", nargs=2, metavar=("BASE_TRACE", "NEW_TRACE"), help="two RAPIDHT_TRACE_KERNELS logs")
    args = parser.parse_args()
    if len(args.files) > 2:
        parser.error("one or two JSON files")
    new = load(args.files[-1])
    base = load(args.files[0]) if len(args.files) == 2 else {}
    if base:
        compare(base, new)
    against_cufft(new, base)
    if args.trace:
        compare_traces(*args.trace)


if __name__ == "__main__":
    main()
