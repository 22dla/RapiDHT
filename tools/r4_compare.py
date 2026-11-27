#!/usr/bin/env python3
"""Compare two bench_transform JSON files of the R-4 and R-5 tracks and give a verdict.

    python3 tools/r4_compare.py BASE.json NEW.json [--trace BASE_trace.txt NEW_trace.txt]
    python3 tools/r4_compare.py NEW.json                 # one file: only the table against the baselines

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
        if bench.get("error_occurred"):  # e.g. cuFFT at 1024^3: does not fit, no time to compare
            continue
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
# Step 5.3: convolutions, any kernel and even kernels, against cuFFT R2C/C2R.
# R-5 (docs/R5_PLAN.md, step 0.1): integer input. Each family has one or more
# opponents; an opponent given as several families is the strongest of them
# present in the same row (the same "u16/3D/512x512x512" after the family), so
# an integer path is always measured against the fastest cuFFT variant that
# was built -- conversion kernel, legacy callback or LTO callback.
CUFFT_INT = ("cuFFTInt", "cuFFTIntCb", "cuFFTIntLto")
OPPONENTS: dict[str, tuple[tuple[str, ...], ...]] = {
    "FactoredResident": (("cuFFT",),),
    "FactoredResidentSeparable": (("cuFFT_R2C",),),
    "ConvHartley": (("ConvCufft",),),
    "ConvHartleyEven": (("ConvCufftEven",),),
    "IntConvertFactored": (CUFFT_INT,),
    "IntFactored": (CUFFT_INT, ("IntConvertFactored",)),
    # Step 3.2: the integer path with each offset policy of the library.
    "IntFactoredNone": (CUFFT_INT, ("IntFactored",)),
    "IntFactoredEvery512": (CUFFT_INT, ("IntFactored",)),
    "IntFactoredEvery64": (CUFFT_INT, ("IntFactored",)),
    "IntFactoredMean": (CUFFT_INT, ("IntFactored",)),
    # Step 4.3: the inverse into integers.
    "IntInverse": (("cuFFTIntInv", "cuFFTIntInvLto"), ("IntInverseConvert",)),
    "IntInverseConvert": (("cuFFTIntInv", "cuFFTIntInvLto"),),
    # Step 5.1: the end-to-end tasks. Their family is the first two parts of the
    # name (Task<N>/<implementation>), the case the rest (type, sigma, from, extent).
    "Task1/IntFactored": (("Task1/cuFFTLto",), ("Task1/FloatFactored",)),
    "Task2/Hartley": (("Task2/cuFFTLto",), ("Task2/Spatial",)),
    "Task2/HartleySeparable": (("Task2/cuFFTLto",), ("Task2/Hartley",)),
    "Task3/Hartley": (("Task3/cuFFTLto",),),
    "Task3/HartleySeparable": (("Task3/cuFFTLto",), ("Task3/Hartley",)),
}
OPPONENT_LABEL = {
    "cuFFT": "cuFFT + преобразование в Хартли",
    "cuFFT_R2C": "cuFFT R2C без преобразования",
    "ConvCufft": "свёртка cuFFT R2C + C2R",
    "ConvCufftEven": "свёртка cuFFT R2C + C2R, чётное ядро",
    "cuFFTInt": "перевод в float + cuFFT R2C + Хартли",
    "cuFFTIntCb": "cuFFT с колбэком загрузки + Хартли",
    "cuFFTIntLto": "cuFFT с колбэком LTO + Хартли",
    "IntConvertFactored": "перевод в float + наш путь",
    "IntFactored": "наш путь из целых, смещение — половина диапазона",
    "cuFFTIntInv": "cuFFT C2R + перевод в целые",
    "cuFFTIntInvLto": "cuFFT C2R с колбэком сохранения LTO",
    "IntInverseConvert": "наш float-обратный + перевод в целые",
    "Task1/cuFFTLto": "cuFFT с колбэком LTO + Хартли",
    "Task1/FloatFactored": "наш путь на float с хоста (как сегодня)",
    "Task2/cuFFTLto": "фильтр cuFFT: LTO-загрузка, R2C, произведение, C2R, LTO-запись",
    "Task2/Spatial": "пространственный фильтр, три прохода",
    "Task2/Hartley": "наш фильтр на истинном преобразовании",
    "Task3/cuFFTLto": "спектр мощности cuFFT: LTO-загрузка, R2C, |F|²",
    "Task3/Hartley": "наш спектр мощности на истинном преобразовании",
}


def split_family(name: str) -> tuple[str, str]:
    """'IntFactored/u16/3D/…' -> ('IntFactored', 'u16/3D/…'); 'Task1/IntFactored/u16/…' -> ('Task1/IntFactored', 'u16/…')."""
    family, _, case = name.partition("/")
    if family.startswith("Task"):
        implementation, _, case = case.partition("/")
        family = f"{family}/{implementation}"
    return family, case


def against_cufft(rows: dict[str, dict[str, float]], reference: dict[str, dict[str, float]]) -> None:
    known = {**reference, **rows}  # the new file wins where both have the case
    lines = []
    for name, row in rows.items():
        family, case = split_family(name)
        for group in OPPONENTS.get(family, ()):
            present = [opponent for opponent in group if f"{opponent}/{case}" in known]
            if not present:
                continue
            opponent = min(present, key=lambda o: known[f"{o}/{case}"]["median"])
            ratio = row["median"] / known[f"{opponent}/{case}"]["median"]
            word = "быстрее" if ratio < 1.0 else "медленнее"
            strongest = f" (сильнейший из {len(present)})" if len(group) > 1 and len(present) > 1 else ""
            lines.append(
                f"  {name:<48} {fmt(row['median'])}   {ratio:5.2f}×  {word}, чем {OPPONENT_LABEL[opponent]}{strongest}"
            )
    if lines:
        print("\nПротив эталонов (тот же размер, тип входа и точность):")
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


def load_trace(path: str, pick=min) -> dict[str, dict[str, float]]:
    """'n=512' -> kernel -> `pick` (min by default) of ms over the lines.

    The minimum, not the median: with the clock locked (2026-11-19, aa3/aa4)
    the minima of two runs of the same code agree within 0.5 % on every kernel,
    while the median of y at 512^3 still moved by 7 % -- some calls of that
    kernel take up to 20 % longer at random, and that noise only ever adds.

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
    return {key: {k: pick(v) for k, v in kernels.items()} for key, kernels in samples.items()}


# Only the samples taken right after a size ran: the one before it is taken
# with the card idle, at 210 MHz even with the clock locked (2026-11-20 s4d),
# and says nothing about the clock the kernels ran at.
GPU = re.compile(r"^(n=[0-9x]+(?:/[a-z0-9]+)?) # gpu after sm_mhz=(\d+) temp_c=(\d+)")


def load_clocks(path: str) -> dict[str, tuple[int, int, int]]:
    """'n=512' -> (lowest SM clock MHz, highest, highest temperature C) from the
    '# gpu' lines r4_measure.sh writes around each size since 2026-11-19."""
    seen: dict[str, list[tuple[int, int]]] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            found = GPU.match(line)
            if found:
                seen.setdefault(found.group(1), []).append((int(found.group(2)), int(found.group(3))))
    return {k: (min(c for c, _ in v), max(c for c, _ in v), max(t for _, t in v)) for k, v in seen.items()}


def compare_traces(base_path: str, new_path: str) -> None:
    base, new = load_trace(base_path), load_trace(new_path)
    base_med, new_med = load_trace(base_path, statistics.median), load_trace(new_path, statistics.median)
    clocks_base, clocks_new = load_clocks(base_path), load_clocks(new_path)
    print("\nПо ядрам, минимум по вызовам (в скобках медиана), мс:")
    shared = [key for key in new if key in base]
    if not shared:
        print("  у трасс нет общего размера — сравнивать нечего")
    for key in shared:
        print(f"  {key or '(без метки размера)'}")
        for kernel, value in new[key].items():
            if kernel in base[key]:
                old = base[key][kernel]
                medians = f"   (медиана {base_med[key][kernel]:.3f} -> {new_med[key][kernel]:.3f})"
                print(f"    {kernel:<10} {old:8.3f} -> {value:8.3f}  ({(value / old - 1) * 100:+6.1f} %){medians}")
        # Step 3.3 folds the Bracewell pass into the x pass: compare the pair.
        if "bracewell" in base[key] and "bracewell" not in new[key] and "axis-x" in new[key]:
            old = base[key]["axis-x"] + base[key]["bracewell"]
            value = new[key]["axis-x"]
            print(f"    {'x+brace':<10} {old:8.3f} -> {value:8.3f}  ({(value / old - 1) * 100:+6.1f} %)  (x and Bracewell in one pass)")
        # The clock around the size (traces from 2026-11-19 on): a gap of more
        # than 3 % between the runs makes the per-kernel numbers suspect.
        if key in clocks_base and key in clocks_new:
            (lo_b, hi_b, t_b), (lo_n, hi_n, t_n) = clocks_base[key], clocks_new[key]
            warn = ""
            if max(hi_b, hi_n) > 1.03 * min(lo_b, lo_n):
                warn = "  ← частоты разошлись больше чем на 3 %: сравнение по ядрам ненадёжно"
            print(f"    частота SM  {lo_b}-{hi_b} -> {lo_n}-{hi_n} МГц, до {t_b} -> {t_n} °C{warn}")


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
