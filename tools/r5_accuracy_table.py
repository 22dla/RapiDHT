#!/usr/bin/env python3
"""Turn an accuracy_integer report of the R-5 track into a table and a verdict.

    python3 tools/r5_accuracy_table.py RUN_accuracy.txt [--all]

Why this exists: docs/R5_PLAN.md picks the default offset policy and checks
criteria C3 and C4 by fixed rules, and a rule applied by eye gets bent (the
same reason tools/r4_compare.py exists). The rules, as the plan states them
after step 0.4 (2026-11-25):

  default policy  the cheapest policy, in the order the report prints them,
                  under which the factored path has, on every input of at least
                  512^3 voxels, RMS error <= 0.01 LSB and largest error <= 10
                  float floors (the floor: the largest error of the exact
                  spectrum rounded to float, on the same input);
  C3              under the default policy, on the synthetic and the real
                  inputs: RMS <= 0.01 LSB and largest <= 1.5 x cuFFT's under the
                  same policy;
  S8 (stop)       no policy passes the default-policy rule for either library
                  on the real inputs, or the round trip loses more than 0.01 %
                  of the voxels of a real input;
  C4 (first look) the round trip through the float spectrum restores every
                  voxel.

Reports from step 3.2 on carry a third row a policy, "R-5 kernel": the integers
read by the first pass, the offset resolved by the library on the device. The
rules above are then checked for it too (step 3.2: C3 by the R-5 kernel, not
by the path with a conversion).

Without --all the table shows the inputs of at least 512^3 voxels, the ones
the rules look at. Standard library only.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field

RMS_LIMIT = 0.01  # LSB
FLOORS_LIMIT = 10.0  # largest error, in float floors
CUFFT_LIMIT = 1.5  # largest error against cuFFT's under the same policy
LOSS_LIMIT = 0.01  # %, voxels the round trip may lose on a real input (S8)
LARGE = 512**3  # the rules look at inputs of at least this many voxels

HEADER = re.compile(r"^(\d+)(?:\^3|x(\d+)x(\d+)), (.*?) \(mean")
ROW = re.compile(r"^  (cuFFT \+ conversion|GPUFactored|R-5 kernel), (.*?)\s+rms (\S+)\s+largest (\S+)\s+zero frequency (\S+)")
FLOOR = re.compile(r"^  floor: exact spectrum stored as float\s+rms (\S+)\s+largest (\S+)")
TRIP = re.compile(r"^      round trip: (\d+) of (\d+) voxels not restored \((\S+) %\), largest \|x' - x\| (\S+)")


@dataclass
class Result:
    rms: float
    largest: float
    lost: int = 0
    trip_largest: float = 0.0


@dataclass
class Input:
    name: str
    voxels: int
    floor: float = 0.0
    policies: list[str] = field(default_factory=list)
    # (library, policy) -> Result; library is "ours", "cufft" or "r5"
    results: dict[tuple[str, str], Result] = field(default_factory=dict)

    @property
    def real(self) -> bool:
        return ".raw" in self.name

    @property
    def realistic(self) -> bool:  # what C3 is about: tomograms, not white noise
        return self.real or "synthetic" in self.name


def parse(path: str) -> list[Input]:
    inputs: list[Input] = []
    last: tuple[str, str] | None = None
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if m := HEADER.match(line):
                w = int(m.group(1))
                h = int(m.group(2)) if m.group(2) else w
                d = int(m.group(3)) if m.group(3) else w
                extent = f"{w}^3" if m.group(2) is None else f"{w}x{h}x{d}"
                inputs.append(Input(f"{extent}, {m.group(4)}", w * h * d))
            elif inputs and (m := FLOOR.match(line)):
                inputs[-1].floor = float(m.group(2))
            elif inputs and (m := ROW.match(line)):
                library = {"cuFFT + conversion": "cufft", "GPUFactored": "ours", "R-5 kernel": "r5"}[m.group(1)]
                policy = m.group(2).strip()
                if policy not in inputs[-1].policies:
                    inputs[-1].policies.append(policy)
                inputs[-1].results[(library, policy)] = Result(float(m.group(3)), float(m.group(4)))
                last = (library, policy)
            elif inputs and last and (m := TRIP.match(line)):
                result = inputs[-1].results[last]
                result.lost = int(m.group(1))
                result.trip_largest = float(m.group(4))
    if not inputs:
        sys.exit(f"{path}: no accuracy_integer report found")
    return [i for i in inputs if i.results]


def passes(inp: Input, library: str, policy: str) -> bool:
    r = inp.results.get((library, policy))
    return r is not None and r.rms <= RMS_LIMIT and r.largest <= FLOORS_LIMIT * inp.floor


def table(inputs: list[Input], show_all: bool) -> None:
    for inp in inputs:
        if not show_all and inp.voxels < LARGE:
            continue
        print(f"\n{inp.name}   (float floor: largest {inp.floor:.3g} LSB)")
        print(f"  {'policy':<32} {'ours: largest':>14} {'floors':>7} {'rms':>9} {'cuFFT: largest':>15} {'ours/cuFFT':>10}  round trip lost (ours, cuFFT)")
        for policy in inp.policies:
            ours, cufft = inp.results.get(("ours", policy)), inp.results.get(("cufft", policy))
            if ours is None or cufft is None:
                continue
            mark = "" if passes(inp, "ours", policy) else "  ✗"
            r5 = inp.results.get(("r5", policy))
            r5_text = f"   R-5: {r5.largest:.3g}, rms {r5.rms:.2e}, lost {r5.lost}" if r5 else ""
            print(
                f"  {policy:<32} {ours.largest:14.3g} {ours.largest / inp.floor if inp.floor else float('inf'):7.1f}"
                f" {ours.rms:9.2e} {cufft.largest:15.3g} {ours.largest / cufft.largest if cufft.largest else float('nan'):10.2f}"
                f"  {ours.lost}, {cufft.lost}{mark}{r5_text}"
            )


def verdict(inputs: list[Input]) -> None:
    large = [i for i in inputs if i.voxels >= LARGE]
    if not large:
        print("\nНет входов от 512^3: правило выбора политики не к чему применять.")
        return
    order = large[0].policies
    print("\nПравило выбора политики (на всех входах от 512^3: СКЗ ≤ 0,01 МР и наибольшая ≤ 10 полов float, наш путь):")
    chosen = None
    for policy in order:
        failing = [i.name for i in large if not passes(i, "ours", policy)]
        print(f"  {policy:<32} {'проходит' if not failing else 'нет: ' + '; '.join(failing)}")
        if not failing and chosen is None:
            chosen = policy
    if chosen is None:
        real = [i for i in large if i.real]
        cufft_ok = [p for p in order if all(passes(i, "cufft", p) for i in real)]
        ours_ok = [p for p in order if all(passes(i, "ours", p) for i in real)]
        print("\nНи одна политика не проходит на всех входах.")
        if not cufft_ok and not ours_ok:
            print("  На реальных входах не проходит ни у нас, ни у cuFFT: ПРАВИЛО С8 — остановиться.")
        else:
            print(f"  На реальных проходят: у нас {ours_ok or '—'}, у cuFFT {cufft_ok or '—'}; С8 не срабатывает, нужен разбор.")
        return
    print(f"\nПолитика по умолчанию: {chosen}")

    print("\nЦ3 при этой политике (синтетика и реальные входы: СКЗ ≤ 0,01 МР, наибольшая ≤ 1,5 от cuFFT):")
    c3 = True
    for inp in inputs:
        if not inp.realistic or ("ours", chosen) not in inp.results:
            continue
        ours, cufft = inp.results[("ours", chosen)], inp.results[("cufft", chosen)]
        ratio = ours.largest / cufft.largest if cufft.largest else float("inf")
        ok = ours.rms <= RMS_LIMIT and ratio <= CUFFT_LIMIT
        c3 &= ok
        print(f"  {inp.name:<58} rms {ours.rms:.2e}  {ratio:5.2f}× cuFFT  {'да' if ok else 'НЕТ'}")
    print(f"  Ц3: {'выполнен' if c3 else 'НЕ выполнен'}")

    if any(("r5", chosen) in i.results for i in inputs):
        failing = [i.name for i in large if ("r5", chosen) in i.results and not passes(i, "r5", chosen)]
        print(f"\nЯдро R-5 при политике «{chosen}» (шаг 3.2):")
        print(f"  правило политики на входах от 512^3: {'проходит' if not failing else 'нет: ' + '; '.join(failing)}")
        c3r5 = True
        for inp in inputs:
            if not inp.realistic or ("r5", chosen) not in inp.results:
                continue
            r5, cufft = inp.results[("r5", chosen)], inp.results[("cufft", chosen)]
            ratio = r5.largest / cufft.largest if cufft.largest else float("inf")
            ok = r5.rms <= RMS_LIMIT and ratio <= CUFFT_LIMIT
            c3r5 &= ok
            print(f"  {inp.name:<58} rms {r5.rms:.2e}  {ratio:5.2f}× cuFFT  {'да' if ok else 'НЕТ'}")
        print(f"  Ц3 ядром R-5: {'выполнен' if c3r5 else 'НЕ выполнен'}")

    print("\nТуда и обратно (Ц4, предварительно; все политики, обе библиотеки):")
    worst = max(
        ((r.lost / i.voxels * 100.0, i.name, lib, pol) for i in inputs for (lib, pol), r in i.results.items()),
        default=(0.0, "", "", ""),
    )
    total = sum(r.lost for i in inputs for r in i.results.values())
    largest = max((r.trip_largest for i in inputs for r in i.results.values()), default=0.0)
    print(f"  не восстановлено вокселей всего: {total}; наибольшая доля {worst[0]:.3g} %; наибольшее |x' - x| {largest:.3g}")
    s8 = [i.name for i in inputs if i.real for r in i.results.values() if r.lost / i.voxels * 100.0 > LOSS_LIMIT]
    if s8:
        print(f"  ПРАВИЛО С8: на реальных входах потеряно больше {LOSS_LIMIT} % вокселей: {sorted(set(s8))}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("report", help="an accuracy_integer report (…_accuracy.txt)")
    parser.add_argument("--all", action="store_true", help="show the inputs below 512^3 too")
    args = parser.parse_args()
    inputs = parse(args.report)
    table(inputs, args.all)
    verdict(inputs)


if __name__ == "__main__":
    main()
