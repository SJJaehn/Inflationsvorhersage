#!/usr/bin/env python3
"""Aggregate per-country FRED-style CSVs into one monthly CSV per country.

Layout
------
``variables.csv`` (the shared series spec) and this script live in this folder.
The actual data CSVs live in one subfolder per country, e.g. ``US/CPI.csv``.
Running the script applies ``variables.csv`` to every country subfolder and
writes the combined result back into that same subfolder as ``aggregated.csv``
(e.g. ``US/aggregated.csv``).

Rules
-----
* Every input CSV has two columns: an observation date and a value.
* The output is monthly. Each series is resampled to a monthly grid depending
  on its native frequency:
    - monthly            -> taken as-is
    - daily (or weekly)  -> last reported value of each month
    - quarterly / yearly -> forward filled until a new value is reported
* A per-series reporting lag (in months) is read from ``variables.csv`` and
  applied by shifting the series forward in time. A lag of 2 means the value
  observed for April is reported in (appears at) June.
* The final rows can be subsampled to a coarser frequency via ``keep_frequency``
  (constant near the top, or ``--keep-frequency`` on the command line):
  "monthly" keeps every month (default), "quarterly" keeps only the quarter-end
  months (Mar/Jun/Sep/Dec) and "yearly" keeps only December. This only drops
  rows; the kept values are the unchanged monthly values.

Files
-----
* ``variables.csv``  : the list of series to aggregate, one row each ->
                         - csv        : input file name
                         - lag_months : reporting lag (see above)
                         - rescaleN / change_typeN : up to three change
                                        operations applied IN ORDER (N = 1,2,3),
                                        each operating on the result of the
                                        previous one. Leave a rescaleN cell
                                        empty (or 0) to skip that step.
                            * rescaleN    : signed integer number of months.
                                            positive N -> trailing change
                                            (April 2026 vs April 2025);
                                            negative N -> forward change
                                            (the next-N-month change).
                            * change_typeN: "pct" (default) = percentage change,
                                            e.g. April 2026 / April 2025 - 1;
                                            "diff" = absolute change,
                                            e.g. April 2026 - April 2025.
                          Example: rescale1=-12/pct, rescale2=12/diff computes
                          the forward 12-month percentage change, then the
                          trailing 12-month absolute change of that series.
                       Only files listed here are read, and the output keeps
                       this order. The same file may appear multiple times with
                       different settings (e.g. CPI with no lag as a target and
                       CPI with a lag as a predictor); each row becomes its own
                       output column.
* ``<country>/``      : one subfolder per country holding that country's input
                       CSVs (named exactly as in ``variables.csv``).
* ``<country>/aggregated.csv`` : the combined monthly result for that country.
"""

from __future__ import annotations

import csv
import os
import re

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
VARIABLES = os.path.join(HERE, "variables.csv")
OUTPUT_NAME = "aggregated.csv"   # written into each country subfolder

# Row frequency of the output. "monthly" keeps every month; "quarterly" keeps
# only the quarter-end months (Mar/Jun/Sep/Dec); "yearly" keeps only the
# year-end month (Dec). Subsampling only drops the other months -- the kept
# values are the unchanged monthly values. Override with --keep-frequency.
KEEP_FREQUENCY = "monthly"

# Calendar months kept for each frequency (None = keep every month).
KEEP_MONTHS = {
    "monthly": None,
    "quarterly": {3, 6, 9, 12},
    "yearly": {12},
}


def read_variables() -> list[dict]:
    """Read variables.csv into an ordered list of row settings.

    Duplicate file names are allowed (each row is kept). Returns a list of
    {"name", "lag", "ops", "column"} dicts in file order, where ``ops`` is the
    ordered list of {"rescale", "change_type"} change operations to apply, and
    ``column`` is a unique name encoding the lag/op settings.
    """
    if not os.path.exists(VARIABLES):
        raise SystemExit(
            f"{VARIABLES} not found. Create it with columns: "
            "csv,lag_months,rescale1,change_type1,rescale2,change_type2,rescale3,change_type3"
        )

    rows: list[dict] = []
    with open(VARIABLES, newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            raise SystemExit(f"{VARIABLES} has no header row.")
        name_key = fieldnames[0]

        # Discover the rescale columns ("rescale", "rescale1", "rescale2", ...)
        # and pair each with its change_type column, in numeric order.
        op_keys: list[tuple[str, str]] = []
        numbered: dict[int, str] = {}
        for fn in fieldnames:
            m = re.fullmatch(r"rescale(\d*)", fn.strip())
            if m:
                numbered[int(m.group(1)) if m.group(1) else 0] = fn
        for n in sorted(numbered):
            rk = numbered[n]
            ck = rk.replace("rescale", "change_type", 1)
            op_keys.append((rk, ck))
        if not op_keys:
            raise SystemExit(f"{VARIABLES} has no rescale/rescaleN column.")

        for raw in reader:
            name = (raw.get(name_key) or "").strip()
            if not name:
                continue
            lag_str = (raw.get("lag_months") or "").strip()

            ops: list[dict] = []
            for rk, ck in op_keys:
                rescale_str = (raw.get(rk) or "").strip()
                if not rescale_str:
                    continue
                rescale = int(rescale_str)
                if rescale == 0:
                    continue
                change_type = (raw.get(ck) or "").strip().lower() or "pct"
                if change_type not in ("pct", "diff"):
                    raise SystemExit(
                        f"Invalid {ck} {change_type!r} for {name}; use 'pct' or 'diff'."
                    )
                ops.append({"rescale": rescale, "change_type": change_type})

            rows.append({
                "name": name,
                "lag": int(lag_str) if lag_str else 0,
                "ops": ops,
            })

    # Build a unique, descriptive column name for every row (handles duplicates).
    seen: dict[str, int] = {}
    for r in rows:
        base = os.path.splitext(r["name"])[0]
        for op in r["ops"]:
            base += f"_{op['change_type']}{op['rescale']}"
        if r["lag"]:
            base += f"_lag{r['lag']}"
        col = base
        if col in seen:
            seen[col] += 1
            col = f"{base}.{seen[col]}"
        else:
            seen[col] = 1
        r["column"] = col

    return rows


def detect_frequency(dates: pd.Series) -> str:
    """Classify a series as 'sub_monthly', 'monthly' or 'low' (quarterly+)."""
    if len(dates) < 2:
        return "monthly"
    median_days = dates.sort_values().diff().dt.days.median()
    if median_days < 25:
        return "sub_monthly"   # daily / weekly -> month-end last value
    if median_days < 45:
        return "monthly"
    return "low"               # quarterly / yearly -> forward fill


def apply_change(monthly: pd.Series, rescale: int, change_type: str) -> pd.Series:
    """Convert a monthly series to a change over ``rescale`` months.

    ``rescale`` is signed: a positive N gives the trailing change (value[t] vs
    value[t-N]); a negative N gives the forward change (value[t+N] vs value[t]).
    ``change_type`` is "pct" (ratio - 1) or "diff" (absolute difference).
    """
    periods = abs(rescale)
    if change_type == "diff":
        change = monthly.diff(periods=periods)
    else:
        change = monthly.pct_change(periods=periods, fill_method=None)
    # For a forward change, slide the trailing change back by N months so row t
    # holds value[t+N] vs value[t] instead of value[t] vs value[t-N].
    return change.shift(-periods) if rescale < 0 else change


def to_monthly(column: str, path: str, lag: int, ops: list[dict]) -> pd.Series:
    """Load one CSV and return a monthly PeriodIndex Series.

    Steps: resample to a monthly grid, apply each change operation in ``ops``
    in order (each operating on the result of the previous), then shift forward
    by ``lag`` months.
    """
    df = pd.read_csv(path)
    date_col, value_col = df.columns[0], df.columns[1]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.dropna(subset=[date_col]).sort_values(date_col)

    values = pd.to_numeric(df[value_col], errors="coerce")
    freq = detect_frequency(df[date_col])

    # Collapse to one value per calendar month (keep the last within a month).
    month = df[date_col].dt.to_period("M")
    monthly = (
        pd.Series(values.values, index=month)
        .groupby(level=0)
        .last()
    )

    # Put the series on a gap-free monthly grid so any later shifting and the
    # percentage change below count true calendar months, not just rows.
    full = pd.period_range(monthly.index.min(), monthly.index.max(), freq="M")
    if freq == "low":
        # Quarterly / yearly: forward fill so the last reported value carries
        # forward until a new value is reported.
        monthly = monthly.reindex(full).ffill()
    else:
        monthly = monthly.reindex(full)

    # Apply each change operation in order; each one operates on the result of
    # the previous (e.g. forward pct change, then trailing diff of that).
    for op in ops:
        monthly = apply_change(monthly, op["rescale"], op["change_type"])

    # Apply the reporting lag by shifting the months forward.
    if lag:
        monthly.index = monthly.index + lag

    monthly.name = column
    return monthly


def filter_frequency(combined: pd.DataFrame, keep_frequency: str) -> pd.DataFrame:
    """Subsample the monthly rows to the requested output frequency.

    "monthly" keeps every row; "quarterly" keeps only the quarter-end months
    (March, June, September, December); "yearly" keeps only December. This only
    drops rows -- the kept values are the unchanged monthly values.
    """
    key = (keep_frequency or "monthly").strip().lower()
    if key not in KEEP_MONTHS:
        raise SystemExit(
            f"Invalid keep_frequency {keep_frequency!r}; "
            f"use one of: {', '.join(KEEP_MONTHS)}."
        )
    months = KEEP_MONTHS[key]
    if months is None:
        return combined
    return combined[combined.index.month.isin(months)]


def list_countries() -> list[str]:
    """Return the country subfolder names under HERE (each holds its own CSVs).

    Hidden folders and ``__pycache__`` (anything starting with '.' or '_') are
    skipped.
    """
    return sorted(
        entry.name
        for entry in os.scandir(HERE)
        if entry.is_dir() and not entry.name.startswith((".", "_"))
    )


def aggregate_country(country_dir: str, rows: list[dict], keep_frequency: str) -> pd.DataFrame:
    """Build the combined monthly frame for one country subfolder."""
    series = [
        to_monthly(r["column"], os.path.join(country_dir, r["name"]), r["lag"], r["ops"])
        for r in rows
    ]
    combined = pd.concat(series, axis=1).sort_index()
    # PeriodIndex -> month-start dates for a clean observation_date column.
    combined.index = combined.index.to_timestamp()
    combined.index.name = "observation_date"
    # Subsample rows to the requested output frequency.
    return filter_frequency(combined, keep_frequency)


def main(keep_frequency: str = KEEP_FREQUENCY) -> None:
    rows = read_variables()
    if not rows:
        print("No variables listed in variables.csv.")
        return

    print("Variables read from variables.csv (lag / ops, in months):")
    for r in rows:
        ops_desc = " -> ".join(f"{op['change_type']}{op['rescale']}" for op in r["ops"]) or "none"
        print(f"  {r['column']}  <- {r['name']}  lag={r['lag']} ops=[{ops_desc}]")

    countries = list_countries()
    if not countries:
        raise SystemExit(
            f"No country subfolders found in {HERE}. Create one (e.g. US/) and "
            "put that country's data CSVs in it."
        )

    needed = [r["name"] for r in rows]
    for country in countries:
        country_dir = os.path.join(HERE, country)
        present = [n for n in needed if os.path.exists(os.path.join(country_dir, n))]
        if not present:
            print(f"\nSkipping {country}/ (no matching data CSVs).")
            continue
        missing = sorted(set(needed) - set(present))
        if missing:
            raise SystemExit(
                f"{country}/ is missing data file(s) listed in variables.csv: "
                f"{', '.join(missing)}"
            )

        combined = aggregate_country(country_dir, rows, keep_frequency)
        out_path = os.path.join(country_dir, OUTPUT_NAME)
        combined.to_csv(out_path, date_format="%Y-%m-%d")
        print(f"\nWrote {out_path}: {combined.shape[0]} rows x {combined.shape[1]} "
              f"series (keep_frequency={keep_frequency}).")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Aggregate the FRED-style CSVs in this folder into one CSV."
    )
    parser.add_argument(
        "--keep-frequency",
        choices=list(KEEP_MONTHS),
        default=KEEP_FREQUENCY,
        help="Output row frequency: monthly (all months), quarterly "
             "(Mar/Jun/Sep/Dec) or yearly (Dec). Default: %(default)s.",
    )
    args = parser.parse_args()
    main(keep_frequency=args.keep_frequency)
