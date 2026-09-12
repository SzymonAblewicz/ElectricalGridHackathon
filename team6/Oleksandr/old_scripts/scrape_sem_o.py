#!/usr/bin/env python3
"""Download a complete SEM-O dynamic report to CSV.

The dynamic reports page (https://www.sem-o.com/market-data/dynamic-reports#BM-086)
shows one page of a table at a time, and its download button exports only the rows
currently on screen. The page is a thin client over a public JSON API, so this
script pages that API directly instead.

    GET https://reports.sem-o.com/api/v1/dynamic/<REPORT>
    -> {"pagination": {"pageSize", "currentPage", "totalItems", "totalPages"},
        "items": [ {...}, ... ]}

No authentication is required. BM-086 carries half-hourly metered output per
resource, with the columns TradeDate, ParticipantName, ResourceName, ResourceType,
StartTime, EndTime, Jurisdiction and MeteredMW.

Two things about this API decide the design, both verified against the live
service on 2026-09-07:

  * `page_size` is capped at 5000. Larger values are silently clamped rather than
    rejected, so asking for 50000 quietly gives you 5000.

  * Filtering on `TradeDate` DOES NOT WORK. The parameter is silently ignored: the
    request returns HTTP 200 and well-formed JSON containing the entire table
    (~3.85M rows), which looks exactly like a successful narrow query until you
    count the rows. `StartTime` is the field that actually filters. Do not
    "simplify" the window below to TradeDate.

The full dataset is a few million rows, so work is chunked one day at a time and
each day is cached as its own CSV. An interrupted run resumes at the first missing
day instead of starting over, and each day file is written atomically so a Ctrl-C
can never leave a truncated file that a later run would mistake for complete.

Usage:
    python scrape_sem_o.py                                  # everything available
    python scrape_sem_o.py --start 2026-07-01 --end 2026-07-03
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

# ---- configuration ---- #

BASE_URL = "https://reports.sem-o.com/api/v1/dynamic"
MAX_PAGE_SIZE = 5000  # server-side cap; larger requests are silently clamped
ATTEMPTS = 4
TIMEOUT = 120

FOLDER = Path(__file__).resolve().parent

# ---- HTTP ---- #


class Unavailable(RuntimeError):
    """The API did not return usable data after retrying."""


def fetch_page(report: str, params: dict, pause: float) -> dict:
    """GET one page of a report, retrying on timeouts, 429s and 5xx."""
    # Default quoting percent-encodes the comparison operators (">=" -> "%3E%3D",
    # "<" -> "%3C"), which is the exact form SEM-O documents and the form these
    # windows were verified against. Do not add safe="<>=".
    url = f"{BASE_URL}/{report}?" + urllib.parse.urlencode(params)
    last = ""

    for attempt in range(ATTEMPTS):
        if attempt:
            time.sleep(pause * 4 * attempt)  # linear backoff
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            last = f"HTTP {exc.code}"
            if exc.code != 429 and exc.code < 500:
                raise Unavailable(f"{last} for {url}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = str(exc)

    raise Unavailable(f"{ATTEMPTS} attempts failed for {url}: {last}")


def fetch_day(report: str, day: date, page_size: int, pause: float) -> list[dict]:
    """Fetch every row whose StartTime falls on `day`.

    The window is half-open (>= day, < day+1) so consecutive days cannot both
    claim the midnight trading period.
    """
    window = f">={day.isoformat()}<{(day + timedelta(days=1)).isoformat()}"
    params = {"StartTime": window, "page_size": page_size, "page": 1}

    payload = fetch_page(report, params, pause)
    pagination = payload["pagination"]
    expected = pagination["totalItems"]
    rows = list(payload["items"])

    for page in range(2, pagination["totalPages"] + 1):
        time.sleep(pause)
        params["page"] = page
        rows.extend(fetch_page(report, params, pause)["items"])

    # The server tells us how many rows exist before we start paging; if we did
    # not end up with exactly that many, something was lost or duplicated and the
    # day is not safe to cache. Raise rather than write a plausible-looking file.
    if len(rows) != expected:
        raise Unavailable(
            f"{day}: collected {len(rows)} rows but the API reported {expected}"
        )

    keys = {(row.get("ResourceName"), row.get("StartTime")) for row in rows}
    if len(keys) != len(rows):
        print(
            f"  warning: {day} has {len(rows) - len(keys)} duplicate "
            f"(ResourceName, StartTime) rows",
            file=sys.stderr,
        )

    return rows


def discover_range(report: str, pause: float) -> tuple[date, date]:
    """Find the first and last day the report currently covers."""
    bounds = []
    for order in ("ASC", "DESC"):
        payload = fetch_page(
            report,
            {"sort_by": "StartTime", "order_by": order, "page_size": 1, "page": 1},
            pause,
        )
        items = payload["items"]
        if not items:
            raise Unavailable(f"{report} returned no rows; cannot determine range")
        bounds.append(datetime.fromisoformat(items[0]["StartTime"]).date())

    return bounds[0], bounds[1]


# ---- output ---- #


def write_day_csv(rows: list[dict], path: Path, fields: list[str]) -> None:
    """Write one day to CSV atomically, so an interrupted run leaves no half file."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def combine(day_files: list[Path], target: Path) -> int:
    """Concatenate day files in order, keeping a single header row."""
    total = 0
    tmp = target.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as out:
        for index, day_file in enumerate(sorted(day_files)):
            with open(day_file, newline="", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle):
                    if line_no == 0 and index:
                        continue  # keep only the first file's header
                    if line_no:
                        total += 1
                    out.write(line)
    os.replace(tmp, target)
    return total


# ---- entry point ---- #


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download a complete SEM-O dynamic report to CSV."
    )
    parser.add_argument("--report", default="BM-086", help="report code, e.g. BM-086")
    parser.add_argument("--start", help="first day, YYYY-MM-DD (default: earliest)")
    parser.add_argument("--end", help="last day, YYYY-MM-DD (default: latest)")
    parser.add_argument("--out", type=Path, help="output directory")
    parser.add_argument("--page-size", type=int, default=MAX_PAGE_SIZE)
    parser.add_argument("--pause", type=float, default=0.3, help="seconds between calls")
    parser.add_argument("--force", action="store_true", help="refetch cached days")
    parser.add_argument("--no-combine", action="store_true")
    args = parser.parse_args()

    slug = args.report.lower()
    out_dir = args.out or FOLDER / "data" / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    page_size = min(args.page_size, MAX_PAGE_SIZE)
    if args.page_size > MAX_PAGE_SIZE:
        print(f"page_size clamped to the server maximum of {MAX_PAGE_SIZE}")

    first, last = discover_range(args.report, args.pause)
    start = date.fromisoformat(args.start) if args.start else first
    end = date.fromisoformat(args.end) if args.end else last
    if start > end:
        parser.error(f"--start {start} is after --end {end}")

    print(f"{args.report}: available {first} to {last}; fetching {start} to {end}")
    print(f"writing to {out_dir}")

    days = [start + timedelta(days=n) for n in range((end - start).days + 1)]
    fields: list[str] = []
    fetched = 0

    for index, day in enumerate(days, start=1):
        path = out_dir / f"{slug}-{day.isoformat()}.csv"
        if path.exists() and not args.force:
            print(f"[{index}/{len(days)}] {day}  cached")
            continue

        rows = fetch_day(args.report, day, page_size, args.pause)
        if not rows:
            # Don't cache an empty day: with no rows there are no column names to
            # write, and combine() takes its header from the earliest file.
            print(f"[{index}/{len(days)}] {day}  no rows, not cached")
            continue

        if not fields:
            fields = list(rows[0].keys())

        unexpected = {key for row in rows for key in row} - set(fields)
        if unexpected:
            raise Unavailable(f"{day}: unexpected columns {sorted(unexpected)}")

        write_day_csv(rows, path, fields)
        fetched += len(rows)
        print(f"[{index}/{len(days)}] {day}  {len(rows):,} rows")

    print(f"fetched {fetched:,} new rows")

    # if not args.no_combine:
    #     day_files = sorted(
    #         p for p in out_dir.glob(f"{slug}-*.csv") if p.name != f"{slug}-all.csv"
    #     )
    #     target = out_dir / f"{slug}-all.csv"
    #     total = combine(day_files, target)
    #     size_mb = target.stat().st_size / 1_048_576
    #     print(f"combined {len(day_files)} files -> {target} ({total:,} rows, {size_mb:.0f} MB)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
