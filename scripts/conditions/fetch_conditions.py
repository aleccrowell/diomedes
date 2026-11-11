"""Per-lap conditions for the lap model (#116): tyre compound, track temperature, rainfall.

    python3 scripts/conditions/fetch_conditions.py [first-last]     # default 2021-<this year>

Writes data/conditions/lap_conditions.csv, one row per car per lap of each race:
    season, round, event_id, driver_number, lap, compound, tyre_age, lap_time_ms,
    track_temp, air_temp, rainfall, weather_n, source

Sources (neither needs an account; both are unofficial, so raw responses are cached):
- 2021-2022: FastF1 (https://docs.fastf1.dev), which reads the F1 live-timing archive from its
  own mirror. The official archive (livetiming.formula1.com/static) needs an F1 account
  (401/403 to plain requests, checked 2026-11-08), and the mirror only has 2021-2022.
  Cache: data/cache/fastf1.
- 2023 on: OpenF1 (https://openf1.org) laps, stints and weather. Free without a key except
  while a session is live; ~30 requests a minute (429 past that). Cache: data/cache/openf1.
- Race rounds and dates: Jolpica (the source of the lap times the model uses), so event_id
  matches `fetch_laps`'s ("<season>-<round, 2 digits>").

Weather comes about once a minute. A lap gets the mean track and air temperature of the
samples inside its time window and rainfall 1 if any of them shows rain (`weather_n` samples);
with none inside, the sample nearest the lap's midpoint (weather_n = 0). `compound` is the
source's label (SOFT, MEDIUM, HARD, INTERMEDIATE, WET; UNKNOWN or empty when not known);
SOFT/MEDIUM/HARD are relative to the race's allocation. `tyre_age` is the tyre's laps
including those it had done before the stint (as the source counts them).

Needs: pip install fastf1 (3.9 tested), run from the repository root.
"""
import bisect
import csv
import datetime as dt
import json
import math
import os
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, "data", "cache")
OUT = os.path.join(ROOT, "data", "conditions", "lap_conditions.csv")
FIELDS = ["season", "round", "event_id", "driver_number", "lap", "compound", "tyre_age", "lap_time_ms",
          "track_temp", "air_temp", "rainfall", "weather_n", "source"]
FASTF1_SEASONS = range(2021, 2023)      # the FastF1 mirror's seasons; OpenF1 from 2023


def log(*a):
    print(*a, flush=True)


def get_json(url, path, min_interval):
    """GET url as JSON, cached at path; backs off on 429/5xx."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    delay = 5.0
    for attempt in range(8):
        time.sleep(min_interval)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Diomedes (github.com/aleccrowell/diomedes)"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.load(r)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(data, f)
            return data
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code == 429 or e.code >= 500:
                log(f"  HTTP {e.code} for {url}; retrying in {delay:.0f}s")
                time.sleep(delay)
                delay = min(delay * 2, 120)
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            log(f"  {e} for {url}; retrying in {delay:.0f}s")
            time.sleep(delay)
            delay = min(delay * 2, 120)
    raise RuntimeError(f"giving up on {url}")


def jolpica_races(season):
    url = f"https://api.jolpi.ca/ergast/f1/{season}/races.json?limit=100"
    d = get_json(url, os.path.join(CACHE, "openf1", f"jolpica_races_{season}.json"), 0.5)
    return [(int(r["round"]), dt.date.fromisoformat(r["date"])) for r in d["MRData"]["RaceTable"]["Races"]]


def lap_weather(t0, t1, wt, wrows):
    """Mean temperatures and any rain over the samples with time in [t0, t1] (seconds); else the nearest."""
    if t0 is None or not wt:
        return (math.nan, math.nan, math.nan, 0)
    if t1 is None or t1 < t0:
        t1 = t0
    i, j = bisect.bisect_left(wt, t0), bisect.bisect_right(wt, t1)
    sel = wrows[i:j]
    if sel:
        tt = [w[0] for w in sel if not math.isnan(w[0])]
        at = [w[1] for w in sel if not math.isnan(w[1])]
        return (sum(tt) / len(tt) if tt else math.nan, sum(at) / len(at) if at else math.nan,
                float(max(w[2] for w in sel)), len(sel))
    mid = (t0 + t1) / 2
    k = bisect.bisect_left(wt, mid)
    cand = [c for c in (k - 1, k) if 0 <= c < len(wt)]
    c = min(cand, key=lambda c: abs(wt[c] - mid))
    return (*wrows[c], 0)


def fnum(x):
    try:
        v = float(x)
        return v if not math.isnan(v) else math.nan
    except (TypeError, ValueError):
        return math.nan


# ---- 2021-2022: FastF1 ---------------------------------------------------------------------
def fastf1_race(season, rnd):
    import fastf1
    s = fastf1.get_session(season, rnd, "R")
    s.load(laps=True, weather=True, telemetry=False, messages=False)
    w = s.weather_data.sort_values("Time")
    wt = [x.total_seconds() for x in w["Time"]]
    wrows = [(fnum(a), fnum(b), float(bool(c))) for a, b, c in zip(w["TrackTemp"], w["AirTemp"], w["Rainfall"])]
    out = []
    for _, r in s.laps.iterrows():
        t0 = r["LapStartTime"].total_seconds() if r["LapStartTime"] == r["LapStartTime"] else None
        lt = r["LapTime"].total_seconds() if r["LapTime"] == r["LapTime"] else None
        t1 = t0 + lt if t0 is not None and lt is not None else (r["Time"].total_seconds() if r["Time"] == r["Time"] else None)
        tt, at, rain, n = lap_weather(t0, t1, wt, wrows)
        comp = r["Compound"] if isinstance(r["Compound"], str) else ""
        out.append(dict(driver_number=int(r["DriverNumber"]), lap=int(r["LapNumber"]), compound=comp,
                        tyre_age=fnum(r["TyreLife"]), lap_time_ms=lt * 1000 if lt is not None else math.nan,
                        track_temp=tt, air_temp=at, rainfall=rain, weather_n=n, source="fastf1"))
    return out


# ---- 2023 on: OpenF1 -----------------------------------------------------------------------
OPENF1 = "https://api.openf1.org/v1"


def openf1(endpoint, query, name):
    return get_json(f"{OPENF1}/{endpoint}?{query}", os.path.join(CACHE, "openf1", f"{name}.json"), 2.1)


def ts(s):
    return dt.datetime.fromisoformat(s).timestamp() if s else None


def openf1_sessions(season):
    s = openf1("sessions", f"year={season}&session_name=Race", f"sessions_{season}_race") or []
    return [x for x in s if not x.get("is_cancelled")]


def openf1_race(key):
    laps = openf1("laps", f"session_key={key}", f"laps_{key}") or []
    stints = openf1("stints", f"session_key={key}", f"stints_{key}") or []
    weather = sorted(openf1("weather", f"session_key={key}", f"weather_{key}") or [], key=lambda w: w["date"])
    wt = [ts(w["date"]) for w in weather]
    wrows = [(fnum(w.get("track_temperature")), fnum(w.get("air_temperature")), float(bool(w.get("rainfall"))))
             for w in weather]
    tyre = {}
    for st in stints:
        if st.get("lap_start") is None or st.get("lap_end") is None:
            continue
        for L in range(st["lap_start"], st["lap_end"] + 1):
            age0 = st.get("tyre_age_at_start")
            tyre.setdefault((st["driver_number"], L),
                            (st.get("compound") or "", age0 + (L - st["lap_start"]) + 1 if age0 is not None else math.nan))
    out = []
    for r in laps:
        t0, lt = ts(r.get("date_start")), r.get("lap_duration")
        tt, at, rain, n = lap_weather(t0, t0 + lt if t0 is not None and lt is not None else None, wt, wrows)
        comp, age = tyre.get((r["driver_number"], r["lap_number"]), ("", math.nan))
        out.append(dict(driver_number=int(r["driver_number"]), lap=int(r["lap_number"]), compound=comp, tyre_age=age,
                        lap_time_ms=lt * 1000 if lt is not None else math.nan, track_temp=tt, air_temp=at,
                        rainfall=rain, weather_n=n, source="openf1"))
    return out


def main():
    first, last = 2021, dt.date.today().year
    if len(sys.argv) > 1:
        first, last = map(int, sys.argv[1].split("-"))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if any(s in FASTF1_SEASONS for s in range(first, last + 1)):
        import fastf1
        os.makedirs(os.path.join(CACHE, "fastf1"), exist_ok=True)
        fastf1.Cache.enable_cache(os.path.join(CACHE, "fastf1"))
        fastf1.set_log_level("ERROR")
    rows, missing = [], []
    today = dt.date.today()
    for season in range(first, last + 1):
        races = [(r, d) for r, d in jolpica_races(season) if d < today]
        sess = openf1_sessions(season) if season not in FASTF1_SEASONS else []
        for rnd, date in races:
            eid = f"{season}-{rnd:02d}"
            t = time.time()
            try:
                if season in FASTF1_SEASONS:
                    laps = fastf1_race(season, rnd)
                else:
                    # the OpenF1 race session within a day of Jolpica's (local) race date
                    m = [x for x in sess if abs((dt.datetime.fromisoformat(x["date_start"]).date() - date).days) <= 1]
                    if len(m) != 1:
                        raise RuntimeError(f"{len(m)} OpenF1 race sessions near {date}")
                    laps = openf1_race(m[0]["session_key"])
            except Exception as e:
                log(f"{eid} FAILED: {type(e).__name__}: {str(e)[:200]}")
                missing.append(eid)
                continue
            if not laps:
                log(f"{eid} no laps")
                missing.append(eid)
                continue
            for x in laps:
                rows.append(dict(season=season, round=rnd, event_id=eid, **x))
            known = sum(1 for x in laps if x["compound"] not in ("", "UNKNOWN", "TEST_UNKNOWN"))
            wet = sum(1 for x in laps if x["compound"] in ("INTERMEDIATE", "WET"))
            log(f"{eid} {len(laps):5d} laps, compound known {known / len(laps):.2f}, inter/wet {wet / len(laps):.2f}, "
                f"rain laps {sum(x['rainfall'] == 1 for x in laps)}, ({time.time() - t:.0f}s)")
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if isinstance(v, float) and math.isnan(v) else v) for k, v in r.items()})
    log(f"wrote {len(rows)} laps to {OUT}; races without data: {', '.join(missing) or 'none'}")


if __name__ == "__main__":
    main()
