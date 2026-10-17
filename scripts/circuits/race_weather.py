# Race start times and race-window air temperature for every GP 1996-2025.
# Start times: 1996-2004 from the documented table (start_times_1996_2004.csv; 14:00 default,
# unconfirmed in 1996-97); 2005-06 Jolpica times are local clock times mislabelled "Z";
# 2007+ Jolpica times are UTC. Temperature: Open-Meteo archive (ERA5), mean of the two hours
# from the start, in the circuit's local time zone; also +-1h and +-2h windows and rain.
#
#   python3 scripts/circuits/race_weather.py <repo root> <Jolpica cache dir> [<Open-Meteo cache dir>]
#
# Reads reference/circuits/start_times_1996_2004.csv and the cached Jolpica race schedules; writes
# reference/circuits/race_weather.csv. One small Open-Meteo request per race date (cached; the free
# tier weights requests by size, so whole-history requests hit its limit).
import json, glob, csv, sys, time, urllib.request, urllib.error
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
ROOT = sys.argv[1]; CACHE = sys.argv[2]; S = ROOT + '/reference/circuits'
METEO = sys.argv[3] if len(sys.argv) > 3 else ROOT + '/data/cache/open-meteo'
races = {}
for f in glob.glob(CACHE + '/*.json'):
    try: d = json.load(open(f))
    except Exception: continue
    for r in d.get('MRData', {}).get('RaceTable', {}).get('Races', []) if isinstance(d, dict) else []:
        y = int(r['season'])
        if y < 1996: continue
        k = (y, int(r['round']))
        if k not in races or ('time' in r and 'time' not in races[k]): races[k] = r
doc = {}
for row in csv.DictReader(open(S + '/start_times_1996_2004.csv')):
    ys = []
    for part in row['years'].split(','):
        a, _, b = part.partition('-'); ys += list(range(int(a), int(b or a) + 1))
    for y in ys: doc[(row['circuit_id'], y)] = (row['local_start'], row['status'])
import os, hashlib
os.makedirs(METEO, exist_ok=True)
def get(u):
    f = METEO + '/' + hashlib.sha1(u.encode()).hexdigest() + '.json'
    if os.path.exists(f): return json.load(open(f))
    for attempt in range(8):
        try:
            d = json.load(urllib.request.urlopen(u, timeout=60)); json.dump(d, open(f, 'w')); time.sleep(1.2); return d
        except urllib.error.HTTPError as e:
            if e.code != 429: raise
            time.sleep(60 * (attempt + 1))
    raise RuntimeError('rate limited: ' + u)
tzname = {}
def tz_of(cid, lat, lon):
    if cid not in tzname:
        tzname[cid] = get(f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date=2010-01-01&end_date=2010-01-01&hourly=temperature_2m&timezone=auto")['timezone']
    return ZoneInfo(tzname[cid])
def hourly(cid, lat, lon, date):
    d0 = (datetime.fromisoformat(date) - timedelta(days=1)).date(); d1 = (datetime.fromisoformat(date) + timedelta(days=1)).date()
    h = get(f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={d0}&end_date={d1}&hourly=temperature_2m,precipitation&timezone=GMT")['hourly']
    return {t: (tt, pp) for t, tt, pp in zip(h['time'], h['temperature_2m'], h['precipitation'])}
years = {}
for (y, rd), r in races.items():
    years.setdefault(r['Circuit']['circuitId'], []).append(y)
out = []
for (y, rd), r in sorted(races.items()):
    c = r['Circuit']; cid = c['circuitId']; lat, lon = c['Location']['lat'], c['Location']['long']
    tz = tz_of(cid, lat, lon)
    if y <= 2004:
        ls, status = doc.get((cid, y), ('14:00', 'assumed'))
        src = 'documented ' + status
        start = datetime.fromisoformat(f"{r['date']}T{ls}").replace(tzinfo=tz)
    elif y <= 2006:
        start = datetime.fromisoformat(f"{r['date']}T{r['time'][:5]}").replace(tzinfo=tz); src = 'jolpica local (labelled Z)'
    else:
        start = datetime.fromisoformat(f"{r['date']}T{r['time'][:5]}").replace(tzinfo=ZoneInfo('UTC')); src = 'jolpica UTC'
    utc = start.astimezone(ZoneInfo('UTC'))
    H = hourly(cid, lat, lon, r['date'])
    def window(shift):
        t0 = (utc + timedelta(hours=shift)).replace(minute=0)
        vals = [H.get((t0 + timedelta(hours=i)).strftime('%Y-%m-%dT%H:00')) for i in range(3)]
        w = [v for v in vals if v is not None and v[0] is not None]
        if not w: return None, None
        return round(sum(v[0] for v in w) / len(w), 2), round(sum((v[1] or 0) for v in w), 2)
    t, p = window(0); tm1, _ = window(-1); tp1, _ = window(1); tm2, _ = window(-2); tp2, _ = window(2)
    out.append(dict(season=y, round=rd, circuit_id=cid, race=r['raceName'], date=r['date'],
                    local_start=start.astimezone(tz).strftime('%H:%M'), utc_start=utc.strftime('%Y-%m-%dT%H:%MZ'),
                    start_source=src, timezone=str(tz), temp_c=t, rain_mm=p, temp_m2h=tm2, temp_m1h=tm1, temp_p1h=tp1, temp_p2h=tp2))
w = csv.DictWriter(open(S + '/race_weather.csv', 'w', newline=''), fieldnames=list(out[0].keys()))
w.writeheader(); w.writerows(out)
print(len(out), 'races written')
