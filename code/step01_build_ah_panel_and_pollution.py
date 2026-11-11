# Source: 01_build_pollution.ipynb, all 55 code cells in order (A/H prices, pair list, mainland and HK PM2.5).
# This notebook mixes building and inspection; cells are separated by "# %%". Run top to bottom in Colab.

# %
# ── 01: build the pollution panel ──────────────────────────────────
!pip install netcdf4 xarray -q

from google.colab import drive
drive.mount('/content/drive')

import glob, numpy as np, pandas as pd
DIR = "/content/drive/MyDrive/pollution-ah"
print(sorted(f.split("/")[-1] for f in glob.glob(f"{DIR}/*")))

# %%
# ── Hong Kong daily PM2.5 ──────────────────────────────────────────
# 15 general stations + 3 roadside (Central, Mong Kok, Causeway Bay).
# Roadside read systematically higher and are excluded from the city
# mean, kept only as a robustness check. North and Southern have ~40%
# coverage and are dropped so the station set stays stable over time.

GENERAL = ["SHATIN","TSUEN WAN","EASTERN","KWUN TONG","TUEN MUN","TUNG CHUNG",
           "SHAM SHUI PO","YUEN LONG","CENTRAL/WESTERN","KWAI CHUNG",
           "TAP MUN","TSEUNG KWAN O","TAI PO"]
ROADSIDE = ["CENTRAL","MONG KOK","CAUSEWAY BAY"]
MIN_STATIONS = 8

parts = []
for f in sorted(glob.glob(f"{DIR}/hk_pm25_*.csv")):
    d = pd.read_csv(f, skiprows=11, na_values=["N.A."], encoding="utf-8-sig")
    d.columns = [c.strip() for c in d.columns]
    parts.append(d)
    print(f"  {f.split('/')[-1]}: {len(d)} rows")

H = pd.concat(parts, ignore_index=True)
H["ts"] = (pd.to_datetime(H.DATE, format="%d/%m/%Y")
           + pd.to_timedelta(H.HOUR.astype(int) - 1, unit="h"))   # HOUR is 1–24
H = H.drop_duplicates("ts").set_index("ts").sort_index()

gen = H[GENERAL].apply(pd.to_numeric, errors="coerce")
n_st = gen.notna().sum(axis=1)
hk_hourly = gen.mean(axis=1).where(n_st >= MIN_STATIONS)

hk = hk_hourly.resample("D").mean().dropna()
hk.name = "hk_pm25"

print(f"\nhourly: {len(H)} rows, {H.index.min().date()} to {H.index.max().date()}")
print(f"stations used: {len(GENERAL)} general (North, Southern dropped for ~40% coverage)")
print(f"daily series: {len(hk)} days")
print(f"  mean {hk.mean():.1f}   sd {hk.std():.1f}   min {hk.min():.1f}   max {hk.max():.1f}")

# %%
# ── mainland China daily PM2.5, all cities ─────────────────────────
# Raw MEP station observations from Zenodo 10.5281/zenodo.6404603
# (Li & Li, Institute of Earth Environment, CAS; CC BY 4.0).
# Winters only: Dec/Jan/Feb, 2013-14 through 2019-20.
#
# Cleaning, all discovered by inspection rather than documented:
#   fill value 9.969e36 is NOT auto-masked   -> drop
#   negative concentrations occur            -> drop
#   timestamps are UTC, China is UTC+8       -> convert
#   duplicate and out-of-month timestamps    -> drop, keep the file's month
#   coverage varies                          -> require >=8 valid hours/day

import urllib.request, os, re, xarray as xr

MONTHS = ["201312","201401","201402","201412","201501","201502",
          "201512","201601","201602","201612","201701","201702",
          "201712","201801","201802","201812","201901","201902",
          "201912","202001","202002"]
BASE = "https://zenodo.org/records/6404603/files"
os.makedirs("/content/cn_raw", exist_ok=True)

frames = []
for m in MONTHS:
    p = f"/content/cn_raw/aqi{m}.nc"
    if not os.path.exists(p):
        urllib.request.urlretrieve(f"{BASE}/aqi{m}.nc?download=1", p)

    ds = xr.open_dataset(p)
    area = np.array([a.decode("utf-8","ignore").strip() for a in ds.area.values])
    t = pd.to_datetime([s.decode().strip() for s in ds.time_point.values],
                       format="%Y-%m-%dT%H:%M:%SZ", utc=True)

    PM = ds.pm2_5.values.astype("float64")
    PM[(PM > 1e6) | (PM < 0)] = np.nan

    X = pd.DataFrame(PM, index=t, columns=area)
    X = X[~X.index.duplicated(keep="first")]
    X.index = X.index.tz_convert("Asia/Shanghai")
    X = X[(X.index.year == int(m[:4])) & (X.index.month == int(m[4:]))]  # drop strays

    # average stations within a city, then require >=8 valid hours per day
    hourly = X.T.groupby(level=0).mean().T
    d = hourly.groupby(hourly.index.normalize())
    daily, hrs = d.mean(), d.count()
    frames.append(daily.where(hrs >= 8))
    ds.close()
    print(f"  {m}: {daily.shape[1]} cities, {len(daily)} days")

CN = pd.concat(frames).sort_index()
CN.index = CN.index.tz_localize(None)
CN.index.name = "date"

print(f"\ncities: {CN.shape[1]}   days: {len(CN)}")
print(f"span: {CN.index.min().date()} to {CN.index.max().date()}")
print(f"non-missing city-days: {100*CN.notna().mean().mean():.1f}%")
for c in ["Shanghai","Shenzhen","Beijing"]:
    s = CN[c].dropna()
    print(f"  {c:<10}{len(s):>5} days   mean {s.mean():5.1f}   max {s.max():6.1f}")

# %%
CN.to_csv(f"{DIR}/cn_pm25_daily.csv")

cov = CN.notna().mean().sort_values(ascending=False)
print(f"cities with >=90% coverage: {(cov >= 0.90).sum()}")
print(f"cities with >=50% coverage: {(cov >= 0.50).sum()}")
print(f"cities with <10% coverage:  {(cov < 0.10).sum()}")
print("\nworst 10:")
print((100*cov.tail(10)).round(1).to_string())

# %%
import re
suff = [c for c in CN.columns if re.search(r"_[a-z]{2}$", c)]
print(f"{len(suff)} suffixed names:", suff)

for c in suff:
    base = c.rsplit("_", 1)[0]
    if base in CN.columns:
        a, b = CN[base], CN[c]
        overlap = (a.notna() & b.notna()).sum()
        print(f"\n{base} ({a.notna().sum()}d) vs {c} ({b.notna().sum()}d)"
              f"  overlapping days: {overlap}")
        if overlap:
            print(f"   correlation on overlap: {a.corr(b):.3f}")

# %%
AMBIG = ["Bozhou","Fuyang","Fuzhou","Suzhou","Taizhou"]
drop = [c for c in CN.columns if c.rsplit("_",1)[0] in AMBIG]
print(f"dropping {len(drop)}: {drop}")

CN = CN.drop(columns=drop)
CN.to_csv(f"{DIR}/cn_pm25_daily.csv")

cov = CN.notna().mean()
print(f"\ncities remaining: {CN.shape[1]}")
print(f"  >=90% coverage: {(cov >= 0.90).sum()}")
print(f"  >=50% coverage: {(cov >= 0.50).sum()}")
for c in ["Shanghai","Shenzhen","Beijing"]:
    print(f"  {c}: {100*cov[c]:.1f}% — present")

# %%
# ── inspect the A-H structure on one pair ──────────────────────────
!pip install akshare -q
import akshare as ak

ah = ak.stock_zh_ah_name()
print(f"pairs: {len(ah)}")
print(ah.head(5).to_string())

# CCB: H-share code 00939
h = ak.stock_zh_ah_daily(symbol="00939", start_year="2013", end_year="2020")
print(f"\ndaily columns: {list(h.columns)}")
print(f"rows: {len(h)}   {h.iloc[0,0]} to {h.iloc[-1,0]}")
print(h.head(3).to_string())

# does the function return the A side, the H side, or both?
print("\nclose price range:", h.iloc[:,2].min(), "to", h.iloc[:,2].max())

# %%
d = h.iloc[:, 0]
print("duplicate dates:", d.duplicated().sum())
print("unique dates:", d.nunique(), "of", len(d))
print("\nfirst 6 rows:")
print(h.head(6).to_string())

# if dates repeat, the two series are stacked
if d.duplicated().sum() > 0:
    g = h.groupby(h.columns[0]).size()
    print("\nrows per date:", g.value_counts().to_dict())
    ex = d[d.duplicated(keep=False)].iloc[0]
    print(f"\nboth rows for {ex}:")
    print(h[d == ex].to_string())

# %%
h = h.drop_duplicates(subset=h.columns[0]).reset_index(drop=True)
print(f"H side after dedup: {len(h)} rows, {h.iloc[0,0]} to {h.iloc[-1,0]}")

# CCB's A-share is 601939 on the Shanghai exchange
try:
    a = ak.stock_zh_a_hist(symbol="601939", period="daily",
                           start_date="20130101", end_date="20201231", adjust="")
    print(f"\nA side: {len(a)} rows")
    print(f"columns: {list(a.columns)}")
    print(a.head(3).to_string())
    print(f"\nJan 2013 close ~ {a.iloc[0,2]}  (A-share should be ~4.5 RMB)")
    print(f"Jan 2013 H close ~ {h.iloc[0,2]}  (H-share should be ~6.4 HKD)")
except Exception as e:
    print("stock_zh_a_hist failed:", type(e).__name__, str(e)[:200])
    print("\navailable A-share functions:")
    print([f for f in dir(ak) if f.startswith("stock_zh_a_") and "hist" in f][:20])

# %%
cands = [f for f in dir(ak) if "_ah_" in f]
print("A-H functions:", cands)

for fn in cands:
    if fn in ("stock_zh_ah_daily", "stock_zh_ah_name"):
        continue
    try:
        r = getattr(ak, fn)()
        print(f"\n=== {fn} ===  {r.shape}")
        print(list(r.columns))
        print(r.head(3).to_string())
    except Exception as e:
        print(f"\n{fn}: {type(e).__name__} {str(e)[:120]}")

# %%
m = ak.stock_zh_ah_spot_em()[["名称", "H股代码", "A股代码"]].copy()
m.columns = ["name", "h_code", "a_code"]
m["h_code"] = m.h_code.astype(str).str.zfill(5)
m["a_code"] = m.a_code.astype(str).str.zfill(6)

print(f"pairs with both codes: {m.notna().all(axis=1).sum()} of {len(m)}")
print(f"duplicate H codes: {m.h_code.duplicated().sum()}   "
      f"duplicate A codes: {m.a_code.duplicated().sum()}")

# does it contain the pair we verified by hand?
print("\nCCB row:")
print(m[m.h_code == "00939"].to_string())

# A-code prefixes tell us which mainland exchange
print("\nA-code prefixes:")
print(m.a_code.str[:3].value_counts().head(10).to_string())

m.to_csv(f"{DIR}/ah_pairs.csv", index=False)

# %%
import time
for attempt in range(5):
    try:
        raw = ak.stock_zh_ah_spot_em()
        print(f"got it on attempt {attempt+1}: {raw.shape}")
        break
    except Exception as e:
        print(f"attempt {attempt+1}: {type(e).__name__}")
        time.sleep(4)
else:
    raw = None
    print("all attempts failed — try again in a minute")

if raw is not None:
    m = raw[["名称", "H股代码", "A股代码"]].copy()
    m.columns = ["name", "h_code", "a_code"]
    m["h_code"] = m.h_code.astype(str).str.zfill(5)
    m["a_code"] = m.a_code.astype(str).str.zfill(6)

    print(f"\npairs: {len(m)}")
    print(f"duplicate H: {m.h_code.duplicated().sum()}   duplicate A: {m.a_code.duplicated().sum()}")
    print("\nCCB:")
    print(m[m.h_code == "00939"].to_string())
    print("\nA-code prefixes (600/601/603/688 = Shanghai, 000/002/300 = Shenzhen):")
    print(m.a_code.str[:3].value_counts().to_string())

    m.to_csv(f"{DIR}/ah_pairs.csv", index=False)
    print(f"\nsaved to {DIR}/ah_pairs.csv")

# %%
# H codes + names from the pair list; A codes + names from the full listing
ah = ak.stock_zh_ah_name()
ah.columns = ["h_code", "name"]
ah["h_code"] = ah.h_code.astype(str).str.zfill(5)
print(f"H-side pairs: {len(ah)}")

try:
    alist = ak.stock_info_a_code_name()
    alist.columns = ["a_code", "name"]
    alist["a_code"] = alist.a_code.astype(str).str.zfill(6)
    print(f"all A-shares: {len(alist)}")
except Exception as e:
    print("A listing failed:", type(e).__name__, str(e)[:150])
    alist = None

if alist is not None:
    m = ah.merge(alist, on="name", how="left")
    ok = m.a_code.notna()
    print(f"\nmatched on exact name: {ok.sum()} of {len(m)}")
    print(f"names matching >1 A-code: {m[ok].h_code.duplicated().sum()}")

    print("\nCCB check (expect 601939):")
    print(m[m.h_code == "00939"].to_string())

    print("\nunmatched, first 15:")
    print(m[~ok].head(15).to_string())

# %%
ah.to_csv(f"{DIR}/ah_h_codes.csv", index=False)
print("saved H-side pair list:", len(ah), "rows")

# %%
ah.to_csv(f"{DIR}/ah_h_codes.csv", index=False)
print(f"saved H-side list: {len(ah)} pairs\n")

# does the daily endpoint still work?
try:
    t = ak.stock_zh_ah_daily(symbol="00939", start_year="2013", end_year="2020")
    print(f"daily endpoint alive: {len(t)} rows for CCB")
    alive = True
except Exception as e:
    print("daily endpoint also blocked:", type(e).__name__)
    alive = False

# %%
# ── H-side daily prices, all 220 pairs ─────────────────────────────
import time, os

OUT = f"{DIR}/h_prices.csv"
done = {}
if os.path.exists(OUT):
    prev = pd.read_csv(OUT, parse_dates=["date"])
    done = {c: g for c, g in prev.groupby("h_code")}
    print(f"resuming: {len(done)} already saved")

rows, failed = [], []
for i, code in enumerate(ah.h_code, 1):
    if code in done:
        rows.append(done[code]); continue
    try:
        d = ak.stock_zh_ah_daily(symbol=code, start_year="2013", end_year="2020")
        d = d.iloc[:, :6]
        d.columns = ["date", "open", "close", "high", "low", "volume"]
        d["date"] = pd.to_datetime(d.date)
        d = d.drop_duplicates("date")
        d["h_code"] = code
        rows.append(d)
    except Exception as e:
        failed.append((code, type(e).__name__))
    if i % 20 == 0:
        pd.concat(rows, ignore_index=True).to_csv(OUT, index=False)
        print(f"  {i}/220   kept {len(rows)}   failed {len(failed)}")
    time.sleep(0.6)

H = pd.concat(rows, ignore_index=True)
H.to_csv(OUT, index=False)

print(f"\nseries: {H.h_code.nunique()} of 220")
print(f"rows: {len(H):,}   {H.date.min().date()} to {H.date.max().date()}")
n = H.groupby("h_code").size()
print(f"days per series: median {n.median():.0f}, min {n.min()}, max {n.max()}")
if failed:
    print(f"\nfailed ({len(failed)}): {failed[:10]}")

# %%
n = H.groupby("h_code").size().sort_values(ascending=False)
print("series with >2500 rows:", (n > 2500).sum())
print(n.head(8).to_string())

dup = H.duplicated(["h_code","date"]).sum()
print(f"\nduplicate (code,date) pairs: {dup}")

H = H.drop_duplicates(["h_code","date"])
n2 = H.groupby("h_code").size()
print(f"\nafter dedup: {H.h_code.nunique()} series, {len(H):,} rows")
print(f"days per series: median {n2.median():.0f}, min {n2.min()}, max {n2.max()}")
print(f"series with <500 days: {(n2 < 500).sum()}")

H.to_csv(OUT, index=False)

# %%
print("CCB present:", "00939" in set(H.h_code))

n2 = H.groupby("h_code").size()
short = n2[n2 < 500]
print(f"\nseries under 500 days ({len(short)}):")
print(short.to_string())

# how much of the sample falls in the winter months the pollution data covers?
w = H[H.date.dt.month.isin([12,1,2])]
print(f"\nwinter rows: {len(w):,} of {len(H):,} ({100*len(w)/len(H):.0f}%)")
print(f"winter days per series: median {w.groupby('h_code').size().median():.0f}")

# %%
H.to_csv(OUT, index=False)
print("saved:", OUT, len(H), "rows")

# %%
import time
raw = None
for attempt in range(3):
    try:
        raw = ak.stock_zh_ah_spot_em()
        print(f"success on attempt {attempt+1}: {raw.shape}")
        break
    except Exception as e:
        print(f"attempt {attempt+1}: {type(e).__name__}")
        time.sleep(20)

if raw is not None:
    m = raw[["名称", "H股代码", "A股代码"]].copy()
    m.columns = ["name", "h_code", "a_code"]
    m["h_code"] = m.h_code.astype(str).str.zfill(5)
    m["a_code"] = m.a_code.astype(str).str.zfill(6)
    m.to_csv(f"{DIR}/ah_pairs.csv", index=False)

    print(f"\npairs: {len(m)}")
    print("CCB (expect 601939):")
    print(m[m.h_code == "00939"].to_string())
    print("\nA-code prefixes (6xx = Shanghai, 0xx/3xx = Shenzhen):")
    print(m.a_code.str[:1].value_counts().to_string())
    have = set(H.h_code)
    print(f"\nmapped pairs where we already have H prices: {m.h_code.isin(have).sum()}")

# %%
!pip install cdsapi -q

CDS_KEY = ""     # paste your CDS key here, run once, then clear it again

with open("/root/.cdsapirc", "w") as f:
    f.write("url: https://cds.climate.copernicus.eu/api\n")
    f.write(f"key: {CDS_KEY}\n")
print("credentials written")

# %%
import cdsapi, os
os.makedirs(f"{DIR}/era5", exist_ok=True)
c = cdsapi.Client()

VARS = ["boundary_layer_height",
        "10m_u_component_of_wind", "10m_v_component_of_wind",
        "surface_pressure", "2m_temperature", "total_precipitation",
        "total_cloud_cover", "surface_solar_radiation_downwards"]

BOX = {"shanghai": [31.50, 121.25, 31.00, 121.75],
       "hongkong": [22.50, 114.00, 22.00, 114.50]}

def pull(city, year, months):
    out = f"{DIR}/era5/era5_{city}_{year}.nc"
    if os.path.exists(out):
        print(f"  exists: {out.split('/')[-1]}"); return
    c.retrieve("reanalysis-era5-single-levels", {
        "product_type": "reanalysis",
        "variable": VARS,
        "year": str(year),
        "month": months,
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": BOX[city],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }, out)
    print(f"  saved {out.split('/')[-1]}  {os.path.getsize(out)/1e6:.1f} MB")

# one test: Shanghai, December 2018 only
pull("shanghai", 2018, ["12"])

# %%
import cdsapi
c = cdsapi.Client(retry_max=1)     # fail fast instead of retrying

try:
    c.retrieve("reanalysis-era5-single-levels", {
        "product_type": "reanalysis",
        "variable": ["2m_temperature"],
        "year": "2018", "month": "12", "day": "01",
        "time": "12:00",
        "area": [31.50, 121.25, 31.00, 121.75],
        "data_format": "netcdf",
    }, "/content/test.nc")
    import os
    print("OK —", os.path.getsize("/content/test.nc"), "bytes")
except Exception as e:
    print(type(e).__name__, str(e)[:400])

# %%
!pip install cdsapi -q

CDS_KEY = ""     # paste your key, run, then clear it again

with open("/root/.cdsapirc", "w") as f:
    f.write("url: https://cds.climate.copernicus.eu/api\n")
    f.write(f"key: {CDS_KEY}\n")
print("credentials written")

# %%
import cdsapi, os
c = cdsapi.Client(retry_max=1)
try:
    c.retrieve("reanalysis-era5-single-levels", {
        "product_type": "reanalysis",
        "variable": ["2m_temperature"],
        "year": "2018", "month": "12", "day": "01",
        "time": "12:00",
        "area": [31.50, 121.25, 31.00, 121.75],
        "data_format": "netcdf",
    }, "/content/test.nc")
    print("OK —", os.path.getsize("/content/test.nc"), "bytes")
except Exception as e:
    print(type(e).__name__, str(e)[:400])

# %%
import time
print("=== A-code mapping (Eastmoney) ===")
try:
    raw = ak.stock_zh_ah_spot_em()
    print("ALIVE:", raw.shape)
except Exception as e:
    raw = None
    print("blocked:", type(e).__name__)

print("\n=== CDS ===")
try:
    import cdsapi, os
    c = cdsapi.Client(retry_max=1)
    c.retrieve("reanalysis-era5-single-levels", {
        "product_type": "reanalysis", "variable": ["2m_temperature"],
        "year": "2018", "month": "12", "day": "01", "time": "12:00",
        "area": [31.50, 121.25, 31.00, 121.75], "data_format": "netcdf",
    }, "/content/test.nc")
    print("ALIVE:", os.path.getsize("/content/test.nc"), "bytes")
except Exception as e:
    print("blocked:", type(e).__name__, str(e)[:120])

# %%
!pip install akshare cdsapi -q
from google.colab import drive
drive.mount('/content/drive')

import glob, numpy as np, pandas as pd, akshare as ak
DIR = "/content/drive/MyDrive/pollution-ah"

CN = pd.read_csv(f"{DIR}/cn_pm25_daily.csv", index_col="date", parse_dates=True)
H  = pd.read_csv(f"{DIR}/h_prices.csv", parse_dates=["date"],
                 dtype={"h_code": str})
ah = pd.read_csv(f"{DIR}/ah_h_codes.csv", dtype={"h_code": str})
print(f"CN {CN.shape}   H {H.h_code.nunique()} series   ah {len(ah)} codes")

try:
    raw = ak.stock_zh_ah_spot_em()
    print("\nA-code mapping ALIVE:", raw.shape)
except Exception as e:
    raw = None
    print("\nA-code mapping blocked:", type(e).__name__)

# %%
funcs = [f for f in dir(ak) if f.startswith("stock_info_")]
print("candidates:", funcs)

alist = None
for fn in ["stock_info_sh_name_code", "stock_info_sz_name_code",
           "stock_info_a_code_name"]:
    if not hasattr(ak, fn):
        continue
    try:
        r = getattr(ak, fn)()
        print(f"\n{fn}: OK  {r.shape}")
        print("  columns:", list(r.columns))
        print(r.head(3).to_string())
    except Exception as e:
        print(f"\n{fn}: {type(e).__name__} {str(e)[:100]}")

# %%
alist.to_csv(f"{DIR}/a_listing.csv", index=False)
print("saved A listing")

A = alist.copy()
A.columns = ["a_code", "name"]
A["a_code"] = A.a_code.astype(str).str.zfill(6)
A["key"] = A.name.str.replace(r"\s+", "", regex=True).str.replace("Ａ", "A")

ah["key"] = ah.name.str.replace(r"\s+", "", regex=True)

m = ah.merge(A[["a_code", "key"]], on="key", how="left")
ok = m.a_code.notna()
print(f"\nexact-name matches: {ok.sum()} of {len(m)}")
print(f"H codes matching >1 A code: {m[ok].h_code.duplicated().sum()}")

print("\nCCB (expect 601939):")
print(m[m.h_code == "00939"][["h_code","name","a_code"]].to_string())

have = set(H.h_code)
usable = m[ok & m.h_code.isin(have)]
print(f"\nmatched AND we have H prices: {len(usable)}")
print(f"  Shanghai A-codes (6xx): {usable.a_code.str.startswith('6').sum()}")
print(f"  Shenzhen A-codes (0/3): {(~usable.a_code.str.startswith('6')).sum()}")

print("\nunmatched, first 12:")
print(m[~ok][["h_code","name"]].head(12).to_string())

# %%
alist = ak.stock_info_a_code_name()
alist.to_csv(f"{DIR}/a_listing.csv", index=False)
print(f"saved {len(alist)} A-shares")

# %%
A = alist.copy()
A.columns = ["a_code", "name"]
A["a_code"] = A.a_code.astype(str).str.zfill(6)
A["key"] = A.name.str.replace(r"\s+", "", regex=True).str.replace("Ａ", "A")

ah["key"] = ah.name.str.replace(r"\s+", "", regex=True)

m = ah.merge(A[["a_code", "key"]], on="key", how="left")
ok = m.a_code.notna()
print(f"exact-name matches: {ok.sum()} of {len(m)}")
print(f"H codes matching >1 A code: {m[ok].h_code.duplicated().sum()}")

print("\nCCB (expect 601939):")
print(m[m.h_code == "00939"][["h_code","name","a_code"]].to_string())

have = set(H.h_code)
usable = m[ok & m.h_code.isin(have)]
print(f"\nmatched AND have H prices: {len(usable)}")
print(f"  Shanghai (6xx): {usable.a_code.str.startswith('6').sum()}")
print(f"  Shenzhen (0/3): {(~usable.a_code.str.startswith('6')).sum()}")

print("\nunmatched, first 12:")
print(m[~ok][["h_code","name"]].head(12).to_string())

# %%
def norm(s):
    s = s.str.replace(r"\s+", "", regex=True).str.replace("Ａ", "A")
    return s.str.replace(r"股份$", "", regex=True)

A["key2"] = norm(A.name)
ah["key2"] = norm(ah.name)

m2 = ah.merge(A[["a_code","key2"]], on="key2", how="left")
ok2 = m2.a_code.notna()
print(f"matches after stripping 股份: {ok2.sum()} of {len(m2)}  (was 157)")

dups = m2[ok2].h_code.duplicated(keep=False)
print(f"\nH codes with >1 A match ({m2[ok2][dups].h_code.nunique()} codes):")
print(m2[ok2][dups][["h_code","name","a_code"]].to_string())

have = set(H.h_code)
usable2 = m2[ok2 & m2.h_code.isin(have) & ~m2.h_code.duplicated(keep=False)]
print(f"\nunambiguous AND have H prices: {len(usable2)}")
print(f"  Shanghai: {usable2.a_code.str.startswith('6').sum()}")
print(f"  Shenzhen: {(~usable2.a_code.str.startswith('6')).sum()}")

print("\nstill unmatched:")
print(m2[~ok2])  # closing parenthesis added for the replication package; the notebook cell was truncated here

# %%
def norm(s):
    s = s.str.replace(r"\s+", "", regex=True).str.replace("Ａ", "A")
    return s.str.replace(r"股份$", "", regex=True)

A["key2"] = norm(A.name)
ah["key2"] = norm(ah.name)

m2 = ah.merge(A[["a_code", "key2"]], on="key2", how="left")
ok2 = m2.a_code.notna()
print(f"matches after stripping 股份: {ok2.sum()} of {len(m2)}  (was 157)")

dups = m2[ok2].h_code.duplicated(keep=False)
print(f"\nH codes with more than one A match: {m2[ok2][dups].h_code.nunique()}")
print(m2[ok2][dups][["h_code", "name", "a_code"]].to_string())

have = set(H.h_code)
usable2 = m2[ok2 & m2.h_code.isin(have) & ~m2.h_code.duplicated(keep=False)]
print(f"\nunambiguous and have H prices: {len(usable2)}")
print(f"  Shanghai: {usable2.a_code.str.startswith('6').sum()}")
print(f"  Shenzhen: {(~usable2.a_code.str.startswith('6')).sum()}")

print("\nstill unmatched:")
print(m2.loc[~ok2, ["h_code", "name"]].to_string())

# %%
print("ah rows:", len(ah), " unique h_code:", ah.h_code.nunique())
print("duplicated h_code rows:", ah.h_code.duplicated().sum())

s = ah.loc[ah.name.str.contains("马鞍山"), "name"]
print("\nraw repr:", [repr(x) for x in s.unique()])
print("after norm:", [repr(x) for x in norm(s).unique()])

hit = A[A.name.str.contains("马鞍山")]
print("\nA-listing candidates:")
print(hit[["a_code","name","key2"]].to_string())

# %%
ah = ah.drop_duplicates("h_code").reset_index(drop=True)
ah["key2"] = norm(ah.name)
print(f"ah now: {len(ah)} unique pairs")

m2 = ah.merge(A[["a_code","key2"]], on="key2", how="left")
ok2 = m2.a_code.notna()
print(f"name matches: {ok2.sum()} of {len(m2)}")

dups = m2[ok2].h_code.duplicated(keep=False)
print(f"ambiguous (one H, several A): {m2[ok2][dups].h_code.nunique()}")

have = set(H.h_code)
final = m2[ok2 & ~m2.h_code.duplicated(keep=False) & m2.h_code.isin(have)]
print(f"\nFINAL usable pairs: {len(final)}")
print(f"  Shanghai: {final.a_code.str.startswith('6').sum()}")
print(f"  Shenzhen: {(~final.a_code.str.startswith('6')).sum()}")

final[["h_code","name","a_code"]].to_csv(f"{DIR}/ah_pairs.csv", index=False)
print(f"\nsaved ah_pairs.csv")

# %%
OUTA = f"{DIR}/a_prices.csv"
rows, failed = [], []

for i, (hc, ac) in enumerate(zip(final.h_code, final.a_code), 1):
    try:
        d = ak.stock_zh_a_hist(symbol=ac, period="daily",
                               start_date="20130101", end_date="20201231",
                               adjust="hfq")          # back-adjusted for splits
        d = d[["日期","开盘","收盘","最高","最低","成交量"]]
        d.columns = ["date","open","close","high","low","volume"]
        d["date"] = pd.to_datetime(d.date)
        d["a_code"], d["h_code"] = ac, hc
        rows.append(d.drop_duplicates("date"))
    except Exception as e:
        failed.append((ac, type(e).__name__))
    if i % 20 == 0:
        pd.concat(rows).to_csv(OUTA, index=False)
        print(f"  {i}/{len(final)}  kept {len(rows)}  failed {len(failed)}")
    time.sleep(0.6)

Ap = pd.concat(rows, ignore_index=True).drop_duplicates(["a_code","date"])
Ap.to_csv(OUTA, index=False)
print(f"\nA series: {Ap.a_code.nunique()} of {len(final)}")
print(f"rows: {len(Ap):,}   {Ap.date.min().date()} to {Ap.date.max().date()}")
if failed: print(f"failed: {failed[:8]}")

# %%
import os
OUTA = f"{DIR}/a_prices.csv"
done = set()
if os.path.exists(OUTA):
    done = set(pd.read_csv(OUTA, dtype={"a_code": str}).a_code.unique())
    print(f"resuming: {len(done)} already saved")

rows = [pd.read_csv(OUTA, parse_dates=["date"], dtype={"a_code": str, "h_code": str})] if done else []
failed = []

for i, (hc, ac) in enumerate(zip(final.h_code, final.a_code), 1):
    if ac in done:
        continue
    try:
        d = ak.stock_zh_a_hist(symbol=ac, period="daily",
                               start_date="20130101", end_date="20201231",
                               adjust="hfq")
        d = d[["日期","开盘","收盘","最高","最低","成交量"]]
        d.columns = ["date","open","close","high","low","volume"]
        d["date"] = pd.to_datetime(d.date)
        d["a_code"], d["h_code"] = ac, hc
        rows.append(d.drop_duplicates("date"))
        pd.concat(rows, ignore_index=True).to_csv(OUTA, index=False)
    except Exception as e:
        failed.append(ac)
    if i % 10 == 0:
        got = pd.concat(rows).a_code.nunique() if rows else 0
        print(f"  {i}/{len(final)}  have {got}  failed {len(failed)}")
    time.sleep(5)

# %%
try:
    d = ak.stock_zh_a_daily(symbol="sh601939", start_date="20130101",
                            end_date="20201231", adjust="hfq")
    print("Sina route ALIVE:", len(d), "rows")
    print(d.head(3).to_string())
except Exception as e:
    print("Sina route:", type(e).__name__, str(e)[:150])

# %%
import os, time
OUTA = f"{DIR}/a_prices.csv"

rows, failed, done = [], [], set()
for i, (hc, ac) in enumerate(zip(final.h_code, final.a_code), 1):
    sym = ("sh" if ac.startswith("6") else "sz") + ac
    try:
        d = ak.stock_zh_a_daily(symbol=sym, start_date="20130101",
                                end_date="20201231", adjust="hfq")
        d = d[["date","open","high","low","close","volume"]].copy()
        d["date"] = pd.to_datetime(d.date)
        d["a_code"], d["h_code"] = ac, hc
        rows.append(d.drop_duplicates("date"))
        done.add(ac)
    except Exception as e:
        failed.append((sym, type(e).__name__))
    if i % 15 == 0:
        if rows: pd.concat(rows, ignore_index=True).to_csv(OUTA, index=False)
        print(f"  {i}/{len(final)}  have {len(done)}  failed {len(failed)}")
    time.sleep(1.0)

Ap = pd.concat(rows, ignore_index=True).drop_duplicates(["a_code","date"])
Ap.to_csv(OUTA, index=False)
print(f"\nA series: {Ap.a_code.nunique()} of {len(final)}")
print(f"rows: {len(Ap):,}   {Ap.date.min().date()} to {Ap.date.max().date()}")
n = Ap.groupby("a_code").size()
print(f"days per series: median {n.median():.0f}, min {n.min()}, max {n.max()}")
if failed: print(f"failed ({len(failed)}): {failed[:8]}")

# %%
h = H[H.h_code == "00939"].set_index("date").close
a = Ap[Ap.a_code == "601939"].set_index("date").close

print("CCB 2013-01-04:")
print(f"  H {h.get(pd.Timestamp('2013-01-04'), float('nan')):.2f} HKD")
print(f"  A {a.get(pd.Timestamp('2013-01-04'), float('nan')):.2f} (hfq)")
print("\nCCB 2020-12-31:")
print(f"  H {h.get(pd.Timestamp('2020-12-31'), float('nan')):.2f}")
print(f"  A {a.get(pd.Timestamp('2020-12-31'), float('nan')):.2f}")

# an unadjusted series shows a discrete drop on ex-dividend days;
# an adjusted one does not
r = pd.DataFrame({"h": h, "a": a}).dropna()
r["ratio"] = r.h / r.a
big = r.ratio.pct_change().abs().nlargest(8)
print("\n8 largest one-day moves in the H/A ratio:")
print(pd.DataFrame({"jump_pct": (100*big).round(1),
                    "ratio": r.ratio.reindex(big.index).round(3)}).to_string())

# %%
d = ak.stock_zh_ah_daily(symbol="00939", start_year="2013", end_year="2020",
                         adjust="hfq")
print(d.head(3).to_string())
print("\n2013-01-04 close:", d[d.iloc[:,0].astype(str) == "2013-01-04"].iloc[:,2].values)

# %%
OUTH2 = f"{DIR}/h_prices_hfq.csv"
rows, failed = [], []

for i, hc in enumerate(final.h_code, 1):
    try:
        d = ak.stock_zh_ah_daily(symbol=hc, start_year="2013",
                                 end_year="2020", adjust="hfq")
        d = d.iloc[:, :6]
        d.columns = ["date","open","close","high","low","volume"]
        d["date"] = pd.to_datetime(d.date)
        d = d[(d.date >= "2013-01-01") & (d.date <= "2020-12-31")]
        d = d.drop_duplicates("date")
        d["h_code"] = hc
        rows.append(d)
    except Exception as e:
        failed.append((hc, type(e).__name__))
    if i % 15 == 0:
        print(f"  {i}/{len(final)}  kept {len(rows)}  failed {len(failed)}")
    time.sleep(0.8)

H2 = pd.concat(rows, ignore_index=True).drop_duplicates(["h_code","date"])
H2.to_csv(OUTH2, index=False)
print(f"\nH hfq series: {H2.h_code.nunique()} of {len(final)}")
print(f"rows: {len(H2):,}   {H2.date.min().date()} to {H2.date.max().date()}")
n = H2.groupby("h_code").size()
print(f"days: median {n.median():.0f}, min {n.min()}, max {n.max()}")
if failed: print(f"failed: {failed[:8]}")

# %%
h2 = H2[H2.h_code == "00939"].set_index("date").close
a  = Ap[Ap.a_code == "601939"].set_index("date").close

r = pd.DataFrame({"h": h2, "a": a}).dropna()
r["ratio"] = r.h / r.a
print(f"CCB overlapping days: {len(r)}")
print(f"ratio: start {r.ratio.iloc[0]:.3f}  end {r.ratio.iloc[-1]:.3f}"
      f"  mean {r.ratio.mean():.3f}  sd {r.ratio.std():.3f}")

big = r.ratio.pct_change().abs().nlargest(8)
print("\n8 largest one-day ratio moves (was up to 10% before):")
print((100*big).round(1).to_string())

# the pair sample: firms with both sides
pairs = set(H2.h_code) & set(final[final.a_code.isin(Ap.a_code)].h_code)
print(f"\nfirms with BOTH sides: {len(pairs)}")
sub = final[final.h_code.isin(pairs)]
print(f"  Shanghai: {sub.a_code.str.startswith('6').sum()}")
print(f"  Shenzhen: {(~sub.a_code.str.startswith('6')).sum()}")

# %%
P = (H2.rename(columns={"close":"h_close"})[["date","h_code","h_close"]]
       .merge(final[["h_code","a_code"]], on="h_code")
       .merge(Ap.rename(columns={"close":"a_close"})[["date","a_code","a_close"]],
              on=["date","a_code"]))
P["ratio"] = P.h_close / P.a_close
P = P.sort_values(["h_code","date"])
P["dratio"] = P.groupby("h_code").ratio.pct_change()

daily = P.groupby("date").dratio.agg(["mean","std","count"])
top = daily["mean"].abs().nlargest(10)
print("dates with the largest AVERAGE ratio move across all firms:")
print(daily.loc[top.index].round(4).to_string())

print(f"\npanel: {P.h_code.nunique()} firms, {len(P):,} firm-days")
w = P[P.date.dt.month.isin([12,1,2])]
print(f"winter firm-days: {len(w):,}")

# %%
P.to_csv(f"{DIR}/ah_panel.csv", index=False)
print("saved ah_panel.csv:", len(P), "firm-days")

# %%
CN.index.name = "date"
sh = CN["Shanghai"].rename("pm_sh")
sz = CN["Shenzhen"].rename("pm_sz")
hk = hk.rename("pm_hk")          # from the earlier HK build

pol = pd.concat([sh, sz, hk], axis=1)
pol.index = pd.to_datetime(pol.index)

P["date"] = pd.to_datetime(P.date)
P["mkt"] = np.where(P.a_code.str.startswith("6"), "sh", "sz")

M = P.merge(pol, left_on="date", right_index=True, how="inner")
M = M[M.date.dt.month.isin([12, 1, 2])]

M["pm_cn"] = np.where(M.mkt == "sh", M.pm_sh, M.pm_sz)
M = M.dropna(subset=["pm_cn", "pm_hk", "dratio"])

print(f"ESTIMATION SAMPLE")
print(f"  firm-days: {len(M):,}")
print(f"  firms: {M.h_code.nunique()}   trading days: {M.date.nunique()}")
print(f"  Shanghai firms: {M[M.mkt=='sh'].h_code.nunique()}"
      f"   days: {M[M.mkt=='sh'].date.nunique()}")
print(f"  Shenzhen firms: {M[M.mkt=='sz'].h_code.nunique()}"
      f"   days: {M[M.mkt=='sz'].date.nunique()}")
print(f"\n  span: {M.date.min().date()} to {M.date.max().date()}")
print(f"  firm-days per winter:")
print(M.groupby(M.date.dt.year).size().to_string())

# %%
GENERAL = ["SHATIN","TSUEN WAN","EASTERN","KWUN TONG","TUEN MUN","TUNG CHUNG",
           "SHAM SHUI PO","YUEN LONG","CENTRAL/WESTERN","KWAI CHUNG",
           "TAP MUN","TSEUNG KWAN O","TAI PO"]

parts = []
for f in sorted(glob.glob(f"{DIR}/hk_pm25_*.csv")):
    d = pd.read_csv(f, skiprows=11, na_values=["N.A."], encoding="utf-8-sig")
    d.columns = [c.strip() for c in d.columns]
    parts.append(d)

Hk = pd.concat(parts, ignore_index=True)
Hk["ts"] = (pd.to_datetime(Hk.DATE, format="%d/%m/%Y")
            + pd.to_timedelta(Hk.HOUR.astype(int) - 1, unit="h"))
Hk = Hk.drop_duplicates("ts").set_index("ts").sort_index()

gen = Hk[GENERAL].apply(pd.to_numeric, errors="coerce")
hk = gen.mean(axis=1).where(gen.notna().sum(axis=1) >= 8).resample("D").mean().dropna()
hk.name = "pm_hk"
print(f"HK rebuilt: {len(hk)} days, mean {hk.mean():.1f}")

# %%
CN.index.name = "date"
sh = CN["Shanghai"].rename("pm_sh")
sz = CN["Shenzhen"].rename("pm_sz")
hk = hk.rename("pm_hk")          # from the earlier HK build

pol = pd.concat([sh, sz, hk], axis=1)
pol.index = pd.to_datetime(pol.index)

P["date"] = pd.to_datetime(P.date)
P["mkt"] = np.where(P.a_code.str.startswith("6"), "sh", "sz")

M = P.merge(pol, left_on="date", right_index=True, how="inner")
M = M[M.date.dt.month.isin([12, 1, 2])]

M["pm_cn"] = np.where(M.mkt == "sh", M.pm_sh, M.pm_sz)
M = M.dropna(subset=["pm_cn", "pm_hk", "dratio"])

print(f"ESTIMATION SAMPLE")
print(f"  firm-days: {len(M):,}")
print(f"  firms: {M.h_code.nunique()}   trading days: {M.date.nunique()}")
print(f"  Shanghai firms: {M[M.mkt=='sh'].h_code.nunique()}"
      f"   days: {M[M.mkt=='sh'].date.nunique()}")
print(f"  Shenzhen firms: {M[M.mkt=='sz'].h_code.nunique()}"
      f"   days: {M[M.mkt=='sz'].date.nunique()}")
print(f"\n  span: {M.date.min().date()} to {M.date.max().date()}")
print(f"  firm-days per winter:")
print(M.groupby(M.date.dt.year).size().to_string())

# %%
M.to_csv(f"{DIR}/estimation_sample.csv", index=False)
print("saved:", len(M), "firm-days")

# %%
import cdsapi, os
try:
    c = cdsapi.Client(retry_max=1)
    c.retrieve("reanalysis-era5-single-levels", {
        "product_type": "reanalysis", "variable": ["2m_temperature"],
        "year": "2018", "month": "12", "day": "01", "time": "12:00",
        "area": [31.50, 121.25, 31.00, 121.75], "data_format": "netcdf",
    }, "/content/test.nc")
    print("CDS ALIVE —", os.path.getsize("/content/test.nc"), "bytes")
except Exception as e:
    print("still down:", type(e).__name__, str(e)[:100])

# %%
!pip install cdsapi -q

CDS_KEY = ""     # paste key, run, then clear again
with open("/root/.cdsapirc", "w") as f:
    f.write("url: https://cds.climate.copernicus.eu/api\n")
    f.write(f"key: {CDS_KEY}\n")


# %%
import cdsapi, os
try:
    c = cdsapi.Client(retry_max=1)
    c.retrieve("reanalysis-era5-single-levels", {
        "product_type": "reanalysis", "variable": ["2m_temperature"],
        "year": "2018", "month": "12", "day": "01", "time": "12:00",
        "area": [31.50, 121.25, 31.00, 121.75], "data_format": "netcdf",
    }, "/content/test.nc")
    print("CDS ALIVE —", os.path.getsize("/content/test.nc"), "bytes")
except Exception as e:
    print("still down:", type(e).__name__, str(e)[:100])

# %%
import requests, pandas as pd

def om(lat, lon, name):
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
        "latitude": lat, "longitude": lon,
        "start_date": "2018-12-01", "end_date": "2018-12-31",
        "hourly": ",".join(["boundary_layer_height","wind_speed_10m",
                            "wind_direction_10m","surface_pressure",
                            "temperature_2m","precipitation",
                            "cloud_cover","shortwave_radiation"]),
        "timezone": "UTC"}, timeout=60)
    print(f"\n=== {name} === HTTP {r.status_code}")
    if r.status_code != 200:
        print(r.text[:300]); return None
    j = r.json()
    d = pd.DataFrame(j["hourly"])
    print("variables:", [c for c in d.columns if c != "time"])
    print(f"hours: {len(d)}")
    blh = d.get("boundary_layer_height")
    if blh is not None:
        print(f"BLH: mean {blh.mean():.0f}  min {blh.min():.0f}  max {blh.max():.0f}"
              f"  missing {blh.isna().sum()}")
    return d

sh = om(31.25, 121.50, "Shanghai")
hk = om(22.25, 114.25, "Hong Kong")

# %%
import numpy as np, time

def pull_city(lat, lon, tag):
    frames = []
    for y0, y1 in [(2013,2014),(2014,2015),(2015,2016),(2016,2017),
                   (2017,2018),(2018,2019),(2019,2020)]:
        r = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
            "latitude": lat, "longitude": lon,
            "start_date": f"{y0}-12-01", "end_date": f"{y1}-02-29",
            "hourly": ",".join(["boundary_layer_height","wind_speed_10m",
                                "wind_direction_10m","surface_pressure",
                                "temperature_2m","precipitation",
                                "cloud_cover","shortwave_radiation"]),
            "timezone": "UTC"}, timeout=90)
        d = pd.DataFrame(r.json()["hourly"])
        frames.append(d)
        print(f"  {tag} {y0}-{y1}: {len(d)} hours")
        time.sleep(1)

    M = pd.concat(frames, ignore_index=True).drop_duplicates("time")
    M["time"] = pd.to_datetime(M.time, utc=True)

    # wind speed/direction -> u,v  (direction is where the wind comes FROM)
    th = np.radians(M.wind_direction_10m)
    M["u"] = -M.wind_speed_10m * np.sin(th)
    M["v"] = -M.wind_speed_10m * np.cos(th)
    M["city"] = tag
    return M

SH = pull_city(31.25, 121.50, "shanghai")
HK = pull_city(22.25, 114.25, "hongkong")

MET = pd.concat([SH, HK], ignore_index=True)
MET.to_csv(f"{DIR}/met_hourly.csv", index=False)
print(f"\nsaved {len(MET):,} city-hours")
print(f"span {MET.time.min().date()} to {MET.time.max().date()}")
print(MET.groupby("city").size().to_string())

# %%
from google.colab import drive
drive.mount('/content/drive')
DIR = "/content/drive/MyDrive/pollution-ah"

MET.to_csv(f"{DIR}/met_hourly.csv", index=False)
print(f"saved {len(MET):,} city-hours")
print(f"span {MET.time.min().date()} to {MET.time.max().date()}")
print(MET.groupby("city").size().to_string())
