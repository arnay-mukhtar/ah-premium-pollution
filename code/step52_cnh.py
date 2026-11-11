# ============================================================
# Step 52: the offshore CNH robustness row (Yahoo had no CNH history)
# Run in the same Colab session right after step 51: it reuses run(),
# window() and the data already loaded there. Tries Eastmoney via
# akshare first, then several Yahoo tickers. Saves hkd_cnh_daily.csv.
# ============================================================
import os, numpy as np, pandas as pd
try:
    run, window, base
except NameError:
    raise SystemExit('Run step 51 first in this session (this cell reuses its functions).')
DATA = '/content/drive/MyDrive/bilateral/data'


def pick(df, keys):
    for c in df.columns:
        if any(k in str(c).lower() for k in keys):
            return c
    return None


def ak_series(sym):
    try:
        import akshare as ak
    except ImportError:
        os.system('pip -q install akshare')
        import akshare as ak
    df = ak.forex_hist_em(symbol=sym)
    dc, cc = pick(df, ['日期', 'date']), pick(df, ['收盘', '最新价', 'close'])
    s = pd.Series(pd.to_numeric(df[cc], errors='coerce').values,
                  index=pd.to_datetime(df[dc], errors='coerce'), name=sym).dropna()
    print('   akshare %s: %d days, %s to %s' % (sym, len(s), s.index.min().date(), s.index.max().date()))
    return s


def yf_series(tk):
    import yfinance as yf
    h = yf.download(tk, start='2014-01-01', end='2020-12-31', progress=False, auto_adjust=False)
    cl = h['Close']
    cl = cl.iloc[:, 0] if isinstance(cl, pd.DataFrame) else cl
    cl.index = pd.to_datetime(cl.index).tz_localize(None).normalize()
    print('   yahoo %s: %d days' % (tk, len(cl.dropna())))
    return cl.dropna()


fx = None
for name, f in [('akshare HKDCNH', lambda: ak_series('HKDCNH')),
                ('akshare USDCNH / USDHKD', lambda: ak_series('USDCNH') / ak_series('USDHKD')),
                ('yahoo HKDCNH=X', lambda: yf_series('HKDCNH=X')),
                ('yahoo USDCNH=X / HKD=X', lambda: yf_series('USDCNH=X') / yf_series('HKD=X')),
                ('yahoo CNH=X / HKD=X', lambda: yf_series('CNH=X') / yf_series('HKD=X'))]:
    try:
        s = f().dropna()
        s = s[(s.index >= '2014-01-01') & (s.index <= '2020-12-31')]
        if len(s) > 1000 and 0.7 < s.median() < 1.0:
            fx, src = s, name
            break
        print('   %s: unusable (%d days, median %.3f)' % (name, len(s), s.median() if len(s) else np.nan))
    except Exception as e:
        print('   %s: failed (%s)' % (name, str(e)[:80]))
if fx is None:
    raise SystemExit('No CNH source worked. Paste this output back.')
fx = fx.rename('hkd_cnh').rename_axis('date').reset_index()
fx.to_csv(f'{DATA}/hkd_cnh_daily.csv', index=False)
print('CNH source: %s, %d days, median %.4f CNH per HKD' % (src, len(fx), fx['hkd_cnh'].median()))

P = pd.read_csv(f'{DATA}/ah_premium_panel.csv', parse_dates=['date'], dtype={'h_code': str})
P['h_code'] = P['h_code'].str.zfill(5)
d = pd.DataFrame({'date': sorted(P['date'].unique())}).merge(fx, on='date', how='left')
d['hkd_cnh'] = d['hkd_cnh'].ffill()
P = P.merge(d, on='date', how='left').sort_values(['h_code', 'date'])
P['d_prem_cnh'] = (np.log(P['a_raw']) - np.log(P['h_raw'] * P['hkd_cnh'])).groupby(P['h_code']).diff()
t = base.merge(P[['h_code', 'date', 'd_prem_cnh']], on=['h_code', 'date'], how='inner')
print('correlation of daily changes, CNH vs parity premium: %.4f' % t['d_prem_cnh'].corr(t['d_prem_raw']))
run('Offshore CNH conversion', t, outcome='d_prem_cnh')
print('\nDone. Paste everything above back.')
