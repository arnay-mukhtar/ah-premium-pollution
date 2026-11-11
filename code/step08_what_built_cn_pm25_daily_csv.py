# Source: paper2_meteorology_build.ipynb, code cell 20 (final version of step 8 as run).
# ============================================================
# Step 8 — what built cn_pm25_daily.csv?
#
# Instant. Prints the code of 01_build_pollution.ipynb so the November
# and December rebuild can be written against the real source rather
# than a guess.
# ============================================================

import json
import os

NB = '/content/drive/MyDrive/pollution-ah/01_build_pollution.ipynb'

nb = json.load(open(NB, encoding='utf-8'))
cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
print('%d code cells\n' % len(cells))

for i, c in enumerate(cells):
    src = ''.join(c['source']).rstrip()
    if not src.strip():
        continue
    print('=' * 68)
    print('CELL %d' % i)
    print('=' * 68)
    # Long cells truncated in the middle; the head and tail carry the
    # source URLs and the write targets, which is what matters here.
    lines = src.splitlines()
    if len(lines) > 60:
        print('\n'.join(lines[:40]))
        print('   ... [%d lines omitted] ...' % (len(lines) - 60))
        print('\n'.join(lines[-20:]))
    else:
        print(src)
    print()
