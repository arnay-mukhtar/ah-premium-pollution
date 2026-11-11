# Step 53: primary specification with the H-leg filter at 0.25 (the registration
# document's threshold). Run after step 51 in the same session (reuses run() and base).
t = base[base['d_logH'].abs() <= 0.25]
print('firm-days dropped:', len(base) - len(t))
run('H-leg filter at 0.25 (registration document)', t)
