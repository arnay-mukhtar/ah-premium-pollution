# Air Pollution and the A-H Premium: replication code

**Paper:** Arnay Mukhtar (2026), *Air Pollution and the A-H Premium: Pre-Registered Evidence from Shanghai–Hong Kong Dual Listings and the Limits of Meteorological Instruments.* Working paper, SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7689778
**Pre-registration:** https://osf.io/px4r6, DOI [10.17605/OSF.IO/BG3ZE](https://doi.org/10.17605/OSF.IO/BG3ZE) (filed 5 November 2026, before any outcome was estimated)
**Archived package with data:** Zenodo, DOI [10.5281/zenodo.24046754](https://doi.org/10.5281/zenodo.24046754)
**Author:** Arnay Mukhtar · mukhtarovarnay@gmail.com · ORCID [0009-0003-5546-0418](https://orcid.org/0009-0003-5546-0418)

A one standard deviation rise in Shanghai PM2.5, instrumented with dispersion conditions, moves the A-H premium by 0.73 basis points (95% Anderson-Rubin set −15.5 to +18.1), and a pre-registered equivalence test rejects effects above 25 basis points. The same instruments fail for Hong Kong, where the first-stage sign inverts in winter.

This repository holds every script that produced a number, table or figure in the paper, the figures, and the original notebooks. The derived datasets are on Zenodo (see `data/README.md`); the Zenodo record is the citable archive.

---

## 1. Contents

| Folder | What it holds |
|---|---|
| `code/` | One script per analysis step, numbered in the order they were run. For steps 01–46 the first line names the notebook cell it came from; steps 47–53 and `fig5_falsification.py` were written as standalone scripts after the pre-registration checks. |
| `code/ar_inference.py` | Anderson-Rubin confidence sets with Driscoll-Kraay errors. Imported by steps 27, 30, 32, 33 and 34. Running it directly executes the simulation self-tests reported in Sections 8.3 and 8.4. |
| `data/` | Empty here: download `replication_data.zip` from Zenodo into it (see Section 4). |
| `figures/` | The five figures as vector PDFs. |
| `notebooks/` | The two original Colab notebooks, including their outputs, with one API key removed. |

## 2. How to run

The analysis was run on Google Colab (free tier). Every script reads and writes under

```
/content/drive/MyDrive/bilateral/data         derived data
/content/drive/MyDrive/pollution-ah           A/H prices and raw pollution files
/content/drive/MyDrive/bilateral/figures      figures
```

To reproduce: download `replication_data.zip` from Zenodo, unzip it to `MyDrive/bilateral/data`, then run the scripts in numeric order in Colab. To run elsewhere, change the `DATA` (and where present `DIR`, `PROJECT`, `OUT`) constants at the top of each script.

Because the derived data are included, any step can be run on its own. The main result needs only `step27` and `ar_inference.py`.

`pip install -r requirements.txt` covers all dependencies. `cdsapi` and `akshare` are needed only to re-download raw inputs.

**Credentials.** Step 02 downloads ERA5 boundary-layer height from the Copernicus Climate Data Store. Set your own key as the environment variable `CDS_KEY` before running it.

## 3. Pipeline, and which script produces which result

### Data construction
| Step | Purpose |
|---|---|
| 01 | A- and H-share prices, A-H pair list, mainland and Hong Kong PM2.5 (notebook `01_build_pollution`, all cells in order) |
| 02 | Meteorology downloads: ERA5 boundary layer (CDS) and Open-Meteo hourly weather; Hong Kong EPD parsing |
| 04–13 | First-stage development: early instrument and pollution-series diagnostics, Hong Kong boundary-layer validation, inversion measures |
| 14 | HKD/CNY central parity |
| 15–17 | Winter against full year; Ministry city values against the Zenodo station series |
| 18 | Full-year mainland PM2.5 (Ministry city values via quotsoft) |
| 19 | The registered Shanghai first stage → `shanghai_first_stage_fullyear.csv` |
| 20–23 | A-H premium construction and the four cleaning rules → `ah_premium_clean.csv`, `split_windows.csv` |
| 24–25 | Ürümqi and Beijing placebo panels → `placebo_city_panel.csv`, `three_city_panel.csv` |

### Paper element → script
| Paper | Script |
|---|---|
| Table 1, descriptive statistics | `step32` |
| §5 first stage (effective F 106.89, partial R² 0.2490) | `step19` |
| §5 full first-stage table | `step40` |
| §5 pollution-source and winter comparisons (0.265 against 0.117; 0.1845 against 0.2490) | `step15`–`step19` |
| §6 dispersion-measures table, first eight rows | `step43` |
| §6 dispersion-measures table, last two rows (parcel-method and bulk Richardson depths) | `step46` |
| §6.2 ERA5 against radiosondes | `step05` |
| §6.4 seasonal split; §6.5 thinness table | `step45`, `step46` |
| §6.8 365-city screen and latitude table | `step35b`, `step36b`, `step42` |
| §6.8 coordinate validation | `step38` |
| §6.8 Hong Kong at the 97th percentile | `step43` |
| §7.1 primary estimate, Wald and Anderson-Rubin sets, overidentification | `step27` |
| §7.2 OLS against 2SLS | `step40` |
| §7.4 decomposition into A leg, H leg and FX | `step27` |
| §8.1 falsification family as registered (placebo cities alongside Shanghai) and six-test Romano-Wolf p-values | `step47`, `step50` |
| §8.1 secondary placebo cities (Ordos, Zhangjiakou, Yangquan) | `step49` |
| §8.2 trading-day leads | `step50` |
| §8.3 serial correlation and the standard errors | `step26`, `ar_inference.py` self-test |
| §8.4 Anderson-Rubin against Wald coverage | `ar_inference.py` self-test |
| §8.5 secondary instrument set | `step32`, `step33` |
| §8.6 registered robustness table (Table 20) | `step47`; `step49` (Hong Kong reduced form, cap-weighted premium); `step51` (Zenodo series, winter days); `step52` (offshore CNH conversion) |
| §8.8 Deviations: placebo cities entered alone, as in the deposited code (item 1); calendar-day leads (item 2) | `step27`, `step28` |
| §8.8 Deviations: eight-test Romano-Wolf family (item 9) | `step48` |
| §8.8 Deviations: H-leg filter at 0.25 (item 10) | `step53` |
| §9 within-day announcement test (H7) | `step29` (dates), `step30`, `step31` (exploratory follow-ups); `step50` (date coverage, 60 of 61 firms) |

### Figures
| Paper figure | Script | File |
|---|---|---|
| Figure 1, first stage | `step34` | `fig1_first_stage` |
| Figure 2, city screen map | `step42` | `fig4_city_screen` |
| Figure 3, distribution of first-stage coefficients | `step42` | `fig5_blh_distribution` |
| Figure 4, Anderson-Rubin curve | `step34` | `fig2_ar_curve` |
| Figure 5, falsification family | `fig5_falsification.py` | `fig3_falsification` |

`fig3_falsification.pdf` is the regenerated Figure 5 with the registered joint placebos; the earlier version made by `step34` is superseded. The file names follow the order the figures were made; the paper numbers them in reading order. `step34` labels the Anderson-Rubin lower bound in `fig2_ar_curve` as −15.6, a rounding of its plotting grid; the estimate is −15.52, and the PDF in `figures/` has that label corrected to −15.5 to match the paper.

### Steps with no script here
- **35** is superseded by **35b**. It built a six-year meteorology cache at city coordinates later found to be wrong, and that cache is not included.
- **37** was a cross-check of two geocoders, abandoned as a method because the second geocoder was less reliable than the first.
- **39** re-geocoded every city as "city, province, China". Its script was not preserved, but its outputs are included (`city_coords.csv`, `city_coords_province.csv`), and every later step reads those files. Step 38 validates the resulting coordinates against the pollution data itself.
- No reported result depends on an output of a step numbered 41.
- **51** was written to rebuild the Zenodo station series for the full year. The Zenodo archive (record 6404603) holds files for December to February only, so the row runs on the 292 winter days it covers (paper, §8.6 and Deviation 3).
- **52** and **53** reuse functions defined in **51** and are meant to run in the same Colab session.

## 4. Data

### In the Zenodo archive (`replication_data.zip`, unzip into `data/`)
Derived datasets produced by the scripts above, plus public inputs: daily and hourly PM2.5 (Ministry of Ecology and Environment city values, Hong Kong EPD stations), ERA5 and Open-Meteo meteorology, IGRA radiosonde measures, HKD/CNY central parity, earnings announcement dates, city coordinates and the city-screen results.

### Excluded
- **Raw A- and H-share prices.** They come from exchange data redistributed by third parties, which this package does not relicense. Steps 01 and 20 re-download them through `akshare`. The premium panels are included with price-level columns removed; daily log changes and premia are kept.
- **The stale `city_meteo` cache** described under step 35.

### Data sources
| Source | Used for | Terms |
|---|---|---|
| Ministry of Ecology and Environment city PM2.5, via quotsoft.net/air | Mainland pollution, city screen | Public data; cite the Ministry |
| Hong Kong Environmental Protection Department | Hong Kong PM2.5 | Use with acknowledgement of the EPD |
| ERA5 reanalysis (Copernicus Climate Change Service) | Boundary-layer height | Copernicus licence; attribution required |
| Open-Meteo (ERA5-based) | Hourly weather and city meteorology | CC BY 4.0 |
| NOAA IGRA v2.2 | Radiosonde stability and depth measures | Public domain (US government) |
| People's Bank of China | HKD/CNY central parity | Public |
| EastMoney via akshare | Earnings announcement dates; A/H prices (not redistributed) | Third-party |

## 5. Software
Python 3.10 or later; see `requirements.txt`. Anderson-Rubin inference and the Driscoll-Kraay estimator are implemented directly in NumPy and SciPy (`ar_inference.py`), with no dependency on an econometrics package.

## 6. Citation
Please cite the paper and the archived package (DOI 10.5281/zenodo.24046754). `CITATION.cff` gives both.

## 7. Licence
Code: MIT. Derived data: CC BY 4.0, subject to the terms of the original sources. See `LICENSE`.
