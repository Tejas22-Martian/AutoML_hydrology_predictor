# Data

This project uses the **CAMELS-IND** dataset. The full dataset (~450 MB) is **not**
bundled here — it is openly available for academic/research use and must be cited.
A small **sample** is included so the pipeline runs out of the box.

## Sample (included)

`data/sample/` contains 8 catchments (2 from each of the four focus basins) with
the same layout the loader expects:

```
data/sample/
├── catchment_mean_forcings/{id}.csv     # daily meteorological forcings
├── streamflow_timeseries/streamflow_observed.csv
├── attributes_csv/camels_ind_*.csv      # static catchment attributes
└── Disclaimer.txt
```

Run inference with the shipped fine-tuned model on the sample:

```bash
python predict.py --catchment 3002 --config configs/config_sample.yaml --output predictions.csv
```

Smoke-test training on the sample (tiny, fast):

```bash
python main.py --config configs/config_sample.yaml --phase train
```

> ⚠️ `configs/config_sample.yaml` writes to `outputs/models/`, which would
> **overwrite the shipped model bundle**. Back up `outputs/models/` first, or set
> `output.*_dir` to a different folder, if you want to keep the shipped result.
> The 8-catchment sample is for smoke-testing only — it does **not** reproduce the
> published results, which require the full dataset.

## Full dataset (download)

1. Obtain CAMELS-IND from the official source (the *Streamflow_Sufficient*
   release = 242 catchments used here):
   - Data description / DOI: search **"CAMELS-IND Mangukiya Sharma"** or see the
     `CAMELS_IND_Data_Description.pdf` distributed with the dataset.
2. Place it next to the repo (or anywhere) and point `configs/config.yaml`
   `data:` paths at it:

```yaml
data:
  fosee_root: ../FOSEE/CAMELS_IND_Catchments_Streamflow_Sufficient
  forcings_dir: ${fosee_root}/catchment_mean_forcings
  streamflow_file: ${fosee_root}/streamflow_timeseries/streamflow_observed.csv
  attributes_dir: ${fosee_root}/attributes_csv
  shapefile: ${fosee_root}/shapefiles_catchment/catchments.shp
```

3. Run the full pipeline:

```bash
python main.py --config configs/config.yaml
```

## Citation

When using CAMELS-IND you **must cite** the data description article (see the
dataset's `CAMELS_IND_Data_Description.pdf`). Per the dataset disclaimer, the data
is provided "as is" for academic/research use and all interpretations are the
user's own.

Dataset contacts: Nikunj K. Mangukiya, Ashutosh Sharma (IIT Roorkee).
