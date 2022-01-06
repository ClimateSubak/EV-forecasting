# EV-forecasting

Electric vehicle uptake in MSOAs of England and Wales

Writeup and code by [Victoria Pereira](http://people.maths.ox.ac.uk/pereira/) during a [Faculty Data Science Fellowship](https://faculty.ai/fellowship-fellows/) with [Climate Subak](https://subak.org)

In this project we connect disparate datasets relevant to Electric Vehicle (EV) uptake in the UK. In particular, we develop a forecast of EV uptake in the UK driven by socioeconomic and energy factors.

**Abbrevations**:  
OA - Output Area  
LA - Local Authority  
LSOA - Lower layer super output areas  
MSOA  - Middle layer super output areas  
EV - Electric vehicle  

The processed datasets for modelling are available here: https://figshare.com/articles/dataset/MSOA_evcount/14995020

## Data

|Name|Description|Link|
|--|--|--|
|D1|Household income|https://www.ons.gov.uk/employmentandlabourmarket/peopleinwork/earningsandworkinghours/datasets/smallareaincomeestimatesformiddlelayersuperoutputareasenglandandwales|
|D2|House price|https://www.ons.gov.uk/peoplepopulationandcommunity/housing/datasets/medianpricepaidbylowerlayersuperoutputareahpssadataset46|
|D3|Rural-urban classification (RUC)|https://data.gov.uk/dataset/b1165cea-2655-4cf7-bf22-dfbd3cdeb242/rural-urban-classification-2011-of-lower-layer-super-output-areas-in-england-and-wales|
|D4|Index of multiple deprivation (IMD)|https://data-communities.opendata.arcgis.com/datasets/d4b79be994ac4820ad44e10ded313df3_0/explore?location=52.854107%2C-2.489783%2C6.81|
|D5|Electricity consumption|https://www.gov.uk/government/statistics/lower-and-middle-super-output-areas-electricity-consumption|
|D6|PV solar panel count through feed-in tariff installations |https://www.ofgem.gov.uk/publications-and-updates/feed-tariff-installation-report-31-march-2021|
|D7|Public chargers |https://www.gov.uk/government/statistics/electric-vehicle-charging-device-statistics-april-2021|
|D8|Government grants for private chargers|https://www.gov.uk/government/statistics/electric-vehicle-charging-device-grant-scheme-statistics-april-2021|

### Regional datasets
|Name|Description|Link
|--|--|--|
|R1|OA to LSOA to MSOA |https://data.gov.uk/dataset/ec39697d-e7f4-4419-a146-0b9c9c15ee06/output-area-to-lsoa-to-msoa-to-local-authority-district-december-2017-lookup-with-area-classifications-in-great-britain|
|R2|LSOA boundaries: (fully clipped 2011) |https://data.gov.uk/dataset/fa883558-22fb-4a1a-8529-cffdee47d500/lower-layer-super-output-area-lsoa-boundaries|
|R3|MSOA boundaries: (fully clipped 2011) |https://data.gov.uk/dataset/2cf1f346-2f74-4c06-bd4b-30d7e4df5ae7/middle-layer-super-output-area-msoa-boundaries|


We also used vehicle data provided by NewAutomotive (https://newautomotive.org/) for MOT test and new car registrations in England and Wales.

## Notebooks
| Stage          | Name | Description                                                                                                                                                                                                                          |
|----------------|------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Preprocess | P1   | Processes raw income data (D1): import 2014, 2016, 2018 datasets. Calculate monthly and annual income for 2014 from weekly averages. Clean data, and interpolate missing years through averaging.                                    |
| Preprocess | P2   | Load IMD and RUC LSOA data (D3 and D4), and map from LSOA to MSOA.                                                                                                                                                                   |
| Preprocess | P3   | Load LSOA electricity consumption data (D5), select only England and Wales LSOA regions (34753), and check that the dataframes for 2010-2019 are all the same size.                                                                  |
| Preprocess | P4   | Load the public chargers (D6) and private chargers (D7) for LAs. Reduce to LAs that are contained in LSOA regional data (R1). Find missing LAs and fill missing values with zeros, assuming that there are no chargers in these LAs. |
| Preprocess | P5   | Load FIT data (D6) and map to LSOA and months.                                                                                                                                                                                       |
| Preprocess | P6   | Load and process houseprice data (D2) and impute missing values with the mean of all numeric values present for that year.                                                                                                           |
| Preprocess | P7   | Merge the processed MSOA files into a single dataframe of steady features for classification.                                                                                                                                        |
| Preprocess | P8   | Merging the processed MSOA files into a single multi-indexed dataframe for time-dependent data for forecasting.                                                                                                                      |
| Analysis       | A1   | Initial exploratory data analysis of the steady features on the MSOA granularity.                                                                                                                                                    |
| Functions      | F1   | Functions to plot forecasting predictions for single MSOA and distribution of EV count.                                                                                                                                              |
| Functions      | F2   | Functions to split data into test/train on temporal or spatial or both.                                                                                                                                                              |
| Modelling      | M1   | Classification of EV present in steady dataset at 04-2021.                                                                                                                                                                           |
| Modelling      | M2   | Classification model wrapped with Shapley package.                                                                                                                                                                                   |
| Modelling      | M3   | XGBoost forecasting model for forecasting EV count.                                                                                                                                                                                  |


## Licensing

All of the ONS and Government datasets are shared under the [Open Government Licence](http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/)

