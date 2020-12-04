# Author: Steefan Contractor
# Date created: 09 Sept 2020
# Description: Script imports gappy mooring timeseries, bins it to daily temperatures, and fits successively longer predicting LSTM models to fill in the gaps

# import packages
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
f'Tensorflow version: {tf.__version__}'
import xarray as xr
import os
import pickle

# open raw netcdf file
temp = xr.open_dataset('/srv/scratch/z3289454/OceanDataScience/Mooring/PH100/IMOS_ANMN-NSW_TZ_20091029_PH100_FV01_TEMP-aggregated-timeseries_END-20190612_C-20190819.nc')

# bin data at various pressure depths
bins = [0,11,19,20,28,106,114,116]
labs = [5.5,15,19.5,24,(106-28)/2,110,115]
# create pd dataframe from scratch
data  = {'TIME': temp.TIME.values,
         'TEMP': temp.TEMP.values,
         'PRES': temp.PRES_REL.values}

temp_df= pd.DataFrame(data, columns = ['TIME','TEMP','PRES'])
# set TIME as the indexing variable
temp_df = temp_df.set_index('TIME')
# create a column with pressure bin labels
temp_df['PRES_BIN'] = pd.cut(temp_df.PRES, bins = bins, labels = labs, include_lowest=True)
# First groupby PRES_BIN column, then resample each group on hourly intervals and mean each resulting bin
# drop unncessary columns and nan rows after
temp_df = temp_df.groupby('PRES_BIN').resample('1D').mean().drop(columns=['PRES']).dropna()
# get 15m bin cross-section
temp_df_15 = temp_df.xs(15)
# create a gappy dataframe with gaps represented as nans
temp_df_15_wgaps = temp_df_15.copy()
temp_df_15_wgaps = temp_df_15_wgaps.asfreq(freq='1D')
