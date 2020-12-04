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
temp = xr.open_dataset('/srv/scratch/z3289452/OceanDataScience/Mooring/PH100/IMOS_ANMN-NSW_TZ_20091029_PH100_FV01_TEMP-aggregated-timeseries_END-20190612_C-20190819.nc')

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


# Model 1 to fill in gaps up to 10 days long
hist_len = 30
targ_len = 10

Time_diff = temp_df_15.index.to_series().diff()
break_index = np.where(Time_diff > pd.Timedelta(days=1))[0] - 1
break_index = np.append(break_index, len(temp_df_15))
window_len = hist_len + targ_len

data = []
labels = []
break_index = np.where(Time_diff > pd.Timedelta(days=1))[0] - 1 
window_len = hist_len + targ_len
for w in range(len(break_index)):
    run_start = hist_len if w == 0 else break_index[w-1] + 1 + hist_len
    run_end = break_index[w] - targ_len   
    for i in range(run_start, run_end):
        indices = range(i-hist_len, i)
        data.append(np.reshape(temp_df_15.values[indices], (hist_len,1)))
        labels.append(temp_df_15.values[i:i+targ_len])        
data = np.array(data)
labels = np.array(labels)
# Split into training and test datasets
trainidx = np.random.choice(len(data), int(np.round(0.9*len(data))), replace=False)
train_data = data[trainidx]
train_labels = labels[trainidx]
train_labels = train_labels.reshape((len(train_labels),targ_len))
val_data = np.delete(data, obj=trainidx, axis=0)
val_labels = np.delete(labels, trainidx, axis=0)
val_labels = val_labels.reshape((len(val_labels), targ_len))
# Normalisation
train_mean = train_data.mean()
train_std = train_data.std()
# save training, validation and normalisation data
pickle_out = open("../data/train_mean-std_train-val_data-labels_hist30_targ10.pickle", "wb")
pickle.dump([train_mean, train_std, train_data, train_labels, val_data, val_labels], pickle_out)
pickle_out.close()
train_data = (train_data-train_mean)/train_std
train_labels = (train_labels-train_mean)/train_std
val_data = (val_data-train_mean)/train_std
val_labels = (val_labels-train_mean)/train_std

# Model 1 setup

class descaled_mape(keras.losses.Loss):
    """
    A loss/metric that (de)scales true and predicted values into absolute units before calculating mean absolute percentage error (mape).
    Args:
        mu: mean (usually training data mean)
        sd: standard dev (usually based on training data)
        reduction: Type of tf.keras.losses.Reduction to apply to loss.
        name: name of loss function
    """
    def __init__(self, mu, sd, reduction=keras.losses.Reduction.AUTO, name='descaled_mape'):
        super().__init__(reduction=reduction, name=name)
        self.mu=mu
        self.sd=sd
    
    def call(self, y_true, y_pred):
        y_true = y_true * self.sd + self.mu
        y_pred = y_pred * self.sd + self.mu
        return tf.math.reduce_mean(tf.abs((y_true - y_pred)/y_true))

BATCH_SIZE = 8
BUFFER_SIZE = data.shape[0]
train_univariate = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
val_univariate = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
val_univariate = val_univariate.batch(BATCH_SIZE).repeat()
bidirectional_lstm_targ10_model = tf.keras.models.Sequential([
    layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True, input_shape=train_data.shape[-2:])),
    layers.Bidirectional(tf.keras.layers.LSTM(32, activation='relu', return_sequences=True,)),
    layers.Bidirectional(tf.keras.layers.LSTM(8, activation='relu')),
    tf.keras.layers.Dense(10)
])
bidirectional_lstm_targ10_model.compile(optimizer='adam', loss=descaled_mape(mu=train_mean, sd=train_std), metrics=[descaled_mape(mu=train_mean, sd=train_std), 'mae', 'mse', keras.metrics.MeanAbsolutePercentageError()])
# checkpoints to save training progress
checkpoint_path = "../data/bidirectional_lstm_daily_hist30_targ10_loss-descaledmape/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
# callbacks
# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 10 == 0: print('loss: {:7.4f}, val_loss: {:7.4f}' .format(logs['loss'], logs['val_loss']))
    print('.', end='')
    # Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=0)
# early stopping callback
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

# Model 1 Training
STEPS = int(train_data.shape[0]/BATCH_SIZE)
VAL_STEPS = int(val_data.shape[0]/BATCH_SIZE)
EPOCHS = 500

history = bidirectional_lstm_targ10_model.fit(train_univariate, epochs=EPOCHS,
                      steps_per_epoch=STEPS,
                      validation_data=val_univariate, validation_steps=VAL_STEPS, verbose=0,
                 callbacks=[cp_callback, PrintDot(), es_callback])
# save history and model
pickle_out = open("../data/temp_gapfilling_daily_bidirectional_lstm_model_hist30_targ10_loss-descaledmape_earlystopping_history.pickle", "wb")
pickle.dump(pd.DataFrame(history.history), pickle_out)
pickle_out.close()
## Save model
bidirectional_lstm_targ10_model.save('../data/saved_models/temp_gapfilling_daily_bidirectional_lstm_model_hist30_targ10_loss-descaledmape.h5')

# Fill gaps with Model 1 predictions
def gap_loc_and_len(sample):
    tmp = 0
    loc = []
    leng = []
    for i in range(len(sample)):
        current=sample[i]
        if not(np.isnan(current)) and tmp>0:
            leng.append(tmp)
            tmp=0
        if np.isnan(current):
            if (tmp==0) : loc.append(i)
            tmp=tmp+1
    df = pd.DataFrame({'location': loc,
                       'length': leng})
    return(df)

def gapfill(samp, lstm_model, train_mean, train_std, hist_len, pred_len, gaps = pd.DataFrame()):
    if gaps.empty: gaps = gap_loc_and_len(samp.TEMP)
    for i in range(len(gaps)):
        gap_start = gaps.location[i]
        gap_length = gaps.length[i]
        hist_end = gap_start
        if gap_start < hist_len:
            hist_start = 0
        else:
            hist_start = gap_start - hist_len
        if ((gap_start >= hist_len) and (not np.any(np.isnan(samp.TEMP[hist_start: hist_end]))) and (gap_length <= pred_len)):
            # create history array of shape (1, 30, 1) ending at gap_start
            history = np.reshape(samp.TEMP.values[:gap_start][-hist_len:], (1,hist_len,1))
            # scale history
            history = (history - train_mean)/train_std
            # use model to predict based on history and fill in sample gap
            # descale model output
            out = lstm_model.predict(history)[0,:gap_length]
            out = out * train_std + train_mean
            samp.TEMP[gap_start : (gap_start + gap_length)] = out
    return samp

# Make a copy of the original timeseries and fill gaps with Model 1
temp_df_15_filled = temp_df_15_wgaps.copy()
temp_df_15_filled = gapfill(temp_df_15_filled, bidirectional_lstm_targ10_model, train_mean, train_std, 30, 10)
# Save Model 1 filled ts
pickle_out = open("../data/temp_df_15_filled.pickle", "wb")
pickle.dump(temp_df_15_filled, pickle_out)
pickle_out.close()

# Model 2
# History: 91 days Prediction: 91 days

# Training and validation data preparation
# drop nans
temp_df_15_filled.dropna(inplace=True)

hist_len = 91
targ_len = 91
Time_diff = temp_df_15_filled.index.to_series().diff()
break_index = np.where(Time_diff > pd.Timedelta(days=1))[0] - 1
break_index = np.append(break_index, len(temp_df_15_filled))
window_len = hist_len + targ_len

data = []
labels = []
break_index = np.where(Time_diff > pd.Timedelta(days=1))[0] - 1 
window_len = hist_len + targ_len
for w in range(len(break_index)):
    run_start = hist_len if w == 0 else break_index[w-1] + 1 + hist_len
    run_end = break_index[w] - targ_len   
    for i in range(run_start, run_end):
        indices = range(i-hist_len, i)
        data.append(np.reshape(temp_df_15_filled.values[indices], (hist_len,1)))
        labels.append(temp_df_15_filled.values[i:i+targ_len])
data = np.array(data)
labels = np.array(labels)
# Split into training and test datasets
trainidx = np.random.choice(len(data), int(np.round(0.9*len(data))), replace=False)
train_data = data[trainidx]
train_labels = labels[trainidx]
train_labels = train_labels.reshape((len(train_labels),targ_len))
val_data = np.delete(data, obj=trainidx, axis=0)
val_labels = np.delete(labels, trainidx, axis=0)
val_labels = val_labels.reshape((len(val_labels), targ_len))
# Normalisation
train_mean = train_data.mean()
train_std = train_data.std()
# save training, validation and normalisation data
pickle_out = open("../data/train_mean-std_train-val_data-labels_hist91_targ91.pickle", "wb")
pickle.dump([train_mean, train_std, train_data, train_labels, val_data, val_labels], pickle_out)
pickle_out.close()
train_data = (train_data-train_mean)/train_std
train_labels = (train_labels-train_mean)/train_std
val_data = (val_data-train_mean)/train_std
val_labels = (val_labels-train_mean)/train_std

# Model 2 Training
BATCH_SIZE = 8
BUFFER_SIZE = train_data.shape[0]
train_univariate = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
val_univariate = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
val_univariate = val_univariate.batch(BATCH_SIZE).repeat()
bidirectional_lstm_targ91_model = tf.keras.models.Sequential([
    layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True, input_shape=train_data.shape[-2:])),
    layers.Bidirectional(tf.keras.layers.LSTM(32, activation='relu', return_sequences=True,)),
    layers.Bidirectional(tf.keras.layers.LSTM(8, activation='relu')),
    tf.keras.layers.Dense(91)
])
bidirectional_lstm_targ91_model.compile(optimizer='adam', loss=descaled_mape(mu=train_mean, sd=train_std), metrics=[descaled_mape(mu=train_mean, sd=train_std), 'mae', 'mse', keras.metrics.MeanAbsolutePercentageError()])
checkpoint_path = "../data/bidirectional_lstm_daily_hist91_targ91_loss-descaledmape/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)    
# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=0)
# Model 2 training
STEPS = int(train_data.shape[0]/BATCH_SIZE)
VAL_STEPS = int(val_data.shape[0]/BATCH_SIZE)
EPOCHS = 500
history = bidirectional_lstm_targ91_model.fit(train_univariate, epochs=EPOCHS,
                      steps_per_epoch=STEPS,
                      validation_data=val_univariate, validation_steps=VAL_STEPS, verbose=0,
                 callbacks=[cp_callback, PrintDot(), es_callback])
## Save model
bidirectional_lstm_targ91_model.save('../data/saved_models/temp_gapfilling_daily_bidirectional_lstm_model_hist91_targ91_loss-descaledmape_Earlystopping.h5')
## Save History
import pickle
pickle_out = open("../data/temp_gapfilling_daily_bidirectional_lstm_model_hist91_targ91_loss-descaledmape_Earlystopping_history.pickle", "wb")
pickle.dump(pd.DataFrame(history.history), pickle_out)
pickle_out.close()

# Use Model 2 to fill in gaps up to 91 days long

# add gaps represented by nans 
temp_df_15_filled = temp_df_15_filled.asfreq(freq="1D")
# calculate gaps
gaps = gap_loc_and_len(temp_df_15_filled.TEMP.values)
# Make a copy of the model 1 filled ts and fill gaps with model 2
temp_df_15_filled_2 = temp_df_15_filled.copy()
temp_df_15_filled_2 = gapfill(temp_df_15_filled_2, bidirectional_lstm_targ91_model, train_mean, train_std, 91, 91)
# Save Model 2 filled ts
pickle_out = open("../data/temp_df_15_filled_2.pickle", "wb")
pickle.dump(temp_df_15_filled_2, pickle_out)
pickle_out.close()

## Model 3
# History: 181 days prediction: 181 days

# Training and validation data preparation
# Before we train model 3 we must remove the predicted gap at the end of July-2014 (index 1216) as it looks unreasonable.
temp_df_15_filled_2.TEMP[1216:(1216+45)] = np.nan # gap length is 45 days
# drop nans
temp_df_15_filled_2.dropna(inplace=True)

hist_len = 181
targ_len = 181
Time_diff = temp_df_15_filled_2.index.to_series().diff()
break_index = np.where(Time_diff > pd.Timedelta(days=1))[0] - 1
break_index = np.append(break_index, len(temp_df_15_filled_2))
window_len = hist_len + targ_len

data = []
labels = []
break_index = np.where(Time_diff > pd.Timedelta(days=1))[0] - 1 
window_len = hist_len + targ_len
for w in range(len(break_index)):
    run_start = hist_len if w == 0 else break_index[w-1] + 1 + hist_len
    run_end = break_index[w] - targ_len   
    for i in range(run_start, run_end):
        indices = range(i-hist_len, i)
        data.append(np.reshape(temp_df_15_filled_2.values[indices], (hist_len,1)))
        labels.append(temp_df_15_filled_2.values[i:i+targ_len])
data = np.array(data)
labels = np.array(labels)
# Split into training and test datasets
trainidx = np.random.choice(len(data), int(np.round(0.9*len(data))), replace=False)
train_data = data[trainidx]
train_labels = labels[trainidx]
train_labels = train_labels.reshape((len(train_labels),targ_len))
val_data = np.delete(data, obj=trainidx, axis=0)
val_labels = np.delete(labels, trainidx, axis=0)
val_labels = val_labels.reshape((len(val_labels), targ_len))
# Normalisation
train_mean = train_data.mean()
train_std = train_data.std()
# save training, validation and normalisation data
pickle_out = open("../data/train_mean-std_train-val_data-labels_hist181_targ181.pickle", "wb")
pickle.dump([train_mean, train_std, train_data, train_labels, val_data, val_labels], pickle_out)
pickle_out.close()
train_data = (train_data-train_mean)/train_std
train_labels = (train_labels-train_mean)/train_std
val_data = (val_data-train_mean)/train_std
val_labels = (val_labels-train_mean)/train_std

# Model 3 Training
BATCH_SIZE = 8
BUFFER_SIZE = train_data.shape[0]
train_univariate = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
val_univariate = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
val_univariate = val_univariate.batch(BATCH_SIZE).repeat()
bidirectional_lstm_targ181_model = tf.keras.models.Sequential([
    layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True, input_shape=train_data.shape[-2:])),
    layers.Bidirectional(tf.keras.layers.LSTM(32, activation='relu', return_sequences=True,)),
    layers.Bidirectional(tf.keras.layers.LSTM(8, activation='relu')),
    tf.keras.layers.Dense(181)
])
bidirectional_lstm_targ181_model.compile(optimizer='adam', loss=descaled_mape(mu=train_mean, sd=train_std), metrics=[descaled_mape(mu=train_mean, sd=train_std), 'mae', 'mse', keras.metrics.MeanAbsolutePercentageError()])
checkpoint_path = "../data/bidirectional_lstm_daily_hist181_targ181_loss-descaledmape/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)    
# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=0)

# Model 3 training
STEPS = int(train_data.shape[0]/BATCH_SIZE)
VAL_STEPS = int(val_data.shape[0]/BATCH_SIZE)
EPOCHS = 100
history = bidirectional_lstm_targ181_model.fit(train_univariate, epochs=EPOCHS,
                      steps_per_epoch=STEPS,
                      validation_data=val_univariate, validation_steps=VAL_STEPS, verbose=0,
                 callbacks=[cp_callback, PrintDot()])
## Save model
bidirectional_lstm_targ181_model.save('../data/saved_models/temp_gapfilling_daily_bidirectional_lstm_model_hist181_targ181_loss-descaledmape_100Epochs.h5')
## Save History
import pickle
pickle_out = open("../data/temp_gapfilling_daily_bidirectional_lstm_model_hist181_targ181_loss-descaledmape_100Epochs_history.pickle", "wb")
pickle.dump(pd.DataFrame(history.history), pickle_out)
pickle_out.close()

# Fill in first two gaps with model3 and the last two with model2
temp_df_15_filled_3 = temp_df_15_filled_2.copy()
temp_df_15_filled_3 = temp_df_15_filled_3.asfreq('1D')
gaps = gap_loc_and_len(temp_df_15_filled_3.TEMP)
# Use model 3 to fill the first two gaps only
gaps = gaps[:2]
temp_df_15_filled_3 = gapfill(temp_df_15_filled_3, bidirectional_lstm_targ181_model, train_mean, train_std, 181, 181, gaps = gaps)
# Use model2 to fill remaining gaps
temp_df_15_filled_3 = gapfill(temp_df_15_filled_3, bidirectional_lstm_targ91_model, train_mean, train_std, 91, 91)
# Save Model 3 filled ts
pickle_out = open("../data/temp_df_15_filled_3.pickle", "wb")
pickle.dump(temp_df_15_filled_3, pickle_out)
pickle_out.close()