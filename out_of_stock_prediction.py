import numpy as np
import matplotlib.pyplot as plt
import datetime
from tensorflow.keras.models import load_model
x = datetime.datetime.now()
d=int(x.strftime("%d"))

def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

def trend(time, slope=0):
    return slope * time

def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.9,
                    np.cos(season_time * np.pi),
                    1 / np.exp(10 * season_time))

def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

def noise(time, noise_level=0.001, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

time = np.arange(15*365 + 1, dtype="float32")
baseline = 4
series = trend(time, 0.5)  
baseline = 25
amplitude = 8
slope = 0.014
noise_level = 0.1
# Create the series
series = baseline + trend(time, slope) + seasonality(time, period=31, amplitude=amplitude)
# Update with noise
series += noise(time, noise_level, seed=42)
series=series.reshape(len(series),1)
mean=series.mean(axis=0)
std=series.std(axis=0)
new_series=(series-mean)/std
mod=load_model('lstm_model_50180_dense.h5')
inventory=int(input("Enter the current inventory of TIDE: "))
day=0
val=new_series[60+d-50:60+d]
val=val.reshape((1,50,1))
while(inventory>=0):
    day=day+1
    next_demand=(mod.predict(val)).reshape((1,1,1))
    val=np.append(val,next_demand,axis=1)
    val=np.delete(val,0,axis=1)
    inventory=inventory-((next_demand*std)+mean)
    print((next_demand*std)+mean,(new_series[60+d+day]*std)+mean)
    if(inventory<=0):
        print("out of stock on day :",day)