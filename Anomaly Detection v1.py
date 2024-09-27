# Databricks notebook source
!pip install pyod

# COMMAND ----------

#import necessary libraries

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf
from pyod.models.mad import MAD

from datetime import datetime
from datetime import timedelta
from datetime import date
from time import time
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import SQLContext
import math as math
import argparse
import statsmodels.api as sm
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from threshold_input_data limit 5;

# COMMAND ----------

# MAGIC %sql
# MAGIC select country,
# MAGIC        min(start_dt) as min_start_dt,
# MAGIC        max(start_dt) as max_start_dt 
# MAGIC from threshold_input_data
# MAGIC group by 1

# COMMAND ----------

# Read dataset as a dataframe

df_dataset = spark.sql("select distinct start_dt, cnt as count_records from threshold_input_data where country = 'mex' order by start_dt")

df_dataset.createOrReplaceTempView("df_dataset")

df_dataset = df_dataset.cache()

df_dataset.show(20, False)

# COMMAND ----------

# Create a function to detect Outliers using IQR

def find_outliers_IQR(df):
    
    q1=df.iloc[:, 1].quantile(0.25)

    q3=df.iloc[:, 1].quantile(0.75)

    IQR=q3-q1
    
    outliers = df.iloc[:, 0:2][((df.iloc[:, 1]<(q1-1.5*IQR)) | (df.iloc[:, 1]>(q3+1.5*IQR)))]
    
    return outliers

# COMMAND ----------

# Detect outliers using Median Absolute Deviation (MAD).
# MAD is a statistical measure that quantifies the dispersion or variability of a dataset.

def anomaly_mad(model_type):
    
    # Reshape residuals

    residuals = model_type.resid

    residuals = residuals.values.reshape(-1, 1)

    # Fit MAD

    mad = MAD().fit(residuals)

    # Find the outliers

    is_outlier = mad.labels_ == 1
  
    outliers = df_pandas[is_outlier]

    return(outliers)
    

# COMMAND ----------

def get_residuals(model_type):

    residuals = model_type.resid
    residuals = residuals.values
    residuals = residuals[~np.isnan(residuals)]
    return(residuals)

# COMMAND ----------

def sum_of_squares(array):
    
    """Calculates the sum of squares of a NumPy array of any shape.

    Args:
      array: A NumPy array of any shape.

    Returns:
     The sum of squares of the array elements.
    """

    # Flatten the array to a 1D array.
    flattened_array = array.flatten()

    # Calculate the square of each element.
    squared_array = flattened_array ** 2

    # Calculate the sum of the squared elements.
    sum_of_squares = np.sum(squared_array)

    return sum_of_squares

# COMMAND ----------

# After decomposing our data, we need to compare the residuals. 
# As we’re just trying to classify the time series, we don’t need to do anything particularly sophisticated – 
# a big part of this exercise is to produce a quick function that could be used to perform an initial classification in a 
# batch processing environment so simpler is better. We’re going to check how much correlation between data points is still 
# encoded within the residuals. 
# This is the Auto-Correlation Factor (ACF) and it has a function for calculating it. As some of the correlations 
# could be negative, we will select the type with the smallest sum of squares of correlation values.

def get_ssacf(residuals,df_pandas):

    range_var = len(df_pandas.index)  
    
    nlags = math.isqrt(range_var) + 45 
    acf_array = acf(residuals,nlags=nlags,fft=True) 
    ssacf = sum_of_squares(acf_array)
    return(ssacf)

# COMMAND ----------

def get_outliers_today(model_type):

    df = anomaly_mad(model_type)
    df = df.tail(1)

    latest = df.index[-1].date().strftime('%Y-%m-%d')
    p = pd.Timestamp.now().to_period('D')
    current_date = p.to_timestamp().strftime('%Y-%m-%d')
    #current_date = '2023-12-27'

    if latest == current_date:
        return(df)
    return('No outliers today')

# COMMAND ----------

### Check with MX dataset
### Use Seasonal-Trend Decomposition to detect outliers (when there is atleast 2 years of data)
### Use Inter Quartile Range to detect outliers when less than 2 years of data

# Convert to Pandas dataframe for easy manipulation
df_pandas = df_dataset.toPandas()

# Find length of time period to decide right outlier algorithm
length_year = len(df_pandas.index) // 365.25
print('Time-series data in years:',length_year)

if length_year >= 2.0:
      
    # Building Additive and Multiplicative time series models
    # In a multiplicative time series, the components multiply together to make the time series. 
    # If there is an increasing trend, the amplitude of seasonal activity increases. 
    # Everything becomes more exaggerated. This is common for web traffic.

    # In an additive time series, the components add together to make the time series. 
    # If there is an increasing trend, we still see roughly the same size peaks and troughs throughout the time series. 
    # This is often seen in indexed time series where the absolute value is growing but changes stay relative.
    
    df_pandas['start_dt'] = pd.to_datetime(df_pandas['start_dt'])
    df_pandas_mx = df_pandas.set_index('start_dt').asfreq('D').dropna()
    
    decomposition_add_mx = sm.tsa.seasonal_decompose(df_pandas_mx.count_records, model='additive')
    residuals_add_mx = get_residuals(decomposition_add_mx)
         
    decomposition_mul_mx = sm.tsa.seasonal_decompose(df_pandas_mx.count_records, model='multiplicative')
    residuals_mul_mx = get_residuals(decomposition_mul_mx)

    print(type(residuals_mul_mx))
    print(residuals_mul_mx)
    print("------------------")
    
    # Get ACF values for both Additive and Multiplicative models

    ssacf_add_mx = get_ssacf(residuals_add_mx,df_pandas_mx)
    ssacf_mul_mx = get_ssacf(residuals_mul_mx,df_pandas_mx) 
    
    print('ssacf_add:',ssacf_add_mx)
    print('ssacf_mul:',ssacf_mul_mx)
    
    if ssacf_add_mx < ssacf_mul_mx:
        print("Additive Model")
        #print('Additive Model - Outliers Count:',len(anomaly_mad(decomposition_add_mx)))
        #print('Additive Time Series Model - Outliers\n',anomaly_mad(decomposition_add_mx))
        print('Additive Model\n', get_outliers_today(decomposition_add_mx))   
        
    else:
        print("Multiplicative Model")
        #print('Multiplicative Model - Outliers Count:',len(anomaly_mad(decomposition_mul_mx))) 
        #print('Multiplicative Time Series Model - Outliers\n',anomaly_mad(decomposition_mul_mx))
        print('Multiplicative Model\n', get_outliers_today(decomposition_mul_mx))
else:
    df_pandas.count_records=pd.to_numeric(df_pandas.count_records)
    df_outliers_mx = find_outliers_IQR(df_pandas)
    print('Record Count:',len(df_outliers_mx))

# COMMAND ----------

 # See sample outliers from Multiplicative time series model - Mexico
anomaly_mad(decomposition_mul_mx).head(50)

# COMMAND ----------

 # See sample outliers from Multiplicative time series model  
anomaly_mad(decomposition_mul).head(50)

# COMMAND ----------

 # See sample outliers from Additive time series model  
anomaly_mad(decomposition_add).head(50)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Inter-Quartile Range - For less than 2 years of data

# COMMAND ----------

# Visualize IQR plot

df_pandas.count_records=pd.to_numeric(df_pandas.count_records)
sns.boxplot(x=df_pandas['count_records'],showmeans=True)
plt.show()

# COMMAND ----------

# Show IQR outliers

#df_pandas = df_dataset.toPandas()
#df_pandas.count_records=pd.to_numeric(df_pandas.count_records)
#df_outliers = find_outliers_IQR(df_pandas)
# df_outliers.head(50)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Diagnostics plots

# COMMAND ----------

# Detecting Seasonality of time-series model - Mexico

df_pandas = df_dataset.toPandas()
df_pandas.count_records=pd.to_numeric(df_pandas.count_records)

df_pandas['MONTH_YEAR'] = pd.to_datetime(df_pandas['start_dt']).dt.to_period('M')
plt.figure(figsize=(30,4))
sns.boxplot(x='MONTH_YEAR', y='count_records', data=df_pandas).set_title("Month-wise Box Plot")
plt.show()

# COMMAND ----------

# Detect Seasonality using Auto correlation plot

from pandas.plotting import autocorrelation_plot
import pandas as pd
import matplotlib.pyplot as plt
df_pandas.count_records=pd.to_numeric(df_pandas.count_records)
plt.rcParams.update({'figure.figsize':(10,4), 'figure.dpi':100})
autocorrelation_plot((df_pandas.count_records).tolist())

# COMMAND ----------

# Plotting Additive and Multiplicative time series models - Mexico

def plotseasonal(res, axes, title):
    axes[0].title.set_text(title)
    res.observed.plot(ax=axes[0], legend=False)
    axes[0].set_ylabel('Observed')
    
    res.trend.plot(ax=axes[1], legend=False)
    axes[1].set_ylabel('Trend')
    
    res.seasonal.plot(ax=axes[2], legend=False)
    axes[2].set_ylabel('Seasonal')
    
    res.resid.plot(ax=axes[3], legend=False)
    axes[3].set_ylabel('Residual')

fig, axes = plt.subplots(ncols=2, nrows=4, sharex=False, figsize=(30,15))


plotseasonal(decomposition_mul_mx, axes[:,0], title="Multiplicative")
plotseasonal(decomposition_add_mx, axes[:,1], title="Additive")

# COMMAND ----------

# Plotting Additive and Multiplicative time series models

def plotseasonal(res, axes, title):
    axes[0].title.set_text(title)
    res.observed.plot(ax=axes[0], legend=False)
    axes[0].set_ylabel('Observed')
    
    res.trend.plot(ax=axes[1], legend=False)
    axes[1].set_ylabel('Trend')
    
    res.seasonal.plot(ax=axes[2], legend=False)
    axes[2].set_ylabel('Seasonal')
    
    res.resid.plot(ax=axes[3], legend=False)
    axes[3].set_ylabel('Residual')

fig, axes = plt.subplots(ncols=2, nrows=4, sharex=False, figsize=(30,15))


plotseasonal(decomposition_mul, axes[:,0], title="Multiplicative")
plotseasonal(decomposition_add, axes[:,1], title="Additive")

# COMMAND ----------

# Plotting anomalies in Additive and Multiplicative time series models

def plotanomaly(model_type, axes, title):
    
    plt.rc('figure',figsize=(12,6))
    plt.rc('font',size=15)

    fig, ax = plt.subplots()
    x = model_type.resid.index
    y = model_type.resid.values
    ax.plot_date(x, y, color='blue',linestyle='--')
    ax.plot_date(x, y, color='blue',linestyle='--')

    ax.annotate('Anomaly', (mdates.date2num(x[35]), y[35]), xytext=(30, 20), 
           textcoords='offset points', color='red',arrowprops=dict(facecolor='red',arrowstyle='fancy'))
    fig.autofmt_xdate()
    plt.show()

       
plotanomaly(decomposition_mul, axes[:,0], title="Multiplicative")
plotanomaly(decomposition_add, axes[:,1], title="Additive")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Test - Checking historical values

# COMMAND ----------

### Ignore - testing

df_dataset_test = spark.sql("select distinct start_dt, cnt as count_records from threshold_input_data where country = 'mex' and start_dt between '2023-03-10' and '2023-03-31' order by start_dt")

df_dataset_test.createOrReplaceTempView("df_dataset_test")

df_dataset_test.show(20, False)
