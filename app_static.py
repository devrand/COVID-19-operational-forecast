#!/usr/bin/python
# -*- coding: utf-8 -*-

#-----------------------------------------------------------------------
# PROGRAM: app_static.py
#-----------------------------------------------------------------------
# Version 0.8
# 7 June, 2020
# Dr Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
#-----------------------------------------------------------------------

# ========================================================================
# SETTINGS
# ========================================================================
run_smoother = True
run_forecast = True
run_plots = True
plot_intervention = True
plot_total_error = True
plot_raw_obs = True
plot_mcmc = True

#value = 'Bangladesh'
#value = 'Brazil'
#value = 'Greece'
#value = 'India'
#value = 'Iran'
#value = 'Italy'
#value = 'Mexico'
#value = 'Peru'
#value = 'Russia'
#value = 'Spain'
#value = 'Saudi Arabia'
#value = 'United Kingdom'
#value = 'US'
value = 'Morocco'

# FAILING:
#value = 'China'
#value = 'South Korea'
#value = 'North Korea'
# ========================================================================

import numpy as np
import scipy.stats as st
from scipy.stats.kde import gaussian_kde
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import seaborn as sns; sns.set()

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px
from plotly.subplots import make_subplots
from flask import Flask
import os
from random import randint

# -----------------------------------------------------------------------------
def quantile(x,q):
    n = len(x)
    y = np.sort(x)
    return(np.interp(q, np.linspace(1/(2*n), (2*n-1)/(2*n), n), y))
    
def smoother(vec):
    newvec = np.ones(len(vec))
    weekly = np.zeros(7)
    for i in np.arange(0,7):
        dy = np.arange(0,int(np.ceil((len(vec)-i)/7)))
        weekly[i] = np.mean(vec[dy*7+i])
    weeklymean = np.mean(weekly)    
    for i in np.arange(0,7):
        dy = np.arange(0,int(np.ceil((len(vec)-i)/7)))
        newvec[dy*7+i] = vec[dy*7+i]*np.mean(weekly)/weekly[i]
    return newvec
    
def nearest_power_of_10(n):
    x = int(10**np.ceil(np.log10(n)))
    return x

def update_status(dp,dl,df,value):
    # Handle inconsistencies in population, intervention and daily loss data sets
    if value == 'US':
        population = dp[dp['Country Name']=='United States']['Value'].values[0]
    elif value == 'China':
        population = dp[dp['Country Code']=='CHN']['Value'].values[0]
    elif value == 'Russia':
        population = dp[dp['Country Code']=='RUS']['Value'].values[0]
    elif value == 'Iran':
        population = dp[dp['Country Code']=='IRN']['Value'].values[0]
    elif value == 'North Korea':
        population = dp[dp['Country Code']=='PRK']['Value'].values[0]
    elif value == 'South Korea':
        population = dp[dp['Country Code']=='KOR']['Value'].values[0]
    else: 
        population = dp[dp['Country Name']==value]['Value'].values[0]

    if value == 'US':
        interventiondate = dl[dl['country_name']=='United States']['date'].values[0]
        interventionlevel = dl[dl['country_name']=='United States']['intervention'].values[0]
    elif value == 'United Kingdom':
        interventiondate = dl[dl['country_name']=='UK']['date'].values[0]
        interventionlevel = dl[dl['country_name']=='UK']['intervention'].values[0]
    else:
        interventiondate = dl[dl['country_name']==value]['date'].values[0]
        interventionlevel = dl[dl['country_name']==value]['intervention'].values[0]

    if value == 'Australia':
        country = df[df['Country/Region'] == value].T[4:]
    elif value == 'Canada':
        country = df[df['Country/Region'] == value].T[4:]
    elif value == 'China':
        country = df[df['Country/Region'] == value].T[4:]
    else:        
        country = df[(df['Country/Region'] == value) & (pd.isnull(df['Province/State']))].T[4:] 

    if value == 'United Kingdom':
        backshift = pd.to_datetime(interventiondate) - pd.to_timedelta(pd.np.ceil(4), unit="D") # 2020-03-21
        interventiondate = str(int(backshift.year)) + "-" + str(int(backshift.month)) + "-" + str(int(backshift.day)) 
    if value == 'US':
        backshift = pd.to_datetime(interventiondate) - pd.to_timedelta(pd.np.ceil(12), unit="D") # 2020-03-22
        interventiondate = str(int(backshift.year)) + "-" + str(int(backshift.month)) + "-" + str(int(backshift.day)) 

    total = country.sum(axis=1).astype('int') # full record cumulative total 
    deaths = total[total>0]                   # 1st case onwards cumulative total    
    daily = total[total>0].diff()             # 1st case onwards counts
    daily[pd.isnull(daily)] = 1               # 1st case onwards counts - fix 1st NaN
    daily = daily.astype('int')               # 1st case onwards counts - integers
    dates = daily.index                       # 1st case onwards dates

    npre = 28                                 # Hindcast spin-up window
    npad = 14                                 # Pad with 2 weeks of zeros
    nforecast = 28                            # Forecast window
        
    startdatestamp = pd.to_datetime(dates[0]) - pd.to_timedelta(npre, unit="D")
    initdatestamp = pd.to_datetime(dates[0]) - pd.to_timedelta(npad, unit="D")
    casedatestamp = pd.to_datetime(dates[0])
    stopdatestamp = pd.to_datetime(dates[-1])
    enddatestamp = pd.to_datetime(dates[-1]) + pd.to_timedelta(nforecast, unit="D")        
    startdate = str(int(startdatestamp.year)) + "-" + str(int(startdatestamp.month)) + "-" + str(int(startdatestamp.day)) 
    initdate = str(int(initdatestamp.year)) + "-" + str(int(initdatestamp.month)) + "-" + str(int(initdatestamp.day)) 
    casedate = str(int(casedatestamp.year)) + "-" + str(int(casedatestamp.month)) + "-" + str(int(casedatestamp.day)) 
    stopdate = str(int(stopdatestamp.year)) + "-" + str(int(stopdatestamp.month)) + "-" + str(int(stopdatestamp.day)) 
    enddate = str(int(enddatestamp.year)) + "-" + str(int(enddatestamp.month)) + "-" + str(int(enddatestamp.day)) 
    dates_pad = pd.date_range(initdatestamp, dates[-1])
    raw_pad = np.concatenate([np.zeros(npad).astype('int'), daily.values])
    
    dt = pd.DataFrame({'case': value, 
                       'N': population, 
                       'startdate': startdate, 
                       'initdate': initdate, 
                       'casedate': casedate, 
                       'interventiondate': interventiondate, 
                       'interventionlevel': interventionlevel,                        
                       'stopdate': stopdate, 
                       'enddate': enddate,
                       'dates': dates_pad,                        
                       'raw': raw_pad})    

    if run_smoother:
#        for i in range(0,dt.shape[0]-6):
#            dt.loc[dt.index[i+2],'ma'] = np.round(((dt.raw.iloc[i] + dt.raw.iloc[i+1] + dt.raw.iloc[i+2] + dt.raw.iloc[i+3] + dt.raw.iloc[i+4] + dt.raw.iloc[i+5] + dt.raw.iloc[i+6])/7),1)
#        dt['daily'] = dt.raw.rolling(7).mean().fillna(0).astype('int') # Pandas rolling mean
        dt['daily'] = dt.raw.iloc[:].ewm(span=3,adjust=False).mean()    # Exponential weighted mean (3)
#        dt['daily'] = smoother(dt.raw).astype('int')                   # Regularisation (James Annan)
    else:
        dt['daily'] = daily_pad        
    dt['cumulative'] = dt['daily'].cumsum()    
    dt['raw_cumulative'] = dt['raw'].cumsum()    
    ma = dt[dt.daily == max(dt.daily)]['dates']
    peakidx = ma[ma == max(ma)].index    
    peakdatestamp = ma[ma.index.values[0]]    
    peakdate = str(int(peakdatestamp.year)) + "-" + str(int(peakdatestamp.month)) + "-" + str(int(peakdatestamp.day)) 
    dt['peakdate'] = peakdate
        
#    fig,ax = plt.subplots(figsize=[15,10])
#    plt.plot(dt.dates, dt.daily, label='smoothed')
#    plt.plot(dt.dates, dt.raw, label='raw')
#    plt.legend()
#    plt.savefig('smoothing.png')

    return dt
# -----------------------------------------------------------------------------

#----------------------------
# LOAD GLOBAL POPULATION DATA
#----------------------------
"""
Country level data from the World Bank:
https://data.worldbank.org/indicator/SP.POP.TOTL

Extract latest population census value per Country Code
"""

url = r'https://raw.githubusercontent.com/datasets/population/master/data/population.csv'
#Index(['Country Name', 'Country Code', 'Year', 'Value'], dtype='object')
df = pd.read_csv(url)
df.to_csv('dataset_population.csv', sep=',', index=False, header=False, encoding='utf-8')
countries = df['Country Code'].unique()
dp = pd.DataFrame(columns = df.columns)
dp['Country Name'] = [ df[df['Country Code']==countries[i]]['Country Name'].tail(1).values[0] for i in range(len(countries)) ] 
dp['Country Code'] = [ df[df['Country Code']==countries[i]]['Country Code'].tail(1).values[0] for i in range(len(countries)) ] 
dp['Year'] = [ df[df['Country Code']==countries[i]]['Year'].tail(1).values[0] for i in range(len(countries)) ] 
dp['Value'] = [ df[df['Country Code']==countries[i]]['Value'].tail(1).values[0] for i in range(len(countries)) ] 

#----------------------------
# LOAD GLOBAL LOCKDOWN STATUS
#----------------------------
"""
Daily updated Coronavirus containment measures taken by governments from 2020-01-23 to date
Provided by Olivier Lejeune: http://www.olejeune.com/ at:
https://github.com/OlivierLej/Coronavirus_CounterMeasures
dataset.csv has structure: country_id, country_name, 20200123_date, ...  
"""

url = r'https://raw.githubusercontent.com/OlivierLej/Coronavirus_CounterMeasures/master/dataset.csv'
#Index(['country_id', 'country_name', '20200123_date', 
df = pd.read_csv(url)
df.to_csv('dataset_lockdown.csv', sep=',', index=False, header=False, encoding='utf-8')
countries = df['country_name'].unique()
dl = pd.DataFrame(columns = ['country_id', 'country_name', 'intervention', 'date'])
dl['country_id'] = [ df[df['country_name']==countries[i]]['country_id'].tail(1).values[0] for i in range(len(countries)) ] 
dl['country_name'] = [ df[df['country_name']==countries[i]]['country_name'].tail(1).values[0] for i in range(len(countries)) ] 
for i in range(len(countries)-1):
    if i  == 182:
        m = df[df['country_id']=='TX'].T[2:]
    else:
        m = df[df['country_name']==countries[i]].T[2:]
    datestr = []
    try:
        datestr = m[m.values>3].index[0]
        level = m[m.values>3].values[0][0]
    except:
        try:         
            datestr = m[m.values>2].index[0]
            level = m[m.values>2].values[0][0]
        except:
            continue
    dl['intervention'][i] = level
    dl['date'][i] = datestr[0:4] + '-' + str(int(datestr[4:6])) + '-' + str(int(datestr[6:8]))
        
#----------------------------
# LOAD GLOBAL DAILY DEATHS
#----------------------------
"""
Daily updated Coronavirus daily death total per country
https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv
"""

url = r'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
#Index(['Province/State', 'Country/Region', 'Lat', 'Long', '1/22/20', 
df = pd.read_csv(url)
df.to_csv('dataset_dailydeaths.csv', sep=',', index=False, header=False, encoding='utf-8')

dt = update_status(dp,dl,df,value)

N = dt.N[0]
startdate = dt.startdate[0]
initdate = dt.initdate[0]
casedate = dt.casedate[0]
interventiondate = dt.interventiondate[0]
peakdate = dt.peakdate[0]
stopdate = dt.stopdate[0]
enddate = dt.enddate[0]

nforecast = (pd.to_datetime(enddate)-pd.to_datetime(stopdate)).days

#-----------------------------------------
# I/O Python <--> R (NB filename mathcing)
#-----------------------------------------

countrystr = value.replace(" ", "_").lower() + '_'
fileout = 'out.csv'
fileout_post_samp = countrystr + 'post_samp.csv'
fileout_obs = countrystr + 'obs.csv'
fileout_allouts = countrystr + 'allouts.csv'
fileout_alldeadout = countrystr + 'alldeadout.csv'
fileout_allcumdeadout = countrystr + 'allcumdeadout.csv'

# Write Python output for loading by R executable
dt.to_csv(fileout, index=False)

# Call R forecast executable

if run_forecast:
#    subprocess.call ("C:\\Program Files\\R\\R-3.6.3\\bin\\Rscript --vanilla C:\\Users\\User\\Desktop\\REPOS\\COVID-19-operational-forecast\\GITHUB\\covid-19_mcmc_prediction_public_executable.R")
    subprocess.call ("Rscript --vanilla covid-19_mcmc_prediction_public_executable.R")
    
# Read R output back into Python

post_samp = pd.read_csv(fileout_post_samp)
obs = pd.read_csv(fileout_obs)
allouts = pd.read_csv(fileout_allouts)
alldeadout = pd.read_csv(fileout_alldeadout)
allcumdeadout = pd.read_csv(fileout_allcumdeadout)
    
n_ens = len(allouts)
data_pts = len(obs.columns)
r0_mean = np.mean(post_samp['V5'])
r0_sd = np.std(post_samp['V5'])
rt_mean = np.mean(post_samp['V6'])
rt_sd = np.std(post_samp['V6'])        
corr = post_samp.corr()

interval_start = int(obs.values[:,0][0])
interval_end = int(obs.values[:,0][-1]) + nforecast
interval = np.arange(interval_start, interval_end+1)  
dates_all = pd.to_datetime(startdate) + pd.to_timedelta(interval, unit='D')

mcmc_daily = alldeadout.values[:,interval]
lowcent_daily = N*np.array([quantile(mcmc_daily[:,i],0.05) for i in range(len(interval))])
midcent_daily = N*np.array([quantile(mcmc_daily[:,i],0.5) for i in range(len(interval))])
upcent_daily = N*np.array([quantile(mcmc_daily[:,i],0.95) for i in range(len(interval))])
# V2---
report_err = 0.05
model_err = 0.05
#total_error = np.sqrt(.2**2 + (0.03*abs(interval_end - nforecast - interval))**2) # V1
#total_error <- sqrt((log((midcent+sqrt(midcent))/midcent))^2+(report_err)^2 + (model_err*pmax(interval-nowdate,0))^2) # V2
total_error = np.sqrt((np.log((midcent_daily + np.sqrt(midcent_daily))/midcent_daily))**2 + (report_err)**2 + (model_err*abs(interval_end - nforecast - interval))**2) # V2
# V2---
up_log_daily = np.sqrt((total_error*1.64)**2 + (np.log(upcent_daily/midcent_daily))**2) #1.64 for 5-95% range
low_log_daily = np.sqrt((total_error*1.64)**2 + (np.log(midcent_daily/lowcent_daily))**2)
upper_daily = midcent_daily*np.exp(up_log_daily)
lower_daily = midcent_daily*np.exp(-low_log_daily)

mcmc_total = allcumdeadout.values[:,interval]
lowcent_total = N*np.array([quantile(mcmc_total[:,i],0.05) for i in range(len(interval))])
midcent_total = N*np.array([quantile(mcmc_total[:,i],0.5) for i in range(len(interval))])
upcent_total = N*np.array([quantile(mcmc_total[:,i],0.95) for i in range(len(interval))])
up_log_total = np.sqrt((total_error*1.64)**2 + (np.log(upcent_total/midcent_total))**2) #1.64 for 5-95% range
low_log_total = np.sqrt((total_error*1.64)**2 + (np.log(midcent_total/lowcent_total))**2)
upper_total = midcent_total*np.exp(up_log_total)
lower_total = midcent_total*np.exp(-low_log_total)

# JDAnnan: 
# note as a matter of preference we include the model error term only for the future in the graphics,
# and the hindcast spread shows only ensemble spread and sampling/obs error. A debateable 
# decision but including model error here makes it look like we have a massive spread in the past,
# which is not the case. It could be the case that our model error term is a little large?      

upcent_daily[dates_all>stopdate] = upper_daily[dates_all>stopdate]
lowcent_daily[dates_all>stopdate] = lower_daily[dates_all>stopdate]
upcent_total[dates_all>stopdate] = upper_total[dates_all>stopdate]
lowcent_total[dates_all>stopdate] = lower_total[dates_all>stopdate]

upper_daily[dates_all<=stopdate] = upcent_daily[dates_all<=stopdate]
lower_daily[dates_all<=stopdate] = lowcent_daily[dates_all<=stopdate]
upper_total[dates_all<=stopdate] = upcent_total[dates_all<=stopdate]
lower_total[dates_all<=stopdate] = lowcent_total[dates_all<=stopdate]
 
# -----------------------------------------------------------------------------
# PLOTS
# -----------------------------------------------------------------------------

if not run_plots:
    print('Python code end')
    import sys
    sys.exit()

# Plot observations

ymin = min(dt[dt.cumulative>0]['daily'])
ymax = max(dt[dt.cumulative>0]['daily'])
yintervention = dt[dt.dates==interventiondate]['daily'].values[0]  
ypeak = dt[dt.dates==peakdate]['daily'].values[0]  

data = [            
            go.Bar(y=dt[dt.cumulative>0]['raw'], x=dt[dt.cumulative>0]['dates'], 
                   marker_color='lightgrey', 
                   name='Daily deaths', 
                   yaxis='y1'),
            go.Scatter(x=dt[dt.cumulative>0]['dates'], y=dt[dt.cumulative>0]['daily'], 
                       mode='lines+markers', 
                       line=dict(width=1.0,color='red'),
                       marker=dict(size=5, opacity=0.5), 
                       name='3-day EWA', 
                       yaxis='y1'),
]
            
if plot_intervention:
    data_intervention = [
            go.Scatter(x=[interventiondate, interventiondate], 
                       y=[ymin, ymax],
                       mode="lines",
                       legendgroup="a",
                       showlegend=False,
                       marker=dict(size=12, line=dict(width=0.5), color="grey"),
                       name="Intervention",
                       yaxis='y1'),                
            go.Scatter(x=[peakdate, peakdate], 
                       y=[ymin, ymax],
                       mode="lines",
                       legendgroup="a",
                       showlegend=False,
                       marker=dict(size=12, line=dict(width=0.5), color="grey"),
                       name="Peak",
                       yaxis='y1'),                
    ]
    data = data + data_intervention
    
if plot_intervention:    
    layout = go.Layout(            
        yaxis=dict(title='Daily', range=[ymin, ymax]),
        annotations = [
                dict(
                    text = 'Lockdown <br>' + interventiondate,
                    x = pd.to_datetime(interventiondate) - pd.to_timedelta(1, unit='D'),
                    y = ymax, 
                    xanchor = 'right',
                    yanchor = 'top',
                    showarrow = False,
                    ),
                dict(
                    text = 'Peak <br>' + peakdate,
                    x = pd.to_datetime(peakdate) + pd.to_timedelta(1, unit='D'),
                    y = ymax, 
                    xanchor = 'left',
                    yanchor = 'top',
                    showarrow = False,
                ),  
                dict(
                    text = "{0:.0f}".format((pd.to_datetime(peakdate)-pd.to_datetime(interventiondate)).days) + ' days',                    
                    x = pd.to_datetime(peakdate) + pd.to_timedelta(1, unit='D'),
                    y = (ymax-ymin)/4, 
                    xanchor = 'left',
                    yanchor = 'middle',
                    showarrow = False,
                ),                         
                dict(
                    x = peakdate, 
                    y = (ymax-ymin)/4,
                    xref = "x", yref = "y1",
                    axref = "x", ayref = "y1",
                    text = "",
                    showarrow = True,
                    arrowhead = 4,
                    arrowwidth = 2,
                    arrowcolor = 'purple',
                    ax = interventiondate,
                    ay = (ymax-ymin)/4,
                    ),
        ],    
        legend_orientation="v", legend=dict(x=0.72, y=0.95, bgcolor='rgba(205, 223, 212, .4)', bordercolor="Black"),
        margin=dict(r=60, l=60, b=60, t=20),                  
    )
else:
    layout = go.Layout(            
        yaxis=dict(title='Daily', range=[ymin, ymax]),
        legend_orientation="v", legend=dict(x=0.72, y=0.95, bgcolor='rgba(205, 223, 212, .4)', bordercolor="Black"),
        margin=dict(r=60, l=60, b=60, t=20),                  
    )
    
fig = go.Figure(data, layout)
fig.write_image(countrystr + 'static-input.png')

# Plot observations (cumulative)

ymin = min(dt[dt.cumulative>0]['cumulative'])
ymax = max(dt[dt.cumulative>0]['cumulative'])
yintervention = dt[dt.dates==interventiondate]['cumulative'].values[0]  
ypeak = dt[dt.dates==peakdate]['cumulative'].values[0]  

data = [            
            go.Bar(y=dt[dt.cumulative>0]['raw_cumulative'], x=dt[dt.cumulative>0]['dates'], 
                   marker_color='lightgrey', 
                   name='Total deaths', 
                   yaxis='y1'),
            go.Scatter(x=dt[dt.cumulative>0]['dates'], y=dt[dt.cumulative>0]['cumulative'], 
                       mode='lines+markers', 
                       line=dict(width=1.0,color='red'), 
                       marker=dict(size=5, opacity=0.8),
                       name='3-day EWA', 
                       yaxis='y1'),                      
]
            
if plot_intervention:
    data_intervention = [
            go.Scatter(x=[interventiondate, interventiondate], 
                       y=[ymin, ymax],
                       mode="lines",
                       legendgroup="a",
                       showlegend=False,
                       marker=dict(size=12, line=dict(width=0.5), color="grey"),
                       name="Intervention",
                       yaxis='y1'),
            go.Scatter(x=[peakdate, peakdate], 
                       y=[ymin, ymax],
                       mode="lines",
                       legendgroup="a",
                       showlegend=False,
                       marker=dict(size=12, line=dict(width=0.5), color="grey"),
                       name="Peak",
                       yaxis='y1'),
    ]
    data = data + data_intervention
    
if plot_intervention:    
    layout = go.Layout(
        yaxis=dict(title='Cumulative', range=[ymin, ymax]),
        annotations = [
                dict(
                    text = 'Lockdown <br>' + interventiondate,
                    x = pd.to_datetime(interventiondate) - pd.to_timedelta(1, unit='D'),
                    y = ymax, 
                    xanchor = 'right',
                    yanchor = 'top',
                    showarrow = False,
                    ),
                dict(
                    text = 'Peak <br>' + peakdate,
                    x = pd.to_datetime(peakdate) + pd.to_timedelta(1, unit='D'),
                    y = ymax, 
                    xanchor = 'left',
                    yanchor = 'top',
                    showarrow = False,
                ),  
                dict(
                    text = "{0:.0f}".format((pd.to_datetime(peakdate) - pd.to_datetime(interventiondate)).days) + ' days',                    
                    x = pd.to_datetime(peakdate) + pd.to_timedelta(1, unit='D'),
                    y = ypeak, 
                    xanchor = 'left',
                    yanchor = 'middle',
                    showarrow = False,
                ),  
                dict(
                    x = peakdate, 
                    y = ypeak,                    
                    xref = "x", yref = "y1",
                    axref = "x", ayref = "y1",
                    text = "",
                    showarrow = True,
                    arrowhead = 4,
                    arrowwidth = 2,
                    arrowcolor = 'purple',
                    ax = interventiondate,
                    ay = ypeak,                    
                    ),
        ],            
        legend_orientation="v", legend=dict(x=0.72, y=0.05, bgcolor='rgba(205, 223, 212, .4)', bordercolor="Black"),
        margin=dict(r=60, l=60, b=60, t=20),                  
    )
else:
    layout = go.Layout(
        yaxis=dict(title='Cumulative', range=[ymin, ymax]),                           
        legend_orientation="v", legend=dict(x=0.72, y=0.05, bgcolor='rgba(205, 223, 212, .4)', bordercolor="Black"),
        margin=dict(r=60, l=60, b=60, t=20),                  
    )
        
fig = go.Figure(data, layout)
fig.write_image(countrystr + 'static-input-cumulative.png')
        
# Plot forecast
          
ymin = np.log10(0.2)        
if plot_total_error:
    ymax = nearest_power_of_10(np.max(upper_daily))
else:
    ymax = nearest_power_of_10(np.max(upcent_daily))
yintervention = midcent_daily[dates_all==interventiondate][0]
ypeak = midcent_daily[dates_all==peakdate][0]

data = [        
            go.Scatter(x=dates_all, y=upcent_daily, 
                       mode='lines', 
                       fill=None,
                       line=dict(width=1.0, color='navajowhite'),
                       name='95% centile',      
                       showlegend=False,
                       yaxis='y1'),                       
            go.Scatter(x=dates_all, y=lowcent_daily, 
                       mode='lines', 
                       fill='tonexty',
                       line=dict(width=1.0, color='navajowhite'),
                       name='5-95% range',      
                       showlegend=True,
                       yaxis='y1'),                       
            go.Scatter(x=dates_all, y=lowcent_daily, 
                       mode='lines', 
                       fill=None,
                       line=dict(width=1.0, color='navajowhite'),
                       name='5% centile',      
                       showlegend=False,
                       yaxis='y1'),                       
            go.Scatter(x=dates_all, y=midcent_daily, 
                       mode='lines', 
                       line=dict(width=3.0, color='red'),
                       name='Hindcast/forecast',
                       showlegend=True,
                       yaxis='y1'),                       
            go.Scatter(x=dt[dt.dates>=casedate]['dates'], y=dt[dt.dates>=casedate]['daily'], 
                       mode='markers', 
                       marker=dict(size=5, symbol='circle-open', color='red'),
                       name='3-day EWA',
                       showlegend=True,
                       yaxis='y1'),                       
]

if plot_raw_obs:
    data_raw_obs = [
            go.Scatter(x=dt[dt.dates>=casedate]['dates'], y=dt[dt.dates>=casedate]['raw'].astype('int'), 
                       mode='markers', 
                       marker=dict(size=5, symbol='cross-thin-open', color='darkslategrey'), 
                       name='Daily deaths',
                       showlegend=True,
                       yaxis='y1'),                       
    ]
else:
    data_raw_obs = []
    
if plot_total_error:
    data_total_error = [
             go.Scatter(x=dates_all, y=upper_daily, 
                       mode='lines', 
                       fill=None,
                       line=dict(width=1.0, color='navajowhite'),
                       name='forecast uncertainty (upper limit)',      
                       showlegend=False,
                       yaxis='y1'),                       
             go.Scatter(x=dates_all, y=lower_daily, 
                       mode='lines', 
                       fill='tonexty',
                       line=dict(width=1.0, color='navajowhite'),
                       name='forecast uncertainty',      
                       showlegend=False,
                       yaxis='y1'),                       
             go.Scatter(x=dates_all, y=lower_daily, 
                       mode='lines', 
                       fill=None,
                       line=dict(width=1.0, color='navajowhite'),
                       name='forecast uncertainty (lower limit)',      
                       showlegend=False,
                       yaxis='y1'),                       
    ]
else:
    data_total_error = []
    
if plot_intervention:
    data_intervention = [
            go.Scatter(x=[interventiondate, interventiondate], 
                       y=[ymin, ymax],
                       mode="lines",
                       legendgroup="a",
                       showlegend=False,
                       marker=dict(size=12, line=dict(width=0.5), color="grey"),
                       name="Intervention",
                       yaxis='y1'),                       
            go.Scatter(x=[peakdate, peakdate], 
                       y=[ymin, ymax],
                       mode="lines",
                       legendgroup="a",
                       showlegend=False,
                       marker=dict(size=12, line=dict(width=0.5), color="grey"),
                       name="Peak",
                       yaxis='y1'),                      
            go.Scatter(x=[stopdate, stopdate], 
                       y=[ymin, ymax],
                       mode="lines",
                       legendgroup="a",
                       showlegend=False,
                       marker=dict(size=12, line=dict(width=0.5), color="grey"),
                       name="",
                       yaxis='y1'),                      
    ]
else:
    data_intervention = []

data = data_total_error + data + data_raw_obs + data_intervention
    
if plot_intervention:    
    layout = go.Layout(         
            title = {'text' : 'Initialised on ' + stopdate + ' (' + str(pd.to_timedelta(pd.to_datetime(stopdate) - pd.to_datetime(interventiondate), unit='D').days) + ' days after intervention)', 'x': 0.5, 'y': 1.0},                
            yaxis=dict(title='Daily', range=[ymin, np.log10( ymax ) ]),     
            yaxis_type="log",
            legend_orientation="v", legend=dict(x=.72, y=0.05, bgcolor='rgba(205, 223, 212, .4)', bordercolor="Black"),
            annotations = [
                dict(
                    text = 'R0 <br>' + "{0:.2f}".format(r0_mean) + '±' + "{0:.2f}".format(r0_sd),
                    x = pd.to_datetime(interventiondate) - pd.to_timedelta(1, unit='D'),
                    y = np.log10( ymax ), 
                    xanchor = 'right',
                    yanchor = 'top',
                    showarrow = False,
                    ),
                dict(
                    text = 'Rt <br>' + "{0:.2f}".format(rt_mean) + '±' + "{0:.2f}".format(rt_sd),
                    x = pd.to_datetime(interventiondate) + pd.to_timedelta(1, unit='D'),
                    y = np.log10( ymax ), 
                    xanchor = 'left',
                    yanchor = 'top',
                    showarrow = False,
                ),        
                dict(
                    text = 'Tomorrow <br>' + "{0:.0f}".format(midcent_daily[dates_all>stopdate][0])
                        + ' (' 
                        + "{0:.0f}".format(lowcent_daily[dates_all>stopdate][0]) 
                        + '-' 
                        + "{0:.0f}".format(upcent_daily[dates_all>stopdate][0])
                        +')',
                    x = pd.to_datetime(stopdate) - pd.to_timedelta(1, unit='D'),
                    y = np.log10( ymax ), 
                    xanchor = 'right',
                    yanchor = 'top',
                    showarrow = False,
                ),   
            ],           
            margin=dict(r=60, l=60, b=60, t=20),                  
    )
else:            
    layout = go.Layout(         
            title = {'text' : 'Initialised on ' + stopdate + ' (' + str(pd.to_timedelta(pd.to_datetime(stopdate) - pd.to_datetime(interventiondate), unit='D').days) + ' days after intervention)', 'x': 0.5, 'y': 1.0},                
            yaxis=dict(title='Daily', range=[ymin, np.log10( ymax ) ], which="major"),     
            yaxis_type="log",
            legend_orientation="v", legend=dict(x=.72, y=0.05, bgcolor='rgba(205, 223, 212, .4)', bordercolor="Black"),
            annotations = [
                dict(
                    text = 'Tomorrow <br>' + "{0:.0f}".format(midcent_daily[dates_all>dates[-1]][0])
                        + ' (' 
                        + "{0:.0f}".format(lowcent_daily[dates_all>stopdate][0]) 
                        + '-' 
                        + "{0:.0f}".format(upcent_daily[dates_all>stopdate][0])
                        +')',
                    x = pd.to_datetime(stopdate) - pd.to_timedelta(1, unit='D'),
                    y = np.log10( ymax ), 
                    xanchor = 'right',
                    yanchor = 'top',
                    showarrow = False,
                ),                            
            ],           
            margin=dict(r=60, l=60, b=60, t=20),                  
    )
                                    
fig = go.Figure(data, layout)
fig.write_image(countrystr + 'static-forecast.png')

# Plot forecast (cumulative)
     
ymin = np.log10(0.2)
if plot_total_error:
    ymax = nearest_power_of_10(np.max(upper_total))
else:
    ymax = nearest_power_of_10(np.max(upcent_total))
yintervention = midcent_total[dates_all==interventiondate][0]
ypeak = midcent_total[dates_all==peakdate][0]
    
data = [             
            go.Scatter(x=dates_all, y=upcent_total, 
                       mode='lines', 
                       fill=None,
                       line=dict(width=1.0, color='navajowhite'),
                       name='95% centile',      
                       showlegend=False,
                       yaxis='y1'),                                              
            go.Scatter(x=dates_all, y=lowcent_total, 
                       mode='lines', 
                       fill='tonexty',
                       line=dict(width=1.0, color='navajowhite'),
                       name='5-95% range',      
                       showlegend=True,
                       yaxis='y1'),                                              
            go.Scatter(x=dates_all, y=lowcent_total, 
                       mode='lines', 
                       fill=None,
                       line=dict(width=1.0, color='navajowhite'),
                       name='5% centile',      
                       showlegend=False,
                       yaxis='y1'),                                              
            go.Scatter(x=dates_all, y=midcent_total, 
                       mode='lines', 
                       line=dict(width=3.0, color='red'),
                       name='Hindcast/forecast',
                       showlegend=True,
                       yaxis='y1'),                                              
            go.Scatter(x=dt[dt.dates>=casedate]['dates'], y=dt[dt.dates>=casedate]['cumulative'], 
                       mode='markers', 
                       marker=dict(size=5, symbol='circle-open', color='red'), 
                       name='3-day EWA',
                       showlegend=True,
                       yaxis='y1'),                                              
]

if plot_raw_obs:
    data_raw_obs = [
            go.Scatter(x=dt[dt.dates>=casedate]['dates'], y=dt[dt.dates>=casedate]['raw_cumulative'].astype('int'),
                       mode='markers', 
                       marker=dict(size=5, symbol='cross-thin-open', color='darkslategrey'),                        
                       name='Cumulative deaths',
                       showlegend=True,
                       yaxis='y1'),                                              
    ]
else:
    data_raw_obs = []

if plot_total_error:
    data_total_error = [
             go.Scatter(x=dates_all, y=upper_total, mode='lines', 
                       fill=None,
                       line=dict(width=1.0, color='navajowhite'),
                       showlegend=False,
                       yaxis='y1'),                                              
             go.Scatter(x=dates_all, y=lower_total, mode='lines', 
                       fill='tonexty',
                       line=dict(width=1.0, color='navajowhite'),
                       name='forecast uncertainty',      
                       showlegend=False,
                       yaxis='y1'),                                              
             go.Scatter(x=dates_all, y=upper_total, mode='lines', 
                       fill=None,
                       line=dict(width=1.0, color='navajowhite'),
                       showlegend=False,
                       yaxis='y1'),                                              
    ]
else:
    data_total_error = []

if plot_intervention:
    data_intervention = [
            go.Scatter(x=[interventiondate, interventiondate], 
                       y=[ymin, ymax],
                       mode="lines",
                       legendgroup="a",
                       showlegend=False,
                       marker=dict(size=12, line=dict(width=0.5), color="grey"),
                       name="Intervention",
                       yaxis='y1'),                                              
            go.Scatter(x=[peakdate, peakdate], 
                       y=[ymin, ymax],
                       mode="lines",
                       legendgroup="a",
                       showlegend=False,
                       marker=dict(size=12, line=dict(width=0.5), color="grey"),
                       name="Peak",
                       yaxis='y1'),                                              
            go.Scatter(x=[stopdate, stopdate], 
                       y=[ymin, ymax],
                       mode="lines",
                       legendgroup="a",
                       showlegend=False,
                       marker=dict(size=12, line=dict(width=0.5), color="grey"),
                       name="",
                       yaxis='y1'),                                              
    ]
else:     
    data_intervention = []
    
data = data_total_error + data + data_raw_obs + data_intervention

if plot_intervention:
    layout = go.Layout(     
            title = {'text' : 'Initialised on ' + stopdate + ' (' + str(pd.to_timedelta(pd.to_datetime(stopdate) - pd.to_datetime(interventiondate), unit='D').days) + ' days after intervention)', 'x': 0.5, 'y': 1.0},                
            yaxis=dict(title='Cumulative', range=[ymin, np.log10( ymax )]),     
            yaxis_type="log",
            legend_orientation="v", legend=dict(x=.72, y=0.05, bgcolor='rgba(205, 223, 212, .4)', bordercolor="Black"),
            annotations = [
                dict(
                    text = 'R0 <br>' + "{0:.2f}".format(r0_mean) + '±' + "{0:.2f}".format(r0_sd),                    
                    x = pd.to_datetime(interventiondate) - pd.to_timedelta(1, unit='D'),
                    y = np.log10( ymax ), 
                    xanchor = 'right',
                    yanchor = 'top',
                    showarrow = False,
                    ),
                dict(
                    text = 'Rt <br>' + "{0:.2f}".format(rt_mean) + '±' + "{0:.2f}".format(rt_sd),
                    x = pd.to_datetime(interventiondate) + pd.to_timedelta(1, unit='D'),
                    y = np.log10( ymax ), 
                    xanchor = 'left',
                    yanchor = 'top',
                    showarrow = False,
                ),        
                dict(
                    text = 'Tomorrow <br>' + "{0:.0f}".format(midcent_total[dates_all>stopdate][0])
                        + ' (' 
                        + "{0:.0f}".format(midcent_total[dates_all>stopdate][0] - lowcent_daily[dates_all>stopdate][0]) 
                        + '-' 
                        + "{0:.0f}".format(midcent_total[dates_all>stopdate][0] + upcent_daily[dates_all>stopdate][0])
                        +')',
                    x = pd.to_datetime(stopdate) - pd.to_timedelta(1, unit='D'),
                    y = np.log10( ymax ), 
                    xanchor = 'right',
                    yanchor = 'top',
                    showarrow = False,
                ),   
            ],           
            margin=dict(r=60, l=60, b=60, t=20),                  
    )
else:
    layout = go.Layout(     
            title = {'text' : 'Initialised on ' + stopdate + ' (' + str(pd.to_timedelta(pd.to_datetime(stopdate) - pd.to_datetime(interventiondate), unit='D').days) + ' days after intervention)', 'x': 0.5, 'y': 1.0},                
            yaxis=dict(title='Cumulative', range=[ymin, np.log10( ymax )]),     
            yaxis_type="log",
            legend_orientation="v", legend=dict(x=.72, y=0.05, bgcolor='rgba(205, 223, 212, .4)', bordercolor="Black"),
            annotations = [
                dict(
                    text = 'Tomorrow <br>' + "{0:.0f}".format(midcent_total[dates_all>stopdate][0])
                        + ' (' 
                        + "{0:.0f}".format(midcent_total[dates_all>stopdate][0] - lowcent_daily[dates_all>stopdate][0]) 
                        + '-' 
                        + "{0:.0f}".format(midcent_total[dates_all>stopdate][0] + upcent_daily[dates_all>stopdate][0])
                        +')',
                    x = pd.to_datetime(stopdate) + pd.to_timedelta(1, unit='D'),
                    y = np.log10(1), 
                    xanchor = 'left',
                    yanchor = 'bottom',
                    showarrow = False,
                ),     
            ],           
            margin=dict(r=60, l=60, b=60, t=20),                  
    )
            
fig = go.Figure(data, layout)
fig.write_image(countrystr + 'static-forecast_total.png')
         
if not plot_mcmc:
    print('Python code end')
    import sys
    sys.exit()         
         
# Plot MCMC tunable parameter traces
    
fig = make_subplots(rows=2, cols=3, start_cell='top-left')
fig.add_trace(go.Scatter(x=post_samp.index, y=post_samp['V1'], line=dict(width=1.0, color='black'), name=post_samp.columns[0]), row=1, col=1)
fig.add_trace(go.Scatter(x=post_samp.index, y=post_samp['V2'], line=dict(width=1.0, color='red'), name=post_samp.columns[1]), row=1, col=2)
fig.add_trace(go.Scatter(x=post_samp.index, y=post_samp['V3'], line=dict(width=1.0, color='blue'), name=post_samp.columns[2]), row=1, col=3)
fig.add_trace(go.Scatter(x=post_samp.index, y=post_samp['V4'], line=dict(width=1.0, color='grey'), name=post_samp.columns[3]), row=2, col=1)
fig.add_trace(go.Scatter(x=post_samp.index, y=post_samp['V5'], line=dict(width=1.0, color='pink'), name=post_samp.columns[4]), row=2, col=2)
fig.add_trace(go.Scatter(x=post_samp.index, y=post_samp['V6'], line=dict(width=1.0, color='cyan'), name=post_samp.columns[5]), row=2, col=3)
fig.update_xaxes(row=2, col=1, title="Iterations")
fig.update_xaxes(row=2, col=2, title="Iterations")
fig.update_xaxes(row=2, col=3, title="Iterations")
fig.update_layout(showlegend = True)
fig.write_image(countrystr + 'static-mcmc-traces.png')

# Plot MCMC tunable parameter kernel densities
      
hist_data = [post_samp['V1'].values, post_samp['V2'].values, post_samp['V3'].values, 
                 post_samp['V4'].values, post_samp['V5'].values, post_samp['V6'].values]
rug_values = np.mean(post_samp)
group_labels = list(post_samp)
colors = ['black', 'red', 'blue', 'grey', 'pink', 'cyan']

fig = make_subplots(rows=2, cols=3, start_cell='top-left')
fig2 = ff.create_distplot(hist_data, group_labels)
fig.add_trace(go.Scatter(fig2['data'][6], line=dict(color='black', width=2.0), name=post_samp.columns[0]), row=1, col=1)    
fig.add_trace(go.Scatter(fig2['data'][7], line=dict(color='red', width=2.0), name=post_samp.columns[1]), row=1, col=2)
fig.add_trace(go.Scatter(fig2['data'][8], line=dict(color='blue', width=2.0), name=post_samp.columns[2]), row=1, col=3)
fig.add_trace(go.Scatter(fig2['data'][9], line=dict(color='grey', width=2.0), name=post_samp.columns[3]), row=2, col=1)
fig.add_trace(go.Scatter(fig2['data'][10], line=dict(color='pink', width=2.0), name=post_samp.columns[4]), row=2, col=2)
fig.add_trace(go.Scatter(fig2['data'][11], line=dict(color='cyan', width=2.0), name=post_samp.columns[5]), row=2, col=3)

fig.add_trace(go.Scatter(x=post_samp['V1'], y = post_samp['V1']*0., mode = 'markers', 
                             marker=dict(color = 'black', symbol='line-ns-open'), name=post_samp.columns[0]), row=1, col=1)
fig.add_trace(go.Scatter(x=post_samp['V2'], y = post_samp['V2']*0., mode = 'markers', 
                             marker=dict(color = 'red', symbol='line-ns-open'), name=post_samp.columns[1]), row=1, col=2)
fig.add_trace(go.Scatter(x=post_samp['V3'], y = post_samp['V3']*0., mode = 'markers', 
                             marker=dict(color = 'blue', symbol='line-ns-open'), name=post_samp.columns[2]), row=1, col=3)
fig.add_trace(go.Scatter(x=post_samp['V4'], y = post_samp['V4']*0., mode = 'markers', 
                             marker=dict(color = 'grey', symbol='line-ns-open'), name=post_samp.columns[3]), row=2, col=1)
fig.add_trace(go.Scatter(x=post_samp['V5'], y = post_samp['V5']*0., mode = 'markers', 
                             marker=dict(color = 'pink', symbol='line-ns-open'), name=post_samp.columns[4]), row=2, col=2)
fig.add_trace(go.Scatter(x=post_samp['V6'], y = post_samp['V6']*0., mode = 'markers', 
                             marker=dict(color = 'cyan', symbol='line-ns-open'), name=post_samp.columns[5]), row=2, col=3)
fig.update_layout(showlegend=True)
fig.write_image(countrystr + 'static-mcmc-histograms.png')
    
# Plot tunable parameter correlation matrix

f = plt.figure()
f.set_size_inches(11,8)
mask = np.triu(np.ones_like(corr, dtype=np.bool))
sns.heatmap(corr, mask=mask, cmap='coolwarm', vmax=1.0, center=0, square=False, linewidths=.5, cbar_kws={"shrink": .5}, label='Correlation Coefficient')
plt.xlabel('Parameter')
plt.ylabel('Parameter')
title_str = 'Correlation matrix of MCMC parameter ensemble'
plt.title(title_str)
plt.savefig(countrystr + 'static-mcmc-corr.png')

# -----------------------------------------------------------------------------
print('Python code end')
