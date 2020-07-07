#!/usr/bin/python
# -*- coding: utf-8 -*-

#-----------------------------------------------------------------------
# PROGRAM: app.py
#-----------------------------------------------------------------------
# Version 0.6
# 8 June, 2020
# Dr Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
#-----------------------------------------------------------------------

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
import pathlib
from random import randint

# -----------------------------------------------------------------------------
def quantile(x,q):
    n = len(x)
    y = np.sort(x)
    return(np.interp(q, np.linspace(1/(2*n), (2*n-1)/(2*n), n), y))
    
def nearest_power_of_10(n):
    x = int(10**np.ceil(np.log10(n)))
    return x

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

    if value == 'United Kingdom':
        backshift = pd.to_datetime(interventiondate) - pd.to_timedelta(pd.np.ceil(4), unit="D") # 2020-03-21
        interventiondate = str(int(backshift.year)) + "-" + str(int(backshift.month)) + "-" + str(int(backshift.day)) 
    if value == 'US':
        backshift = pd.to_datetime(interventiondate) - pd.to_timedelta(pd.np.ceil(12), unit="D") # 2020-03-22
        interventiondate = str(int(backshift.year)) + "-" + str(int(backshift.month)) + "-" + str(int(backshift.day)) 

    if value == 'Australia':
        country = df[df['Country/Region'] == value].T[4:]
    elif value == 'Canada':
        country = df[df['Country/Region'] == value].T[4:]
    elif value == 'China':
        country = df[df['Country/Region'] == value].T[4:]
    else:        
        country = df[(df['Country/Region'] == value) & (pd.isnull(df['Province/State']))].T[4:] 

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

#   dt['daily'] = smoother(dt.raw).astype('int')                    # Regularisation (James Annan)
#   dt['daily'] = dt.raw.rolling(7).mean().fillna(0).astype('int')  # Pandas rolling mean
    dt['daily'] = dt.raw.iloc[:].ewm(span=3,adjust=False).mean()    # Exponential weighted mean (3)
    dt['cumulative'] = dt['daily'].cumsum()    
    dt['raw_cumulative'] = dt['raw'].cumsum()    
    ma = dt[dt.daily == max(dt.daily)]['dates']
    peakidx = ma[ma == max(ma)].index    
    peakdatestamp = ma[ma.index.values[0]]    
    peakdate = str(int(peakdatestamp.year)) + "-" + str(int(peakdatestamp.month)) + "-" + str(int(peakdatestamp.day)) 
    dt['peakdate'] = peakdate

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

countermeasures = {
'0': 'No or few containment measures in place',
'1': 'Ban on public gatherings, cancellation of major events and conferences. It’s not always easy to know when a country goes from step 0 to step 1. I’ve used public announcements by the government and looked at major sporting and cultural events in the country and when they became cancelled',
'2': 'Schools and universities closed. As from the first day of closure, not when the announcement is made. Dates largely match those available from Unesco at the following web address: https://en.unesco.org/themes/education-emergencies/coronavirus-school-closures',
'3': 'Non-essential shops, restaurants and bars closed. As from the first day of closure, not when the announcement is made',
'4': 'Night curfew/Partial lockdown in place. Applied for part of the day (usually at night) or for broad population categories (eg, people aged over 60)',
'5': 'All-day lockdown. Government requires citizens to shelter in place all day long. Citizens are allowed to come out to buy essential items',
'6': 'Harsh lockdown. Citizens are not allowed to come out of their home, even to buy essential items'
}
"""

url = r'https://raw.githubusercontent.com/OlivierLej/Coronavirus_CounterMeasures/master/dataset.csv'
#Index(['country_id', 'country_name', '20200123_date', 
df = pd.read_csv(url)
df.to_csv('dataset_lockdown.csv', sep=',', index=False, header=False, encoding='utf-8')
countries = df['country_name'].unique()
dl = pd.DataFrame(columns = ['country_id', 'country_name', 'intervention', 'date'])
dl['country_id'] = [ df[df['country_name']==countries[i]]['country_id'].tail(1).values[0] for i in range(len(countries)) ] 
dl['country_name'] = [ df[df['country_name']==countries[i]]['country_name'].tail(1).values[0] for i in range(len(countries)) ] 
#dl[dl['country_name']=='United States']['country_id'].values[0] == 'NY'
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
countries = df['Country/Region'].unique()
countries.sort()
dropdown_countries = [{'label' : i, 'value' : i} for i in countries]

# ========================================================================
# Start the App
# ========================================================================

server = Flask(__name__)
server.secret_key = os.environ.get('secret_key', str(randint(0, 1000000)))
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, server=server, external_stylesheets=external_stylesheets)
app.config.suppress_callback_exceptions = True

app.layout = html.Div(children=[

    html.H1(children='Coronavirus Operational Forecast',
        style={'padding' : '10px', 'width': '100%', 'display': 'inline-block'},
    ),
            
# ------------
    html.Div([            
        dcc.Dropdown(
            id = 'input',
            options = dropdown_countries,   
            value = 'United Kingdom',
#            value = 'US',
            style = {'padding' : '10px', 'width': '240px', 'fontSize' : '20px', 'display': 'inline-block'}
        ),    
    ],
    style={'columnCount': 2}),
# ------------
                            
# ------------            
    html.Div([
        html.P([html.H3(children='Input parameters'),

            html.Div([dcc.Graph(id="country-parameters"),
            ],
            style = {'padding' : '10px', 'display': 'inline-block'}),                
 
            html.Label(['Source of population data: ', html.A('World Bank', href='https://raw.githubusercontent.com/datasets/population/master/data/population.csv')]),               
            html.Label(['Source of daily loss data: ', html.A('CSSE at Johns Hopkins University', href='https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')]),               
            html.Label(['Source of intervention data: ', html.A('Olivier Lejeune', href='https://raw.githubusercontent.com/OlivierLej/Coronavirus_CounterMeasures/master/dataset.csv')]),   
            html.Br(),            
            html.Label([html.A('Global Coronavirus lockdown status app', href='https://patternizer-covid19.herokuapp.com/'), ' by Michael Taylor']),                       
        ],
        style = {'padding' : '10px', 'display': 'inline-block'}),                
            
        html.P([html.H3(children='Output parameters'),
            html.Br(),
            dbc.Button('Run Forecast', id='button', color="info", className="mr-1"),
            html.Div(id='container-button'),
            html.Br(),
            html.Label(['Hindcast/forecast model ', html.A('R code', href='https://github.com/jdannan/COVID-19-operational-forecast'), ' by James D. Annan and Julia C. Hargreaves']),
            html.Label(['medRxiv preprint: ', html.A('Model calibration, nowcasting, and operational prediction of the COVID-19 pandemic', 
                                                     href='https://doi.org/10.1101/2020.04.14.20065227'), ', James D. Annan and Julia C. Hargreaves (2020)']),
            html.Br(),
            html.Label(['App created by ', html.A('Michael Taylor', href='https://patternizer.github.io'),' in Plotly Dash Python']),                    
        ],
        style = {'padding' : '10px', 'display': 'inline-block'}),
    ],
    style={'columnCount': 2}),
# ------------

# ------------
    html.Div([                                                
        html.P([html.H3(children='Country Status: Daily')],
            style = {'padding' : '10px', 'display': 'inline-block'}),
            dcc.Graph(id='input-graph', style = {'width': '85%'}),
        html.P([html.H3(children='Country Forecast: Daily')],
            style = {'padding' : '10px', 'display': 'inline-block'}),
            dcc.Graph(id='output-graph', style = {'width': '85%'}),
    ],    
    style={'columnCount': 2}),
# ------------

# ------------
    html.Div([                                                
        html.P([html.H3(children='Country Status: Cumulative')],
            style = {'padding' : '10px', 'display': 'inline-block'}),
            dcc.Graph(id='input-graph2', style = {'width': '85%'}),
        html.P([html.H3(children='Country Forecast: Cumulative')],
            style = {'padding' : '10px', 'display': 'inline-block'}),
            dcc.Graph(id='output-graph2', style = {'width': '85%'}),
    ],    
    style={'columnCount': 2}),
# ------------

# ------------
    html.Div([                                                
        html.P([html.H3(children='MCMC Tunable Parameter Traces'),
            html.Label(['V1: Latent period [days]']), 
            html.Label(['V2: Infectious period [days]']),
            html.Label(['V3: 0.5 x Initial infection rate']),
            html.Label(['V4: Death rate']),
            html.Label(['V5: Initial reproductive rate, R0']),
            html.Label(['V6: Post-lockdown reproductive rate, Rt'])],
            style = {'padding' : '10px', 'fontSize':12, 'display': 'inline-block'}),
            dcc.Graph(id='mcmc-trace-graph', style = {'width': '100%'}),
        html.P([html.H3(children='MCMC Tunable Parameter Kernel Density Estimates'),
            html.Label(['V1: Latent period [days]']), 
            html.Label(['V2: Infectious period [days]']),
            html.Label(['V3: 0.5 x Initial infection rate']),
            html.Label(['V4: Death rate']),
            html.Label(['V5: Initial reproductive rate, R0']),
            html.Label(['V6: Post-lockdown reproductive rate, Rt'])],
            style = {'padding' : '10px', 'fontSize':12, 'display': 'inline-block'}),
            dcc.Graph(id='mcmc-histogram-graph', style = {'width': '100%'}),
        html.P([html.H3(children='MCMC Tunable Parameter Correlation Matrix'),
            html.Label(['V1: Latent period [days]']), 
            html.Label(['V2: Infectious period [days]']),
            html.Label(['V3: 0.5 x Initial infection rate']),
            html.Label(['V4: Death rate']),
            html.Label(['V5: Initial reproductive rate, R0']),
            html.Label(['V6: Post-lockdown reproductive rate, Rt'])],
            style = {'padding' : '10px', 'fontSize':12, 'display': 'inline-block'}),
            dcc.Graph(id='mcmc-corr-graph', style = {'width': '100%'}),         
    ],    
    style={'columnCount': 3}),
# ------------
         
    ],
    style={'columnCount': 1})
            
@app.callback(
    Output(component_id='country-parameters', component_property='figure'),
    [Input(component_id='input', component_property='value')]
    )

def update_parameters(value):
    
    dt = update_status(dp,dl,df,value)
              
    data = [
        go.Table(
            header=dict(values=['Parameter', 'Value'],
                line_color='darkslategray',
                fill_color='lightgrey',
                align='left'),
            cells=dict(values=[['Population', 'First loss', 'Last update', 'Intervention'], # 1st column
                [str(dt.N[0]), dt.casedate[0], dt.stopdate[0], dt.interventiondate[0] + ' (level=' + str(dt.interventionlevel[0]) +')']], # 2nd column
                line_color='darkslategray',
                fill_color='white',
                align='left')
        ),
    ]
    layout = go.Layout(  height=120, width=500, margin=dict(r=5, l=5, b=0, t=0))
    return {'data': data, 'layout':layout} 
        
@app.callback(
    Output(component_id='input-graph', component_property='figure'),
    [Input(component_id='input', component_property='value')]
    )
def update_input_graph(value):

    dt = update_status(dp,dl,df,value)
    
    interventiondate = dt.interventiondate[0]
    peakdate = dt.peakdate[0]
    
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

    layout = go.Layout(            
        yaxis=dict(title='Daily', range=[ymin, ymax]),
        annotations = [
                dict(
                    text = 'Lockdown',
                    x = pd.to_datetime(interventiondate) - pd.to_timedelta(1, unit='D'),
                    y = ymax, 
                    xanchor = 'right',
                    yanchor = 'top',
                    showarrow = False,
                    ),
                dict(
                    text = 'Peak',
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
        legend_orientation="v", legend=dict(x=.72, y=0.95, bgcolor='rgba(205, 223, 212, .4)', bordercolor="Black"),
        margin=dict(r=60, l=60, b=60, t=80),                  
    )
    
    return {'data': data, 'layout':layout} 

@app.callback(
    Output(component_id='input-graph2', component_property='figure'),
    [Input(component_id='input', component_property='value')]
    )
def update_input_graph(value):

    dt = update_status(dp,dl,df,value)

    interventiondate = dt.interventiondate[0]
    peakdate = dt.peakdate[0]

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

    layout = go.Layout(
        yaxis=dict(title='Cumulative', range=[ymin, ymax]),
        annotations = [
                dict(
                    text = 'Lockdown',
                    x = pd.to_datetime(interventiondate) - pd.to_timedelta(1, unit='D'),
                    y = ymax, 
                    xanchor = 'right',
                    yanchor = 'top',
                    showarrow = False,
                    ),
                dict(
                    text = 'Peak',
                    x = pd.to_datetime(peakdate) + pd.to_timedelta(1, unit='D'),
                    y = ymax, 
                    xanchor = 'left',
                    yanchor = 'top',
                    showarrow = False,
                ),  
                dict(
                    text = "{0:.0f}".format((pd.to_datetime(peakdate)-pd.to_datetime(interventiondate)).days) + ' days',                    
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
                    ax=interventiondate,
                    ay = ypeak,
                    ),
        ],            
        legend_orientation="v", legend=dict(x=.72, y=0.05, bgcolor='rgba(205, 223, 212, .4)', bordercolor="Black"),
        margin=dict(r=60, l=60, b=60, t=80),                  
    )
    
    return {'data': data, 'layout':layout} 

@app.callback(
    Output(component_id='container-button', component_property='children'),
    [Input(component_id='button', component_property='n_clicks'), 
     Input(component_id='input', component_property='value')],
)
def update_forecast_button(n_clicks, value):
    
    # -----------------------------------------------------------------------------
    # Call the executable form of the operational COVID-19 model of James and Jules:
    # -----------------------------------------------------------------------------

    dt = update_status(dp,dl,df,value)
        
    #-----------------------------------------
    # I/O Python <--> R (NB filename mathcing)
    #-----------------------------------------

    fileout = 'out.csv'
    countrystr = value.replace(" ", "_").lower() + '_'
    fileout_post_samp = countrystr + 'post_samp.csv'
    fileout_obs = countrystr + 'obs.csv'
    fileout_allouts = countrystr + 'allouts.csv'
    fileout_alldeadout = countrystr + 'alldeadout.csv'
    fileout_allcumdeadout = countrystr + 'allcumdeadout.csv'

    # Write Python output for loading by R executable
    dt.to_csv(fileout, index=False)

    if n_clicks == 1:
        filepath = pathlib.Path(__file__).resolve().parent        
        R_str = pathlib.Path(filepath, "covid-19_mcmc_prediction_public_executable").with_suffix(".R")        
        subprocess.call(['Rscript', '--vanilla', 'R_str'])
#        subprocess.call("C:\\Program Files\\R\\R-3.6.3\\bin\\Rscript --vanilla C:\\Users\\User\\Desktop\\REPOS\\COVID-19-operational-forecast\\GITHUB\\covid-19_mcmc_prediction_public_executable.R")
        return    
    else:    
        n_clicks = 0
        return
        
@app.callback(
    Output(component_id='output-graph', component_property='figure'),
    [Input(component_id='button', component_property='n_clicks'), 
     Input(component_id='input', component_property='value')],
)
def update_output_graph(n_clicks, value):

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

        filepath = pathlib.Path(__file__).resolve().parent        
        countrystr = value.replace(" ", "_").lower() + '_'
        fileout_post_samp = countrystr + 'post_samp'
        fileout_obs = countrystr + 'obs'
        fileout_allouts = countrystr + 'allouts'
        fileout_alldeadout = countrystr + 'alldeadout'
        fileout_allcumdeadout = countrystr + 'allcumdeadout'

        post_samp = pd.read_csv(pathlib.Path(filepath, fileout_post_samp).with_suffix('.csv'))
        obs = pd.read_csv(pathlib.Path(filepath, fileout_obs).with_suffix('.csv'))
        allouts = pd.read_csv(pathlib.Path(filepath, fileout_allouts).with_suffix('.csv'))
        alldeadout = pd.read_csv(pathlib.Path(filepath, fileout_alldeadout).with_suffix('.csv'))
        allcumdeadout = pd.read_csv(pathlib.Path(filepath, fileout_allcumdeadout).with_suffix('.csv'))
        
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

        report_err = 0.2
        model_err = 0.05
        #total_error = np.sqrt(.2**2 + (0.03*abs(interval_end - nforecast - interval))**2) # V1
        #total_error <- sqrt((log((midcent+sqrt(midcent))/midcent))^2+(report_err)^2 + (model_err*pmax(interval-nowdate,0))^2) # V2
        total_error = np.sqrt((np.log((midcent_daily + np.sqrt(midcent_daily))/midcent_daily))**2 + (report_err)**2 + (model_err*abs(interval_end - nforecast - interval))**2) # V2

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
        
        # Plot forecast

        ymin = np.log10(0.2)        
        ymax = nearest_power_of_10(np.max(upper_daily))
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
            go.Scatter(x=dt[dt.dates>=casedate]['dates'], y=dt[dt.dates>=casedate]['raw'].astype('int'), 
                       mode='markers', 
                       marker=dict(size=5, symbol='cross-thin-open', color='darkslategrey'), 
                       name='Daily deaths',
                       showlegend=True,
                       yaxis='y1'),                                              
        ]

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
        data = data_total_error + data
        
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
                       name="Initialisation",
                       yaxis='y1'),                      
        ]
        data = data + data_intervention
        
        layout = go.Layout(         
#            title = {'text' : 'Initialised on ' + stopdate + ' (' + str(pd.to_timedelta(pd.to_datetime(stopdate) - pd.to_datetime(interventiondate), unit='D').days) + ' days after intervention)', 'x': 0.5, 'y': 1.0},                
            yaxis=dict(title='Daily', range=[np.log10(0.2), np.log10( ymax ) ]),     
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
            margin=dict(r=60, l=60, b=60, t=60),                  
        )
            
        return {'data': data, 'layout':layout}       
              
@app.callback(
    Output(component_id='output-graph2', component_property='figure'),
    [Input(component_id='button', component_property='n_clicks'), 
     Input(component_id='input', component_property='value')],
)
def update_output_graph2(n_clicks, value):

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

        filepath = pathlib.Path(__file__).resolve().parent        
        countrystr = value.replace(" ", "_").lower() + '_'
        fileout_post_samp = countrystr + 'post_samp'
        fileout_obs = countrystr + 'obs'
        fileout_allouts = countrystr + 'allouts'
        fileout_alldeadout = countrystr + 'alldeadout'
        fileout_allcumdeadout = countrystr + 'allcumdeadout'

        post_samp = pd.read_csv(pathlib.Path(filepath, fileout_post_samp).with_suffix('.csv'))
        obs = pd.read_csv(pathlib.Path(filepath, fileout_obs).with_suffix('.csv'))
        allouts = pd.read_csv(pathlib.Path(filepath, fileout_allouts).with_suffix('.csv'))
        alldeadout = pd.read_csv(pathlib.Path(filepath, fileout_alldeadout).with_suffix('.csv'))
        allcumdeadout = pd.read_csv(pathlib.Path(filepath, fileout_allcumdeadout).with_suffix('.csv'))

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

        report_err = 0.2
        model_err = 0.05
        #total_error = np.sqrt(.2**2 + (0.03*abs(interval_end - nforecast - interval))**2) # V1
        #total_error <- sqrt((log((midcent+sqrt(midcent))/midcent))^2+(report_err)^2 + (model_err*pmax(interval-nowdate,0))^2) # V2
        total_error = np.sqrt((np.log((midcent_daily + np.sqrt(midcent_daily))/midcent_daily))**2 + (report_err)**2 + (model_err*abs(interval_end - nforecast - interval))**2) # V2

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
        
        # Plot forecast (cumulative)

        ymin = np.log10(0.2)
        ymax = nearest_power_of_10(np.max(upper_total))
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
            go.Scatter(x=dt[dt.dates>=casedate]['dates'], y=dt[dt.dates>=casedate]['raw_cumulative'].astype('int'),
                       mode='markers', 
                       marker=dict(size=5, symbol='cross-thin-open', color='darkslategrey'),                        
                       name='Cumulative deaths',
                       showlegend=True,
                       yaxis='y1'),                                                                     
        ]

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
        data = data_total_error + data

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
                       name="Initialisation",
                       yaxis='y1'),                                              
        ]
        data = data + data_intervention

        layout = go.Layout(     
#            title = {'text' : 'Initialised on ' + stopdate + ' (' + str(pd.to_timedelta(pd.to_datetime(stopdate) - pd.to_datetime(interventiondate), unit='D').days) + ' days after intervention)', 'x': 0.5, 'y': 1.0},                
            yaxis=dict(title='Cumulative', range=[np.log10(0.2), np.log10( ymax )]),     
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
            margin=dict(r=60, l=60, b=60, t=60),                  
        )

        return {'data': data, 'layout':layout}       
                     
@app.callback(
    Output(component_id='mcmc-trace-graph', component_property='figure'),
    [Input(component_id='button', component_property='n_clicks'), 
     Input(component_id='input', component_property='value')],
)
def update_mcmc_trace_graph(n_clicks, value):

    # Read in R output data.frames from the forecast run
        
    filepath = pathlib.Path(__file__).resolve().parent
    countrystr = value.replace(" ", "_").lower() + '_'
    fileout_post_samp = countrystr + 'post_samp'
    post_samp = pd.read_csv(pathlib.Path(filepath, fileout_post_samp).with_suffix('.csv'))

    # Plot MCMC parameter traces
    
    fig = make_subplots(rows=2, cols=3, start_cell='top-left')
    fig.add_trace(go.Scatter(x=post_samp.index, y=post_samp['V1'], line=dict(width=1.0, color='black'), name=post_samp.columns[0]), row=1, col=1)
    fig.add_trace(go.Scatter(x=post_samp.index, y=post_samp['V2'], line=dict(width=1.0, color='red'), name=post_samp.columns[1]), row=1, col=2)
    fig.add_trace(go.Scatter(x=post_samp.index, y=post_samp['V3'], line=dict(width=1.0, color='blue'), name=post_samp.columns[2]), row=1, col=3)
    fig.add_trace(go.Scatter(x=post_samp.index, y=post_samp['V4'], line=dict(width=1.0, color='grey'), name=post_samp.columns[3]), row=2, col=1)
    fig.add_trace(go.Scatter(x=post_samp.index, y=post_samp['V5'], line=dict(width=1.0, color='pink'), name=post_samp.columns[4]), row=2, col=2)
    fig.add_trace(go.Scatter(x=post_samp.index, y=post_samp['V6'], line=dict(width=1.0, color='cyan'), name=post_samp.columns[5]), row=2, col=3)
    fig.update_xaxes(row=2, col=2, title="Iterations")
    fig.update_layout(showlegend = True)

    return fig    
      
@app.callback(
    Output(component_id='mcmc-histogram-graph', component_property='figure'),
    [Input(component_id='button', component_property='n_clicks'), 
     Input(component_id='input', component_property='value')],
)
def update_mcmc_histogram_graph(n_clicks, value):

    # Read in R output data.frames from the forecast run

    filepath = pathlib.Path(__file__).resolve().parent
    countrystr = value.replace(" ", "_").lower() + '_'
    fileout_post_samp = countrystr + 'post_samp'
    post_samp = pd.read_csv(pathlib.Path(filepath, fileout_post_samp).with_suffix('.csv'))    

    # Plot MCMC parameter histograms

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

#    ADD HISTOGRAMS
#    fig.add_trace(go.Histogram(fig2['data'][0], marker_color='black'), row=1, col=1)
#    fig.add_trace(go.Histogram(fig2['data'][1], marker_color='grey'), row=1, col=2)
#    fig.add_trace(go.Histogram(fig2['data'][2], marker_color='red'), row=1, col=3)
#    fig.add_trace(go.Histogram(fig2['data'][3], marker_color='pink'), row=2, col=1)
#    fig.add_trace(go.Histogram(fig2['data'][4], marker_color='blue'), row=2, col=2)
#    fig.add_trace(go.Histogram(fig2['data'][5], marker_color='cyan'), row=2, col=3)

#    ADD RUG PLOTS
#    fig.add_trace(go.Scatter(x=post_samp['V1'], y = post_samp['V1']*0., mode = 'markers', marker=dict(color = 'black', symbol='line-ns-open'), name=post_samp.columns[0]), row=1, col=1)
#    fig.add_trace(go.Scatter(x=post_samp['V2'], y = post_samp['V2']*0., mode = 'markers', marker=dict(color = 'red', symbol='line-ns-open'), name=post_samp.columns[1]), row=1, col=2)
#    fig.add_trace(go.Scatter(x=post_samp['V3'], y = post_samp['V3']*0., mode = 'markers', marker=dict(color = 'blue', symbol='line-ns-open'), name=post_samp.columns[2]), row=1, col=3)
#    fig.add_trace(go.Scatter(x=post_samp['V4'], y = post_samp['V4']*0., mode = 'markers', marker=dict(color = 'grey', symbol='line-ns-open'), name=post_samp.columns[3]), row=2, col=1)
#    fig.add_trace(go.Scatter(x=post_samp['V5'], y = post_samp['V5']*0., mode = 'markers', marker=dict(color = 'pink', symbol='line-ns-open'), name=post_samp.columns[4]), row=2, col=2)
#    fig.add_trace(go.Scatter(x=post_samp['V6'], y = post_samp['V6']*0., mode = 'markers', marker=dict(color = 'cyan', symbol='line-ns-open'), name=post_samp.columns[5]), row=2, col=3)

    fig2.update_layout(showlegend=True)

    return fig

@app.callback(
    Output(component_id='mcmc-corr-graph', component_property='figure'),
    [Input(component_id='button', component_property='n_clicks'), 
     Input(component_id='input', component_property='value')],
)
def update_mcmc_corr_graph(n_clicks, value):

    # Read in R output data.frames from the forecast run
        
    filepath = pathlib.Path(__file__).resolve().parent
    countrystr = value.replace(" ", "_").lower() + '_'
    fileout_post_samp = countrystr + 'post_samp'
    post_samp = pd.read_csv(pathlib.Path(filepath, fileout_post_samp).with_suffix('.csv'))
    corr = post_samp.corr()
    mask = np.triu(np.ones_like(corr, dtype=np.bool))
    corr[mask == True] = np.nan
    group_labels = list(post_samp)

    # Plot MCMC parameter correlation matrix
    
    data = [
            
        go.Heatmap(x=group_labels, y=group_labels, z=corr, zmin=-1, zmax=1,
                   colorscale='Picnic', 
                   colorbar={'title': 'Corr Coef', 'thickness': 10})
    ]
    
    layout = go.Layout(                         
         yaxis=dict(autorange='reversed'),
    )
    
    return {'data': data, 'layout':layout}  
        
##################################################################################################
# Run the dash app
##################################################################################################

if __name__ == "__main__":
    app.run_server(debug=True)

print('** END')
