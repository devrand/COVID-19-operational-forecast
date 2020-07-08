![image](https://github.com/patternizer/COVID-19-operational-forecast/blob/master/app-snapshot.png)

# COVID-19-operational-forecast

This is a Plotly Dash Python app frontend that calls a version of the hindcast/forecast model of COVID-19 deaths using 
MCMC parameter estimation by James D Annan and Julia C Hargreaves: https://github.com/jdannan/COVID-19-operational-forecast.

The python interface is designed to allow users to run the MCMC version of the hindcast/forecast model written in R code by 
James D. Annan and Julia C. Hargreaves (2020) for different country data. I adopted a 3-day exponentially weighted average 
smoothing scheme to adjust for weekend reporting effects. The original MCMC model is described in the medRxiv preprint by 
James D. Annan and Julia C. Hargreaves (2020) here https://www.medrxiv.org/content/10.1101/2020.04.14.20065227v2. 
James and Julia’s hindcast/forecast model produces excellent results and is continually evolving. Please refer to their blog 
at https://bskiesresearch.wordpress.com/ to keep up with latest developments as I am no longer working on this python frontend.

The way the Plotly app I made works is that it imports population data for countries and the daily death data made available by 
CSSE at Johns Hopkins University together with publicly available lockdown data to provide the inputs needed by the R code which 
I created a stripped out executable version from.

The online app at https://patternizer-covid19-forecast.herokuapp.com/ is no longer maintained and has been discontinued.

## Contents

* `app_static.py` - main script for local runs to be run with Python 3.6+
* `app.py` - dashboard script (no longer maintained)
* `covid-19_mcmc_prediction_public_executable.R` - stripped down executable version of the forecast model R markdown code written by https://github.com/jdannan

The first step is to clone the latest COVID-19-operational-forecast code and step into the check out directory: 

    $ git clone https://github.com/patternizer/COVID-19-operational-forecast.git
    $ cd COVID-19-operational-forecast
    
### Using Standard Python 

The code should run with the [standard CPython](https://www.python.org/downloads/) installation and was tested 
in a conda virtual environment running a 64-bit version of Python 3.6+.

The Heroku build deployed successfully but there is an issue (see issues) associated with calling the R executable from the
Heroku online build location. As a work-around I have written a static version of the app that can be run locally for all countries.
For other than UK country forecasts edit the value=[country] line in app_static.py in your CLI and run with:

    $ python app_static.py

Please note that this implementation uses a 3-day exponential weighted average smoother to adjust for the weekend effect associated with case number reporting lulls.
I am no longer maintaining this codebase which was experimental but feel free to use it if it serves your purposes with the usual disclaimers.

		    
## License

My code for the app is distributed under terms and conditions of the [MIT license](https://opensource.org/licenses/MIT).
The original R code has its own license. Please see: https://github.com/jdannan/COVID-19-operational-forecast/blob/master/LICENSE.

## Contact information

* [Michael Taylor](https://patternizer.github.io)


