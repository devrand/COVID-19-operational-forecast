![image](https://github.com/patternizer/COVID-19-operational-forecast/blob/master/app-snapshot.png)

# COVID-19-operational-forecast

Plotly Dash Python app implementation of the operational hindcast/forecast model of COVID-19 deaths using 
MCMC parameter estimation by James D Annan: https://github.com/jdannan/COVID-19-operational-forecast

## Contents

* `app.py` - main script to be run with Python 3.6+
* `app_static.py` - script for local runs
* `covid-19_mcmc_prediction_public_executable.R` - stripped down executable version of the forecast model R markdown code written by https://github.com/jdannan

The first step is to clone the latest COVID-19-operational-forecast code and step into the check out directory: 

    $ git clone https://github.com/patternizer/COVID-19-operational-forecast.git
    $ cd COVID-19-operational-forecast
    
### Using Standard Python 

The code should run with the [standard CPython](https://www.python.org/downloads/) installation and was tested 
in a conda virtual environment running a 64-bit version of Python 3.6+.

Run with:

    $ python app.py

The Heroku build deploys successfully but there is an issue (see issues) associated with calling the R executable from the
Heroku online build location. As a work-around I have written a static version of the app that can be run locally for all countries.
For other than UK country forecasts edit the value=[country] line in app_static.py in your CLI and run with:

    $ python app_static.py
		    
## License

The code is distributed under terms and conditions of the [MIT license](https://opensource.org/licenses/MIT).

## Contact information

* [Michael Taylor](https://patternizer.github.io)


