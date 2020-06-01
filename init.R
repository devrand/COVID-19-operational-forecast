# init.R
#
# R code to install packages if not already installed
#
# app name: patternizer-covid19-forecast
# $ heroku create --stack patternizer-covid19-forecast --buildpack https://github.com/virtualstaticvoid/heroku-buildpack-r.git#heroku-16

my_packages <- c(“date”, “deSolve”, “jsonlite”, “coda”, “MCMCpack”, “MASS”)

install_if_missing <- function(p) {

	if (p %in% rownames(installed.packages()) == FALSE) {
		install.packages(p)
	}
}
invisible(sapply(my_packages, install_if_missing))





