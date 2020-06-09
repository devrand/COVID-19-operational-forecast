#!/usr/bin/env Rscript

# Re-purposed executable version of R-code by James Annan and Julia Hargreaves:
# https://github.com/jdannan/COVID-19-operational-forecast/blob/master/covid-19_mcmc_prediction_public.Rmd

# Version 0.3
# 24 May, 2020
# Dr Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com

# READ INPUTS FROM PLOTLY PYTHON DASHBOARD OUTPUT 
df <- read.csv(file="out.csv", head=TRUE, sep=",", stringsAsFactors=F)

case <- df$case[1]                           # Country
N <- df$N[1]                                 # Population
startdate <- df$startdate[1]                 # Date of first observation - 2 weeks (for spin-up) 
initdate <- df$initdate[1]                   # Date of first observation
interventiondate <- df$interventiondate[1]   # Date of state intervention
stopdate <- df$stopdate[1]                   # Date of first observation
enddate <- df$enddate[1]                     # Date of last observation + 30 days (for forecast window)
daily <- df$daily                            # Vector of observations

# Smooth data with 7-day MA
#library(zoo)  
#tempdaily <- rollmean(daily, 5, na.pad = TRUE, align = "right")
#tempdaily[is.na(tempdaily)] = 0
#daily <- tempdaily

write(case, stdout())
write(N, stdout())
write(startdate, stdout())
write(initdate, stdout())
write(interventiondate, stdout())
write(stopdate, stdout())
write(enddate, stdout())
write(daily, stdout())

library(date)
library(deSolve)
library(jsonlite)
library(coda)	
library(MCMCpack)
library(MASS)

write("R libraries loaded successfully", stdout())

# title: "MCMC Coronavirus Model"
#
# This R code performs a MCMC parameter estimation procedure in a simple SEIR model using death data which I've mostly cut n pasted from the worldometer web pages but it would be simple enough to use other sources. Set the "case" variable and it should run for a variety of countries though the output plots may need tweaking for suitable start and end dates.
# Represent the basic dynamics in a 6-box version of SEIR based on this post from Thomas House:
# https://personalpages.manchester.ac.uk/staff/thomas.house/blog/modelling-herd-immunity.html
# There are two E and I boxes, the reasons for which I
# can guess at but it's not my model so I won't :-)

centile <- function(data,cent){
  len <- dim(data)[2] #length of series
  num <- dim(data)[1]
  out <- rep(0,len)
  for (i in 1:len){
    so <- sort(data[,i])
    out[i] <- so[max(1,num*cent)] #max operator to stop falling out of range, this just takes the floor which is sloppy but fine for my purposes. Improve it if you care!
  }  
  return(out)
}

odefun <-function(t,state,parameters){
	with(as.list(c(state, parameters)),{
		beta <- parameters[1]
		sigma <- parameters[2]
		gamma <- parameters[3]
		x <- state
		dx <- rep(0,6)
		dx[1] <- -beta*x[1]*(x[4] + x[5]) #susceptible
		dx[2] <- beta*x[1]*(x[4] + x[5]) - sigma*x[2] #newly infected but latent
		dx[3] <- sigma*x[2] - sigma*x[3] #late stage latent
		dx[4] <- sigma*x[3] - gamma*x[4] #newly infectious
		dx[5] <- gamma*x[4] - gamma*x[5] #late infectious
		dx[6] <- gamma*x[5] #recovered
		return(list(dx))
	})
}

# function to calculate deaths from a vector of infectious
dead <- function(infectious,death,infectious_period){
 
  deadout  <- 0*infectious #empty array of correct size
	# parameters deduced from Ferguson except changing their mean of 18.8 to 15 for somewhat subjective reasons. If you want to replicate their function just change the 15 back to 18.8
	sh=4.9
	sc=15/sh
	death_gam <- dgamma((0:60),scale=sc,shape=sh)
	death_gam <- death_gam/sum(death_gam)
	death_rev<- rev(death_gam)
	for (j in 1:length(deadout)){
		deadout[j] <- (death/infectious_period)*sum(death_rev[max(1,62-j):61]*infectious[max(1,j-60):j])
	} 
	return(deadout)  
}

#This code is how I run the model in chunks with different R0 values sequentially
runner <- function(rundeck,latent_p,infectious_p,i0_p){

	allout <- array(0,dim=c(1+tail(rundeck[,1],1),7))
	for (tt in 1:dim(rundeck)[1]){
		if (tt>1) {
			start <- rundeck$dy[tt-1] 
			state <- tail(out,n=1)[2:7]
		}
		else{
			start = 0
			state=array(c(1.0-2.0*i0_p, 0.0, 0.0, i0_p, i0_p, 0.0))
		}
		finish <- rundeck$dy[tt]
		beta <- rundeck$R0[tt] / infectious_p
		sigma <- 2.0 / latent_p
		gamma <- 2.0 / infectious_p
		parameters <- c(beta,sigma,gamma)
		if(finish > start){ #only run if it's a positive interval
			out <- ode(y=state,times=seq(start,finish),func = odefun, parms = parameters,method="ode45") 
			#not sure about integration method.....default was fine unless R0 went too small..don't think precision is really an issue here
			allout[start:finish+1,] <- out
		}
	}
	return(allout)
}

#code to run model and evaluate cost (log likelihood) vs observations. Including prior.
modelcost <- function(params,obs){ 

	latent_period <- max(.5,min(params[1],10)) #bound with 0.1 and 10
	infectious_period <- max(.5,min(params[2],10)) #bound with 0.1 and 10
	i0 <- max(0.,min(exp(params[3]),.01)) #bound with 0. and 0.01 NB this one is logarithmic!
	death <- max(0.001,min(params[4],0.05)) #bound with 0.1 and 5%
    R0 <- max(1.,min(params[5],10)) #bound with 0.1 and 10 also not less than 1 in initial segment
    Rt <- max(0.1,min(params[6],10)) #bound with 0.1 and 10
	# prior mean for parameters in order
    par_pri <- c(4.,2,-15,0.0075,3,1.)
	# prior sd
    par_sd <- c(.5,0.1,15,0.0003,2,.5)
	# set up the rundeck
    # total length of run is hard-wired here, doesn't need to be too long until we get a lot of obs
    enddate <- dates_n[length(dates_n)] + 28 # index of last observation + 4 weeks  
    rundeck <- data.frame(dy = c(as.numeric(as.Date(interventiondate)-as.Date(startdate)), enddate), R0 = c(R0,Rt))
        
    # run the model    
	outs <- runner(rundeck,latent_period,infectious_period,i0)
	infectout  <- rowSums(outs[,5:6]) #calculated total infected over time
	deadout <- dead(infectout,death,infectious_period) #daily deaths
    cumdeadout = cumsum(deadout) #convenient to have cumulative deaths
    # Cost function = log likelihood  
    # need to make sure that zero/low deaths doesn't give a huge error in log space, so I've imposed lower bounds on both model and data
    # note even thoguh I have modified data to eliminate missed days there can still some occasional zeros in the early stages of the epidemic   
    data_cost <- -0.5*sum(((log((pmax(N*deadout[obs[,1]],0.1)))-log(pmax(obs[,2],.1)))^2)/obs[,3]^2)
    pri_cost <- -0.5*sum((params-par_pri)^2/par_sd^2)      
	cost <- data_cost + pri_cost
	return(cost)
}

# input the data and a touch of pre-processing depending on the case
# start point for MCMC calculation also set here...can be anything so long as it's not too bad. Prior mean is usually a sensible choice.
par_pri <- c(4.,2,-15,0.0075,3,1.)

#dates <- seq(as.Date(initdate, format="%y-%m-%d"), by=1, length.out=length(daily))
#dates_n <- as.numeric(as.Date(dates, format="%d/%m/%y")-as.Date(startdate, format="%d/%m/%y"))
dates <- seq(as.Date(initdate), by=1, length.out=length(daily))
dates_n <- as.numeric(as.Date(dates)-as.Date(startdate))
first <- min(which(daily > 0))
obs <- data.frame(tail(dates_n,n=-(first-1)),tail(daily,n=-(first-1)),tail(daily,n=-(first-1))*0)

# smooth out to eliminate gaps - note slightly clumsy code to ensure conservation
# I'm just taking 1/3 of obs from both neighbours of a zero under the assumption this is
# a reporting error
obsfix <- which(obs[,2]==0)
delta_1 <- obs[obsfix-1,2]/3
delta_2 <- obs[obsfix+1,2]/3
obs[obsfix,2] <- obs[obsfix,2] + delta_1+delta_2
obs[obsfix+1,2] <- obs[obsfix+1,2] - delta_2
obs[obsfix-1,2] <- obs[obsfix-1,2] -delta_1

# SET k HERE to withold final k obs for validation e.g. k=7 is the last week of data
k <- 0
obs_extra <- NULL

if(k>0){
	num_obs <- length(obs[,2])
	obs_extra <- obs[(num_obs-(k-1)):num_obs,]
	obs=obs[1:(num_obs-k),]
}

# Setting the observational error...this is quite important.
# theory is to have a minimum of x% obs error and also a model error of y% per day
# I have set at 20% and 3%. Actually this is done in log terms so exp(0.2) ~ 22% etc.
obs[,3] <- rev(sqrt(0.2^2 + (0.03*(0:(length(obs[,3])-1)))^2))

write("obs vector constructed successfully", stdout())

# This is the  bit that actually does the work...calls the mcmc routine
# this should be a decent production-level length
# use these lines for shorter tests when setting up changes...saves a bit of time
#burn<-1000
#runlength<-1000
burn<-3000
runlength<-5000

set.seed(42) #reproducibility!

write("MCMC initializing with true optimum ...", stdout())
control <- list()
control$fnscale <- -1
control$trace <- 10
control$maxit <- 500
pp <-  optim(par_pri, modelcost, obs=obs, control=control)
print(pp)
if(pp$convergence){   #has not converged
	print("has not converged first time")
for (i in 1:10){
	pp <-  optim(pp$par, modelcost, obs=obs,control=control)
	print(pp)
	# add error handling to halt execution on failure to find true optimum which will cause MCMC simulation to break
	if(!pp$convergence)
		{ stop("Runtime error in MCMC initialization")} 
		break
}
}
write("MCMC initialized successfully", stdout())

write("MCMC trace generation ...", stdout())
mcmc_msg <- capture.output(
#	post.samp <- MCMCmetrop1R(modelcost, theta.init=par_pri, obs=obs,thin=1, mcmc=runlength, burnin=burn, verbose=500, logfun=TRUE))
	post.samp <- MCMCmetrop1R(modelcost, theta.init=pp$par, obs=obs,thin=1, mcmc=runlength, burnin=burn, verbose=500, logfun=TRUE)
)
if(grepl("Execution halted", mcmc_msg, fixed=TRUE))
	{ stop("Runtime error in MCMC sampler")
}
write(mcmc_msg, stdout())
write("MCMC simulation run successfully", stdout())

# set generic filename for each run with country prefix
lowercase <- lapply(gsub(" ", "_", case), tolower)

# write output files
fileout_post_samp <- paste(lowercase, "post_samp.csv" , sep="_")
fileout_obs <- paste(lowercase, "obs.csv" , sep="_")
write.csv(post.samp, fileout_post_samp, row.names = FALSE)
write.csv(obs, fileout_obs, row.names = FALSE)

# ensemble of n_ens model runs based on posterior parameter distribution
run_ensemble <- function(post.samp,n_ens,modelrunlen){

	allouts <- array(0,dim=c(n_ens,modelrunlen+1,7))
	alldeadout <- array(0,dim = c(n_ens,modelrunlen+1))
	allcumdeadout <- array(0,dim = c(n_ens,modelrunlen+1))

	for (loop in 1:n_ens){
  
		params <- post.samp[loop*(runlength/n_ens),]  
		latent_period <- max(.5,min(params[1],10)) #bound with 0.1 and 10
		infectious_period <- max(.5,min(params[2],10)) #bound with 0.1 and 10
		i0 <- max(0.,min(exp(params[3]),.01)) #bound with 0. and 10
		death <- max(0.001,min(params[4],0.05)) #bound with 0.1 and 5%  
		R0 <- max(1.,min(params[5],10)) #bound with 0.1 and 10 not less than 1
		Rt <- max(0.1,min(params[6],10)) #bound with 0.1 and 10
		#set up the rundeck
		rundeck <- data.frame(dy = c(as.numeric(as.Date(interventiondate)-as.Date(startdate)), modelrunlen), R0 = c(R0,Rt))
		#run the model
		outs <- runner(rundeck,latent_period,infectious_period,i0)  
		infectout  <- rowSums(outs[,5:6])
		deadout <- dead(infectout,death,infectious_period)
		cumdeadout = cumsum(deadout)  
		allouts[loop,,]<- outs
		alldeadout[loop,] <- deadout
		allcumdeadout[loop,] <- cumdeadout
	}

	runobject <- list()
	runobject$allouts <- allouts
	runobject$alldeadout <- alldeadout
	runobject$allcumdeadout <- allcumdeadout

	# write output files
	fileout_allouts <- paste(lowercase, "allouts.csv" , sep="_")
	fileout_alldeadout <- paste(lowercase, "alldeadout.csv" , sep="_")
	fileout_allcumdeadout <- paste(lowercase, "allcumdeadout.csv" , sep="_")
	write.csv(allouts, fileout_allouts, row.names = FALSE)
	write.csv(alldeadout, fileout_alldeadout, row.names = FALSE)
	write.csv(allcumdeadout, fileout_allcumdeadout, row.names = FALSE)

	return(runobject)
}

run.obj <- run_ensemble(post.samp,500,200)

write("R code end", stdout())




