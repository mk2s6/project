# a = int(input("enter first number: "))
# b = int(input("enter second number: "))
 
# sum = a + b
 
# print(sum)

import os.path
import rpy2 as R
# import pandas as pd
# import pandas.rpy2.common as com
# import base

import rpy2.robjects as robjects
from rpy2.robjects import default_py2ri as hello
import rpy2.rlike.container as rlc

# import pandas.rpy2.common as com

r = robjects.r


# output = r.source('./ex.R')
import pandas as pd

from rpy2.rinterface import StrSexpVector, SexpVector

from rpy2.robjects.packages import importr
from rpy2.robjects import DataFrame
from rpy2.robjects.methods import RS4
from rpy2.robjects import pandas2ri

# import rpy2.robjects as ro
# from rpy2.robjects.packages import importr
# from rpy2.robjects import pandas2ri

from rpy2.robjects.conversion import localconverter


# utils = importr('utils')
# utils.chooseCRANmirror(ind=1)
# utils.install_packages('seewave')

warbleR = importr('warbleR')
tuneR = importr('tuneR')
seewave = importr('seewave')
base = importr('base')

# # data = {'sound.files' : './testFiles/sine.wav', 'selec' : 0, 'start' : robjects.IntVector(0), 'end': robjects.IntVector(20)}
# df = robjects.DataFrame(data)
# print(df)

# b = robjects.IntVector([0, 22])

# tuneR_r = tuneR.readWave(os.path.join('./testFiles/sine.wav'), 0, to = 20, units = "seconds")
# # tuneR_rR = r('''tuneR_r''')
# # res = r('''resultdf <- as.data.frame(tuneR_rR)''')
# dollar = base.__dict__["$"]

# # print(tuneR_r@samp.rate)
# wavspec = seewave.spec(tuneR_r, plot = False)
# wavspecprop = seewave.specprop(wavspec, plot = False)
# print(tuneR_r)
# print(wavspecprop[0][0])

# mean = wavspecprop[0][0]/1000
# sd = wavspecprop[1][0]/1000
# median = wavspecprop[2][0]/1000
# sem = wavspecprop[3][0]
# mode = wavspecprop[4][0]/1000
# Q25 = wavspecprop[5][0]/1000
# Q75 = wavspecprop[6][0]/1000
# IQR = wavspecprop[7][0]/1000
# centroid = wavspecprop[8][0]/1000
# skew = wavspecprop[9][0]
# kurt = wavspecprop[10][0]
# sfm = wavspecprop[11][0]
# sh = wavspecprop[12][0]

# ylim = robjects.IntVector((0, 280/1000))

# wavefundf = seewave.fund(wavspec, f = 16000, ovlp = 50, fmax = 280, ylim = ylim, plot = False)
# # print(wavefundf)

# spe = seewave.dfreq(wavspec, f = 16000, plot = False)
# print(spe)


acoustics = r('''

specan3 <- function(X = data, bp = c(0,22), wl = 2048, threshold = 5, parallel = 1){
  # To use parallel processing: library(devtools), install_github('nathanvan/parallelsugar')
  if(class(X) == "data.frame") {if(all(c("sound.files", "selec", 
                                         "start", "end") %in% colnames(X))) 
  {
    start <- as.numeric(unlist(X$start))
    end <- as.numeric(unlist(X$end))
    sound.files <- as.character(unlist(X$sound.files))
    selec <- as.character(unlist(X$selec))
  } else stop(paste(paste(c("sound.files", "selec", "start", "end")[!(c("sound.files", "selec", 
                                                                        "start", "end") %in% colnames(X))], collapse=", "), "column(s) not found in data frame"))
  } else  stop("X is not a data frame")
  
  #if there are NAs in start or end stop
  if(any(is.na(c(end, start)))) stop("NAs found in start and/or end")  
  
  #if end or start are not numeric stop
  if(all(class(end) != "numeric" & class(start) != "numeric")) stop("'end' and 'selec' must be numeric")
  
  #if any start higher than end stop
  if(any(end - start<0)) stop(paste("The start is higher than the end in", length(which(end - start<0)), "case(s)"))  
  
  #if any selections longer than 20 secs stop
  if(any(end - start>20)) stop(paste(length(which(end - start>20)), "selection(s) longer than 20 sec"))  
  options( show.error.messages = TRUE)
  
  #if bp is not vector or length!=2 stop
  if(!is.vector(bp)) stop("'bp' must be a numeric vector of length 2") else{
    if(!length(bp) == 2) stop("'bp' must be a numeric vector of length 2")}
  
  #return warning if not all sound files were found
  fs <- list.files(path = getwd(), pattern = ".wav$", ignore.case = TRUE)
  if(length(unique(sound.files[(sound.files %in% fs)])) != length(unique(sound.files))) 
    cat(paste(length(unique(sound.files))-length(unique(sound.files[(sound.files %in% fs)])), 
              ".wav file(s) not found"))
  
  #count number of sound files in working directory and if 0 stop
  d <- which(sound.files %in% fs) 
  if(length(d) == 0){
    stop("The .wav files are not in the working directory")
  }  else {
    start <- start[d]
    end <- end[d]
    selec <- selec[d]
    sound.files <- sound.files[d]
  }
  
  # If parallel is not numeric
  if(!is.numeric(parallel)) stop("'parallel' must be a numeric vector of length 1") 
  if(any(!(parallel %% 1 == 0),parallel < 1)) stop("'parallel' should be a positive integer")
  
  # If parallel was called
   if(parallel > 1)
   { options(warn = -1)
     if(all(Sys.info()[1] == "Windows",requireNamespace("parallelsugar", quietly = TRUE) == TRUE)) 
       lapp <- function(X, FUN) parallelsugar::mclapply(X, FUN, mc.cores = parallel) else
         if(Sys.info()[1] == "Windows"){ 
           cat("Windows users need to install the 'parallelsugar' package for parallel computing (you are not doing it now!)")
           lapp <- pbapply::pblapply} else lapp <- function(X, FUN) parallel::mclapply(X, FUN, mc.cores = parallel)} else lapp <- pbapply::pblapply
  
  options(warn = 0)
  
  wave <- NULL
  
  if(parallel == 1) cat("Measuring acoustic parameters:")
  x <- as.data.frame(lapp(1:length(start), function(i) { 
    r <- tuneR::readWave(file.path(getwd(), sound.files[i]), from = start[i], to = end[i], units = "seconds") 
    
    b<- bp #in case bp its higher than can be due to sampling rate
    if(b[2] > ceiling(r@samp.rate/2000) - 1) b[2] <- ceiling(r@samp.rate/2000) - 1 
    
    
    #frequency spectrum analysis
    songspec <- seewave::spec(r, f = r@samp.rate, plot = FALSE)
    analysis <- seewave::specprop(songspec, f = r@samp.rate, flim = c(0, 280/1000), plot = FALSE)
    
    #save parameters
    meanfreq <- analysis$mean/1000
    sd <- analysis$sd/1000
    median <- analysis$median/1000
    Q25 <- analysis$Q25/1000
    Q75 <- analysis$Q75/1000
    IQR <- analysis$IQR/1000
    skew <- analysis$skewness
    kurt <- analysis$kurtosis
    sp.ent <- analysis$sh
    sfm <- analysis$sfm
    mode <- analysis$mode/1000
    centroid <- analysis$cent/1000
    
    #Fundamental frequency parameters
    ff <- seewave::fund(r, f = r@samp.rate, ovlp = 50, threshold = threshold, 
                        fmax = 280, ylim=c(0, 280/1000), plot = FALSE, wl = wl)[, 2]
    meanfun<-mean(ff, na.rm = T)
    minfun<-min(ff, na.rm = T)
    maxfun<-max(ff, na.rm = T)
    
    #Dominant frecuency parameters
    y <- seewave::dfreq(r, f = r@samp.rate, wl = wl, ylim=c(0, 280/1000), ovlp = 0, plot = F, threshold = threshold, bandpass = b * 1000, fftw = TRUE)[, 2]
    meandom <- mean(y, na.rm = TRUE)
    mindom <- min(y, na.rm = TRUE)
    maxdom <- max(y, na.rm = TRUE)
    dfrange <- (maxdom - mindom)
    duration <- (end[i] - start[i])
    
    #modulation index calculation
    changes <- vector()
    for(j in which(!is.na(y))){
      change <- abs(y[j] - y[j + 1])
      changes <- append(changes, change)
    }
    if(mindom==maxdom) modindx<-0 else modindx <- mean(changes, na.rm = T)/dfrange

    wave <<- r
    
    #save results
    return(c(meanfreq, sd, median, Q25, Q75, IQR, skew, kurt, sp.ent, sfm, mode, 
             centroid, meanfun, minfun, maxfun, meandom, mindom, maxdom, dfrange, modindx))
  }))
  
  #change result names
  
  rownames(x) <- c("meanfreq", "sd", "median", "Q25", "Q75", "IQR", "skew", "kurt", "sp.ent", 
                   "sfm","mode", "centroid", "meanfun", "minfun", "maxfun", "meandom", "mindom", "maxdom", "dfrange", "modindx")
  x <- data.frame(as.data.frame(t(x)))
  
  return(list(acoustics = x, wave = wave))
}

# Start with empty data.frame.
data <- data.frame()

folderName = './testFiles'

# Get list of files in the folder.
list <- list.files(folderName, '*.wav')

# Add file list to data.frame for processing.
for (fileName in list) {
    row <- data.frame(fileName, 0, 0, 20)
    data <- rbind(data, row)
}
# print(data)

# Set column names.
names(data) <- c('sound.files', 'selec', 'start', 'end')

# Move into folder for processing.
setwd(folderName)

# print(data)

result <- specan3(data, parallel=1)
# print(result)

setwd('..')

data.frame(result$acoustics)

''')

# gender = 
# pandas2ri.activate()

# print(type(acoustics))

# con = pd.DataFrame(data=acoustics, columns=("duration", "meanfreq", "sd", "median", "Q25", "Q75", "IQR", "skew", "kurt", "sp.ent", "sfm","mode", "centroid", "peakf", "meanfun", "minfun", "maxfun", "meandom", "mindom", "maxdom", "dfrange", "modindx"))

# print(con)
# print(DataFrame(acoustics))
# print(acoustics.rx(1, True))
# acoustics = acoustics.rx(1, True)
# @robjects.conversion.rpy2py.register(DataFrame)

acoustics = pandas2ri.ri2py(acoustics)
# print(acoustics)
# print(type(gender))

# with localconverter(robjects.default_converter + pandas2ri.converter):
#   pd_from_r_df = robjects.conversion.rpy2py(acoustics)

# pd_from_r_df

# print(pd_from_r_df)
# print(type(pd_from_r_df))

# print(pandas2ri.ri2py_dataframe(gender))


# print(robjects.default_py2ri(acoustics))
# print(dollar(tuneR_r, "Samplingrate"))



# warbleR.specan(df , bp = "frange", wl = 2048, threshold = 5, parallel = 1)

# print('hello')


# output1 = r.source('server.R')

# import wave, struct

# waveFile = wave.open('sine.wav', 'r')

# length = waveFile.getnframes()
# for i in range(0,length):
#     waveData = waveFile.readframes(1)
#     data = struct.unpack("<h", waveData)
#     print(int(data[0]))

# output.specan3(['sound.files', 'selec', 'start', 'end'], parallel =1)
# print(output.specan3(X = df))