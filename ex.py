# a = int(input("enter first number: "))
# b = int(input("enter second number: "))
 
# sum = a + b
 
# print(sum)

import os.path
import rpy2 as R
# import base

import rpy2.robjects as robjects
from rpy2.robjects import default_py2ri as hello
import rpy2.rlike.container as rlc
from rpy2.robjects import pandas2ri

# import pandas.rpy2.common as com

r = robjects.r


output = r.source('./ex.R')

from rpy2.robjects.packages import importr
from rpy2.robjects.methods import RS4

# utils = importr('utils')
# utils.chooseCRANmirror(ind=1)
# utils.install_packages('seewave')

warbleR = importr('warbleR')
tuneR = importr('tuneR')
seewave = importr('seewave')
base = importr('base')

start = int(0)
end = int(20)
df = robjects.DataFrame({})
data = rlc.OrdDict([('sound.files', './testFiles/sine.wav'),
                    ('selec', 0),
                    ('start', start),
                    ('end', end)])

# data = {'sound.files' : './testFiles/sine.wav', 'selec' : 0, 'start' : robjects.IntVector(0), 'end': robjects.IntVector(20)}
df = robjects.DataFrame(data)
print(df)

b = robjects.IntVector([0, 22])

tuneR_r = tuneR.readWave(os.path.join('./testFiles/sine.wav'), 0, to = 20, units = "seconds")
# tuneR_rR = r('''tuneR_r''')
# res = r('''resultdf <- as.data.frame(tuneR_rR)''')
dollar = base.__dict__["$"]

# print(tuneR_r@samp.rate)
wavspec = seewave.spec(tuneR_r, plot = False)
wavspecprop = seewave.specprop(wavspec, plot = False)
print(tuneR_r)
print(wavspecprop[0][0])

mean = wavspecprop[0][0]/1000
sd = wavspecprop[1][0]/1000
median = wavspecprop[2][0]/1000
sem = wavspecprop[3][0]
mode = wavspecprop[4][0]/1000
Q25 = wavspecprop[5][0]/1000
Q75 = wavspecprop[6][0]/1000
IQR = wavspecprop[7][0]/1000
centroid = wavspecprop[8][0]/1000
skew = wavspecprop[9][0]
kurt = wavspecprop[10][0]
sfm = wavspecprop[11][0]
sh = wavspecprop[12][0]

ylim = robjects.IntVector((0, 280/1000))

wavefundf = seewave.fund(wavspec, f = 16000, ovlp = 50, fmax = 280, ylim = ylim, plot = False)
# print(wavefundf)

spe = seewave.dfreq(wavspec, f = 16000, plot = False)
print(spe)

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