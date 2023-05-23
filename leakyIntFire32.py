'''
32nd iteration for LIF

Additionally this code is found on Github under Foelley
as wanted to have an isolated copy connected with my thesis
'''
import time
start_time = time.time()

#Importation of libraries
import numpy as np
import matplotlib.pyplot as mpl
import numpy.random as rndm
import numpy.fft as trans

#Setting seeds for consistency
rndm.seed(11723)

#Function for plotting the power spectrum using fourier data
def powerFun(fourData, timeData, plotName):
    i = 0
    #Removing first data point as comes from infinite
    data = fourData[1:]
    tempLen = len(data)
    power = []
    #limit = len(data)
    #'''
    #Dealing with symmetry, neglecting middle point which should be fine
    limit = 0
    if tempLen % 2 == 0:
        limit = tempLen/2
    else:
        limit = (tempLen + 1)/2
    #'''
    temp = 0
    i = 0
    while i < limit:
        pow1 = np.absolute(data[i])**2
        pow2 = np.absolute(data[tempLen - 1 - i])**2
        #if pow1 != pow2:
            #print("Huh", i, tempLen, pow1, pow2)
        temp  = (pow1 + pow2)/2
        power.append(temp)
        i = i + 1
    
    timeCut = timeData[:int(limit)]
    #1000 is to convert from ms-1 to s-1
    #Original Power is 1000/(timeCut[1]-timeCut[0]) with symmetry, needs to be
    #cut in half as well
    freq = np.linspace(0, 1000/(2*(timeCut[1]-timeCut[0])), len(timeCut))
    
    
    mpl.title("Power")
    mpl.xlabel("Frequency (Hz)")
    mpl.ylabel("Power Function")
    mpl.plot(freq, power)
    #print(freq)
    #print(freq[17])
    #mpl.savefig(plotName, dpi=137)
    #mpl.close()
    mpl.show()
    return power
    

#Function for the firerate, replacing the code under graphing section
#Function for the firerate, replacing the code under graphing section
def fireRateFun(spikeTrain, eulTimes, numEx, numNeur, h = 1):
    #Variables for scanning window
    #rateSizeNEEDS to be even
    rateSize = 20
    winExRates = [0 for i in range(rateSize)]
    winInRates = [0 for i in range(rateSize)]
    #Note window is size rateSize*h from eul, so rateSize*0.5 by default
    rastTime = [[] for i in range(numNeur)]
    #i = 1 as can't spike at i = 0, and messes with modulo check
    i = 1
    j = 0
    numExSpikes = 0
    numInhSpikes = 0
    #Arrays for full fire rate and individual (prob more than needed)
    exRate = []
    exRateInd = []
    inhRate = []
    inhRateInd = []
    rateTime = []
    while i < len(eulTimes):
        while j < numEx:
            if spikeTrain[j][i] != 0:
                rastTime[j].append(eulTimes[i])
                numExSpikes = numExSpikes + 1
            j = j + 1
        while j < numNeur:
            if spikeTrain[j][i] != 0:
                rastTime[j].append(eulTimes[i])
                numInhSpikes = numInhSpikes + 1
            j = j + 1
        #Calculating rates and times, using time for neuron 1 as the example
        #Slight differences from adding h might change times for each neuron and not 
        #keep them equal
        
        #*1000 for units, to convert inverse ms to inverse s
        #This info is for a rate size of 1 from before, every 1s
        tempExRate = 1000*numExSpikes/(h)
        tempInhRate = 1000*numInhSpikes/(h)
        exRateInd.append(tempExRate)
        inhRateInd.append(tempInhRate)
        tempTime = eulTimes[i]
        rateTime.append(tempTime)
        
        #Dealing with moving window of rateSize
        #Even if h isn't 1, accounted for above
        winExRates.append(tempExRate)
        winExRates.pop(0)
        winInRates.append(tempInhRate)
        winInRates.pop(0)
        tempExRate = np.mean(winExRates)
        exRate.append(tempExRate)
        tempInhRate = np.mean(winInRates)
        inhRate.append(tempInhRate)
        
        numExSpikes = 0
        numInhSpikes = 0
        i = i + 1
        j = 0
    
    #Could centralise, would need to get rid of some values but
    #avoiding as easiest to manage with all values
    
    exFour = trans.fft(exRate)
    inFour = trans.fft(inhRate)
    
    return exRate, inhRate, exRateInd, inhRateInd, rateTime, rastTime, exFour, inFour

def fireRatePlotFun(exRate, inhRate, rateTime, numEx, numIn, exPlotName, inPlotName):
    mpl.title("Firing Rate of Excitatory Neurons per\nExcitatory Neuron")
    mpl.xlabel("Time (ms)")
    mpl.ylabel("Average Firing rate (Hz)")
    #print(len(exRate))
    sclExRate = [exRate[i]/numEx for i in range(len(exRate))]
    mpl.plot(rateTime, sclExRate)
    #mpl.savefig(exPlotName, dpi=137)
    #mpl.close()
    mpl.show()
    
    mpl.title("Firing Rate of Inhibitory Neurons per\nInhibitory Neuron")
    mpl.xlabel("Time (ms)")
    mpl.ylabel("Average Firing rate (Hz)")
    sclInRate = [inhRate[i]/numIn for i in range(len(inhRate))]
    mpl.plot(rateTime, sclInRate)
    #mpl.savefig(inPlotName, dpi=137)   
    #mpl.close()
    mpl.show()
    return
#Function for determining the synchrony
#Voltages are passed in as voltages[neuron][time]
#Using np.var to calculate variance rather than explicit
#time averages used in paper
#Can change initial i, likely int(np.floor(numTime/3))
def synchFun(voltages, numNeur):
    #Decleration of Variables
    synch = 0
    sigTot = 0
    sigI = []
    vTot = []
    numTime = len(voltages[0])
    tempV = 0
    
    #Determing vTot and sigTot
    i = int(np.floor(numTime/3))
    j = 0
    while i < numTime:
        while j < numNeur:
            tempV = tempV + voltages[j][i]
            j = j + 1
        j = 0
        #As vTot[time] = sum(vInd[time])*(1/numNeur) from paper
        vTot.append(tempV/numNeur)
        tempV = 0
        i = i + 1
    sigTot = np.var(vTot)
    
    #Determining sigI's, may be able to make more efficient
    #Only looking at same time as above
    i = 0
    while i < numNeur:
        tempVar = np.var(voltages[i][int(np.floor(numTime/3)):])
        sigI.append(tempVar)
        i = i + 1
    #Determining the synchrony term
    synch = numNeur*sigTot/(np.sum(sigI))
    return synch

#Functions to describe the current for inhibitory and excitatory
#Units are in nA
def eleCurInFun(eleCur = 1.51, amp = 0, hFreq = 10, t = 0):
    #No Spikes when = 1.4, spikes with 1.6, both cases no noise
    #eleCur of 1.5 seems to be the rheobase
    eleCur = eleCur + amp*(np.sin(2*np.pi*hFreq*t))
    return eleCur

def eleCurExFun(eleCur = 1.51, amp = 0, hFreq = 10, t = 0):
    #No Spikes when = 1.4, spikes with 1.6, both cases no noise
    #eleCur of 1.5 seems to be the rheobase
    eleCur = eleCur + amp*(np.sin(2*np.pi*hFreq*t))
    return eleCur

#Function for creating rasterplots
def rasterFun(rastTime, numEx, numNeur, rastName):
    #Color coordinating the events, Excitatory in Blue, Inhibitory in Red
    fig, ax = mpl.subplots()
    fig.set_size_inches(14.02, 9.39)
    colAr = []
    i = 0
    while i < numEx:
        colAr.append('blue')
        i = i + 1
        
    while i < numNeur:
        colAr.append('red')
        i = i + 1
        
    mpl.title("Rasterplot")
    mpl.xlabel("Time (ms)")
    mpl.ylabel("Corresponding Neuron")
    mpl.eventplot(rastTime, color = colAr)
    #mpl.savefig(rastName, dpi=137)    
    #mpl.close()
    mpl.show()    
    return

#Function for plotting voltages
def voltPlotFun(volts, times, numEx, numNeur, voltName):
    #Graphing voltages against time
    #How Many Excitatory and Inhibitory to plot, no checks for consistency
    #Comment out max values of numStuff, and put value before this
    fig, ax = mpl.subplots()
    fig.set_size_inches(14.02, 9.39)      
    sampEx = 1#numEx
    sampIn = 1#numNeur - numEx
    i = 0
    while i < sampEx:
        mpl.plot(times, volts[i], color = 'blue')
        i = i + 1
    #As volt[numEx] == first inhibitory neuron voltage
    i = numEx
    while i < numEx + sampIn:
        mpl.plot(times, volts[i], color = 'red')
        i = i + 1
    mpl.title("Voltage for All Neurons")
    mpl.xlabel("Time (ms)")
    mpl.ylabel("Voltage (mV)")
    #mpl.savefig(voltName, dpi=137)
    #mpl.close()
    mpl.show()
    return

#Function for determing weighting coefficients
def weightNeurFun(numNeur, numEx, sclCoef):
    #Values for probability of neurons being connected
    eeProb = 1#0.1
    eiProb = 1#0.5
    ieProb = 1#0.5
    iiProb = 0#0.3
    #Values for mean weight for neuron types
    #Values taken from paper in overview
    eeMean = 10#53#1.6
    eiMean = 50#100#3
    ieMean = -30#-157#-4.7
    iiMean = 0#-0.13
    #Standard deviations around each randomly generated spike
    eeStd = 0
    eiStd = 0
    ieStd = 0
    iiStd = 0
    #Values for the weighting matrix
    wghtNeurOut = [[0 for i in range(numNeur)] for i in range(numNeur)]
    i = 0
    j = 0
    while i < numNeur:
        #Weighting of neuron i with the excitatory neurons
        while j < numEx:
            tempRnd = rndm.uniform()
            if i < numEx:
                if tempRnd < eeProb:
                    wghtNeurOut[i][j] = rndm.normal(loc=eeMean, scale=eeStd)*sclCoef[i]/numEx
            else:
                if tempRnd < eiProb:
                    #Lower left quadrant is ei in literature, so keeping with this
                    wghtNeurOut[i][j] = rndm.normal(loc=eiMean, scale=eiStd)*sclCoef[i]/numEx
            j = j + 1
        #Weighting of neuron i with the inhibitory neurons
        while j < numNeur:
            tempRnd = rndm.uniform()
            if i < numEx:
                if tempRnd < ieProb:
                    #Upper right quadrant is ie in literature, so keeping with this
                    wghtNeurOut[i][j] = rndm.normal(loc=ieMean, scale=ieStd)*sclCoef[i]/numIn
            else:
                if tempRnd < iiProb:
                    wghtNeurOut[i][j] = rndm.normal(loc=iiMean, scale=iiStd)*sclCoef[i]/numIn      
            j = j + 1
        i = i + 1
        j = 0    
    return wghtNeurOut

#Function that the differential equation follows.
def voltPrimeFun(vRest, resMem, eleCurT, vNoiseT, voltT, memTimeCon, wghtNeur, numSpike, h, time):
    weightVal = 0
    m = 0
    #Always adding as negative weights for are now accounted for
    #Dividing by numIn and numEx as what the Scott paper does
    '''
    while m < numEx:
        weightVal = weightVal + sclCoef*wghtNeur[m]*numSpike[m]/numEx
        m = m + 1
    while m < (numEx + numIn):
        weightVal = weightVal + sclCoef*wghtNeur[m]*numSpike[m]/numIn
        m = m + 1
    '''
    weightVal = np.dot(wghtNeur, numSpike)
    #Setting so weights are all 0 until 100ms
    if time*h < 100:
        weightVal = 0
    #Dividing by h to scale
    fVal = (vRest + vNoiseT/(np.sqrt(h)) - voltT + (resMem*eleCurT) + weightVal/h)/memTimeCon
    return fVal

#''' 
#Each of the variable passed in will be an array of size NumNeur
def eulerFun(vRest, resMem, memTimeCon, vZero, vThresh, sclCoef, wghtNeur, numEx, h = 1, tInit = 0, tFinal = 3000, noiseStrEx = 2, noiseStrIn = 2, varA = None, varB = None):
    numNeur = len(vThresh)
    numIn = numNeur - numEx
    numVal = int((tFinal - tInit)/h)
    #Initiallizing each matrix
    volts = [[0 for i in range(numVal)] for i in range(numNeur)]
    times = [0 for i in range(numVal)]
    spikeTrain = [[0 for i in range(numVal)] for i in range(numNeur)]
    noise = [[0 for i in range(numVal)] for i in range(numNeur)]
    
    #Calculating Noise. Can scale as needed
    #Also randomizing initial voltages between
    #Checking the number of entries there will be. Not applying noise to initial values
    #As Normal distributed, noiseStr is just the standard deviation of the noise
    i = 0
    j = 0
    while i < numEx:
        while j < numVal:
            noise[i][j] = np.random.normal()*noiseStrEx
            j = j + 1
        volts[i][0] = rndm.uniform(low = vZero[i], high = vThresh[i])
        j = 0
        i = i + 1
    while i < numNeur:
        while j < numVal:
            noise[i][j] = np.random.normal()*noiseStrIn
            j = j + 1
        #volts[i][0] = rndm.uniform(low = vZero[i], high = vThresh[i])
        volts[i][0] = rndm.uniform(low = -51, high = vThresh[i])
        j = 0
        i = i + 1
    k = 0
    i = 0
    #tss = [0 for i in range(numNeur)]
    #Calculating and appending the voltages.
    #Shifting this as no longer append, so need -1
    while i < numVal - 1:
        #Check to see if previous voltage > threshold, if so resetting the voltage to its rest value
        #Reminder: -46.999 > -47
        #Checking if ANY neuron spiked 1 unit of time ago, but preventing any spikes for first 100ms
        j = 0
        spikeCount = [0 for i in range(numNeur)]
        while j < numNeur:
            if (volts[j][i] > vThresh[j]):
                volts[j][i] = vRest[j]
                #Keeping track of the spike, associating with value over vThresh
                spikeTrain[j][i] = 1
                spikeCount[j] = 1
                #tss[j] = 0      
            #This was moved above in LIF11
            #spikeCount[j] = spikeTrain[j][i]
            j = j + 1
            
        while k < numEx:
            #Calculating Voltages
            volts[k][i+1] = volts[k][i] + h*voltPrimeFun(vRest[k], resMem[k], eleCurExFun(eleCur = exCur, amp = varA, hFreq = varB, t = times[i]), noise[k][i], volts[k][i], memTimeCon[k], wghtNeur[k], spikeCount, h, i)
            #tss[k] = tss[k] + h    
            k = k + 1
        
        while k < numNeur:
            #Calculating Voltages
            volts[k][i+1] = volts[k][i] + h*voltPrimeFun(vRest[k], resMem[k], eleCurInFun(eleCur = inCur, amp = varA, hFreq = varB, t = times[i]), noise[k][i], volts[k][i], memTimeCon[k], wghtNeur[k], spikeCount, h, i)
            #tss[k] = tss[k] + h    
            k = k + 1
        times[i+1] = times[i] + h
        i = i + 1
        k = 0
        #print(i)
    return volts, times, spikeTrain
#'''


#'''
#Decleration of Variables
#Arrays are used as may look at different types of
#Nuerons with different firing conditions
#Number of neurons in the model, ensure = numIn + numEx
numNeur = 1000#200
#Number of inhibitory cells
numIn = 200#40
#Number of excitatory cells
numEx = 800#160
#Potential that values are reset to after a spike, units of mV
vRest = [-65 for i in range(numNeur)]
#Potential that values are initially set to, units of mV
vZero = [-65 for i in range(numNeur)]
#Potential that values are reset to after a spike, units of mV
vThresh = [-50 for i in range(numNeur)]
#Value for Membrane Resistance (ie 1/ Membrane Conductance), units of MOhms
resMem = [10 for i in range(numNeur)]
#Value for the membrane time constant, units of ms
#Changing this changes firerate, increased
memTimeCon = [30 for i in range(numNeur)]
#Value for the coupling gain
sclCoef = [1 for i in range(numNeur)]
#Values for the weighting matrix
wghtNeur = weightNeurFun(numNeur, numEx, sclCoef)
#Using Reshape to ensure about expected behavior with hit or miss method
#testWght = np.reshape(wghtNeur, (numNeur, numNeur))
#print(testWght)

#'''
#Heterogeneity stuff for threshhold values
spreadE = [0 for i in range(1)]
spreadI = [0 for i in range(1)]
m = 0
n = 0
while m < len(spreadE):
    #Dealing with threshold spreads
    vThresh = []
    n = 0
    while n < numEx:
        tempVal = rndm.normal(loc = -50, scale = spreadE[m])
        #Avoiding threshold too close to rest
        if (tempVal - vRest[n]) < 1:
            tempVal = -50
        vThresh.append(tempVal)
        n = n + 1
    while n < numNeur:
        tempVal = rndm.normal(loc = -50, scale = spreadI[m])
        #Avoiding threshold too close to rest
        if (tempVal - vRest[n]) < 1:
            tempVal = -50            
        vThresh.append(tempVal)
        n = n + 1
    m = m + 1
#'''

exCur = 1.51
inCur = 1.51

#Note, freq is in ms-1
hFreq = 0.012
amp = 0#0.8

eulVolts, eulTimes, spikeTrain = eulerFun(vRest, resMem, memTimeCon, vZero, vThresh, sclCoef, wghtNeur, numEx, varA = amp, varB = hFreq)
syncEx = synchFun(eulVolts, numEx)
print("Calculated Synchrony For Excitatory Neurons is:", syncEx)
print("--- %s seconds ---" % (time.time() - start_time))
#testWght = np.reshape(eulVolts, (numNeur, len(eulVolts[0])))
#print(testWght)
#'''

#'''
print("Starting Graphs:")

#Names are int and *100 as 1.45 is recognized as a file .45

tempName = "v" + "A" + str(int(amp*100)) + "Ome" + str(int(hFreq*100))
voltName = str(tempName)
voltPlotFun(eulVolts, eulTimes, numEx, numNeur, voltName)

exRate, inhRate, exRateInd, inhRateInd, rateTime, rastTime, exFour, inFour = fireRateFun(spikeTrain, eulTimes, numEx, numNeur)
tempName = "r" + "A" + str(int(amp*100)) + "Ome" + str(int(hFreq*100))
#print(tempName)
rasterName = str(tempName)
#rasterFun(rastTime, numEx, numNeur, rasterName)

print("Average Excitatory Firing Rate per Excitatory Neuron = ", np.mean(exRateInd)/numEx)
print("Average Inhibitory Firing Rate per Inhibitory Neuron = ", np.mean(inhRateInd)/numIn)
y = 0
z = 0
k = 0
exPlotName = 'fE' + 'Amp' + str(y) + 'Frq' + str(z) + 'Run' + str(k)
inPlotName = 'fI' + 'Amp' + str(y) + 'Frq' + str(z) + 'Run' + str(k)
fireRatePlotFun(exRate, inhRate, rateTime, numEx, numIn, exPlotName, inPlotName)


temp = powerFun(exFour, rateTime, "Testing")
temp = powerFun(inFour, rateTime, "Testing")
#print(rastTime)
rasterFun(rastTime, numEx, numNeur, rasterName)

#'''