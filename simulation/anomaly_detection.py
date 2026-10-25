import time
import math
import board 
import adafruit_bpm3xx as ad

'''
This is a list of possibly useful (static) functions/methods. 
checkSensors() performs the fast and slow loops.  It didn't make
sense to initialize sensors here, so I opted not to.

All associated variables use placeholder values that can be easily changed or accessed.
--> For example, "getV() and getA() are not defined, but should be accessible from 
    weather data analysis"
--> Thresholds and loop statements are arbitrarily set.
The amount of memory allocated for lists are fixed, though.


'''

#Frequency in hz
Fs = 20                             #sample frequency
f0 = 0.2*Fs                         #cutoff frequency
Q = math.sqrt(2)/2                  #quality factor
omega = 2*math.pi * f0/Fs           #normal frequency
alpha = math.sin(omega)/(2*Q)       #bandwidth scaling
a0 = 1 + alpha                      # a --> biquad filtered altitude
b0 = (1-math.cos(omega))/(2*a0)     # b --> unfilitered altitude 
b1 = (1-math.cos(omega))/a0         # 0 - 2 (0 is most recent) 
b2 = (1-math.cos(omega))/(2*a0)      
a1 = -2*math.cos(omega)/a0         
a2 = (1-alpha)/a0                  
k = (b0+b1+b2)/(1+a1+a2)            #optional 
a0 /= a0                             

abs_threshold = 1                     #when abs sum of residuals are too high, 1 is a placeholder
true_cutoff = -1                      #when true sum of residuals is too low, -1 is a placeholder
f_loop = 40                           #how frequent the fast loop executes

def checkSensors(sensors: list) -> None:
    h = 0       #initial altitude
    tl = 0      #indefinite time loop constant

    quadFilter = [[0]*4]*3  #setup biquad filter

    altLst = [[0]*4]*3      #setup sensor readings

    while t < 30:           #using t instead of tl for now to reduce runtime for quicker code debugging 
        t = 0
        residuals = []      #tracking residuals
        res_abs_sum = 0
        res_sum = 0
        while t < 0.05*f_loop:
            del altLst[0]
            altLst.append([alt1, alt2, alt3, alt4]) #sensor.readAltidude()
            if isAnomalous(altLst[2]):
                pass #initiate storm mode
            del quadFilter[0]
            quadFilter.append([Quad(altLst[0][i],altLst[1][i],altLst[2][i],quadFilter[0][i],quadFilter[1][i]) for i in range(len(altLst[0]))])
            altEXP = getAltitude(h, getV(t), getA(t), 0.05) #access public function velocity and acceleration
            res_calc = altEXP - 0.25 * (quadFilter[2][0] + quadFilter[2][1] + quadFilter[2][2] + quadFilter[2][3])
            residuals.append(abs(res_calc))
            res_abs_sum += abs(res_calc)
            res_sum += res_calc
            t += 0.05
            time.sleep(0.05)
        if len(residuals) <= 0:
            print("Issue calculating residuals.")
            continue
        res_abs_avg = res_abs_sum/len(residuals)    #residuals should have nonzero length unless all sensors fail
        res_avg = res_sum/len(residuals)
        if res_abs_avg > threshold:
            pass #initiate storm mode
            f_loop = 200
            if res_avg < true_cutoff:
                pass #initiate dropping maneuver
        else:
            f_loop = 40
            #exit storm mode if in storm mode
            

#Biquad filter function
def Quad(x1: float, x2: float, x3: float, y1: float, y2: float) -> float: 
    return b0*x1 + b1*x2 + b2*x3 - a0*y1 - a1*y2

def getAltitude(h0: float, v: float, a: float, dt:float) -> None: #get past history
    h1 = 1/2*(a)*(dt)^2 + (v)*dt + h0

#Accesses Temperature 
def getTemp():
    temp1 = 0       #sensor.readTemperature()
    temp2 = 0
    temp3 = 0
    temp4 = 0
    T = 0.25*(temp1 + temp2 + temp3 + temp4)
    return [T, [temp1, temp2, temp3, temp4]]

#Accesses Pressure
def getPressure():
    press1 = 0      #sensor.readPressure()
    press2 = 0
    press3 = 0
    press4 = 0
    P = 0.25*(press1 + press2 + press3 + press4)
    return [P, [press1, press2, press3, press4]]

#Checking altitude differences to see if sensors are picking up anomalous data
def isAnomalous(a: list) -> bool:
    mean = 0.25*(a[0] + a[1] + a[2] + a[3])
    for i in range(len(a)):
        if(abs(a[i] - mean) >= 1.2):
            return True
    return False
