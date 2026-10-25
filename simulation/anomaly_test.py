import time
import math
import random as r


Fs = 20 
f0 = 0.2*Fs 
Q = math.sqrt(2)/2
omega = 2*math.pi * f0/Fs       
alpha = math.sin(omega)/(2*Q)   
a0 = 1 + alpha                  
b0 = (1-math.cos(omega))/(2*a0)      
b1 = (1-math.cos(omega))/a0          
b2 = (1-math.cos(omega))/(2*a0)      
a1 = -2*math.cos(omega)/a0         
a2 = (1-alpha)/a0                  
k = (b0+b1+b2)/(1+a1+a2)        
a0 /= a0                  

def main():
    quadFilter = [[0.2*k]*4]*3
    altLst = [[0.2]*4]*3
    del altLst[0]
    altLst.append([.25, .25, .25, .25]) #sensor.readAltidude()
    del quadFilter[0]
    quadFilter.append([Quad(altLst[0][i],altLst[1][i],altLst[2][i],quadFilter[0][i],quadFilter[1][i]) for i in range(len(altLst[0]))])
    print(altLst)
    print(quadFilter)

    iteration = 25
    i = 0
    while i < iteration:
        del altLst[0]
        altLst.append([altLst[1][0]+1/100*r.random(), altLst[1][1]+1/100*r.random(), altLst[1][2]+1/100*r.random(), altLst[1][3]+1/100*r.random()]) #sensor.readAltidude()
        del quadFilter[0]
        quadFilter.append([Quad(altLst[0][i],altLst[1][i],altLst[2][i],quadFilter[0][i],quadFilter[1][i]) for i in range(len(altLst[0]))])
        print(altLst)
        print(quadFilter)
        i += 1


def Quad(x1: float, x2: float, x3: float, y1: float, y2: float) -> float: 
    return b0*x1 + b1*x2 + b2*x3 - a0*y1 - a1*y2

main()
