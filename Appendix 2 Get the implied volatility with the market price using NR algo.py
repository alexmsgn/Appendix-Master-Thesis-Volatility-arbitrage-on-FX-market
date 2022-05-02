#Libraries
import numpy as np
import scipy.stats as si
import math as math


#Function to get the distribution function of the normal law cumulated 
def Normsdistcumulated(x):
    Normsdistcumulated = si.norm.cdf(x,0.0,1.0)
    return (Normsdistcumulated)

#Function to get the distribution function of the normal law 
def Normdistdensity(x):
    Normdistdensity = si.norm.pdf(x,0.0,1.0)
    return (Normdistdensity)


#Function to get the GarmanKohlhagen price of an European call
def GarmanKohlhagenCallPrice(Xt, K, T, r_d, r_f, vol,t):
    dplus = (np.log(Xt/K)+(r_d-r_f+0.5*vol**2)*(T-t))/(vol*np.sqrt(T-t))
    dminus = (np.log(Xt/K)+(r_d-r_f-0.5*vol**2)*(T-t))/(vol*np.sqrt(T-t))
    GarmanKohlhagenCallPrice = Xt*np.exp(-r_f*(T-t))*Normsdistcumulated(dplus)-K*np.exp(-r_d*(T-t))*Normsdistcumulated(dminus)
    return(GarmanKohlhagenCallPrice)

#Inputs YOU MUST FILL (floats)
Xt = 1.16
K = 1.14
T = 2
t= 1
r_d = 0.02
r_f = 0.05
market_price= 0.14
epsilon = 0.0000001

#Function to find the implied volatility of an option using Newton Raphson algorithm
def implied_vol_gk (Xt, K, T, r_d, r_f, market_price,t,epsilon):
    
    max_iter=200
    old_vol=0.9
    temp = 1
    
    print('Iteration', temp, ':', old_vol)
    
    for i in range(max_iter):
        gk_price= GarmanKohlhagenCallPrice(Xt, K, T, r_d, r_f, old_vol,t)
        dminus = (np.log(Xt/K)+(r_d-r_f-0.5*old_vol**2)*(T-t))/(old_vol*np.sqrt(T-t))
        f_prime = math.exp(-r_d*(T-t))*K*math.sqrt((T-t))*Normdistdensity(dminus)
        f= gk_price - market_price
        new_vol= old_vol-f/f_prime
        temp += 1
        print('Iteration', temp, ':', new_vol)

        if(abs(old_vol-new_vol)<epsilon):
            break
        
        old_vol = new_vol
        implied_vol = new_vol 
        
    print('\nFinal result of implied volatility :  '+ str(round(implied_vol*100,4)) +'%')
    return implied_vol


implied_vol_gk(Xt, K, T, r_d, r_f, market_price,t,epsilon)

