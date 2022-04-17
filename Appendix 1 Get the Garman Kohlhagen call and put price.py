#Libraries
import numpy as np
import scipy.stats as si

#Function to get the distribution function of the normal law 
def Normsdist(x):
    Normsdist = si.norm.cdf(x,0.0,1.0)
    return (Normsdist)
    
#Function to get the GarmanKohlhagen price of an European call
def GarmanKohlhagenCallPrice(Xt, K, T, r_d, r_f, vol,t):
    dplus = (np.log(Xt/K)+(r_d-r_f+0.5*vol**2)*(T-t))/(vol*np.sqrt(T-t))
    dminus = (np.log(Xt/K)+(r_d-r_f-0.5*vol**2)*(T-t))/(vol*np.sqrt(T-t))
    GarmanKohlhagenCallPrice = Xt*np.exp(-r_f*(T-t))*Normsdist(dplus)-K*np.exp(-r_d*(T-t))*Normsdist(dminus)
    return(GarmanKohlhagenCallPrice)

#Function to get the GarmanKohlhagen price of an European put
def GarmanKohlhagenPutPrice(Xt, K, T, r_d, r_f, vol,t):
    dplus = (np.log(Xt/K)+(r_d-r_f+0.5*vol**2)*(T-t))/(vol*np.sqrt(T-t))
    dminus = (np.log(Xt/K)+(r_d-r_f-0.5*vol**2)*(T-t))/(vol*np.sqrt(T-t))
    GarmanKohlhagenPutPrice = K*np.exp(-r_d*(T-t))*Normsdist(-dminus)-Xt*np.exp(-r_f*(T-t))*Normsdist(-dplus)
    return(GarmanKohlhagenPutPrice)

#Inputs YOU MUST FILL (floats)
Xt =
K = 
T = 
t= 
r_d = 
r_f = 
vol = 


#Get the outputs 
print("The price of a European Call option of a maturity T= {0} at time t= {1} using Garman & Kohlhagen Model is:{2:.5}".format(T,t,(GarmanKohlhagenCallPrice(Xt, K, T, r_d, r_f, vol,t))))
print("The price of a European Put option of a maturity T= {0} at time t= {1} using Garman & Kohlhagen Model is:{2:.5}".format(T,t,(GarmanKohlhagenPutPrice(Xt, K, T, r_d, r_f, vol,t))))