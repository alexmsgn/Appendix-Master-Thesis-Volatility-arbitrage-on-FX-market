#Librairies

import scipy.stats as si
import numpy as np
import pandas as pd

#Function to get the inverse of the density function of the normal law 
def Normsdist(x):
    Normsdist = si.norm.ppf(x,0.0,1.0)
    return (Normsdist)

#initialize the parameters YOU MUST FILL (floats)
rd = 0.02
rf = 0.05
Xt = 1.16

#get the volatility data CHANGE THE PATH
df = pd.read_csv(r"C:\Users\User\Desktop\EURUSD_Call_Mid.csv",sep=";")

#Create a dataframe
implied_vol=pd.DataFrame(data = df)

#Keep the first row, corresponding to the volatility of shortest maturity for each strike (1 day)
data_wo_time = implied_vol.iloc[:,1:] 

#Create an array
values=np.array(data_wo_time.iloc[:1,:])

t = 1/252
deltas =  [0.05,0.1,0.15,0.25,0.35,0.5,0.65,0.75,0.85,0.90,0.95]

#Get the inverse of the density function of the normal law for each strike
arr_inversed_d=np.array([Normsdist(val)for val in deltas])

#Deomposition in two components
log_k_S_first_comp= - arr_inversed_d*values*np.sqrt(t)
log_k_S_second_comp=((values**2)/2 + rd - rf)*t

#Get the strikes values
log_K_S = log_k_S_first_comp+log_k_S_second_comp
K= np.exp(log_K_S)*Xt
