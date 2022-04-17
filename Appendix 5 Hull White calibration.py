#Librairies

import numpy as np
import pandas as pd
from scipy.optimize import minimize 
import matplotlib.pyplot as plt
                                                
#Function to get the B part in the closed form formula of a ZC price
def B(t,T,a):
    return (1 - np.exp(-a*(T-t)))/a

#Function to get the A part in the closed form formula of a ZC price
def A_wo_ratio_P(t,T,a,sigma, F_t):
    val = B(t,T,a)*F_t - ((sigma**2)/(4*(a**3)))*((np.exp(-a*T) - np.exp(-a*t))**2)*(np.exp(2*a*t) - 1)
    return np.exp(val)

#Function to get the price of a ZC using Hull White model
def Price_ZC(a,sigma,t,T,P_T,P_t,X_t,F_t):
    ratio_P = P_T/P_t 
    return ratio_P * A_wo_ratio_P(t,T,a,sigma, F_t) * np.exp(-B(t,T,a) * X_t)


#Get the data needed : forward exchange rate and ZC curve CHANGE THE PATH
data = pd.read_csv(r"C:\Users\User\Desktop\data_ZC_FWD.csv", 
                       sep=";", skip_blank_lines=True).iloc[:14,:4]

#Formatting the dataframe
data['fwd_fx'] = data['fwd_exchange_rates']
del  data['fwd_exchange_rates']

data = data.iloc[1:,:].reset_index(drop=True)

#Initialize the value of T and P_T
T = max(data['Maturity_year'])
P_T = data.loc[12,"zc_price"]

#Delete the last row of the dataframe
data = data.loc[:11,:]

#From dataframe to dict
dict_data = {data.columns[i] : 
             list(data[data.columns[i]]) for i in range(len(data.columns))}
    
#Create a new column with the right inputs
list_tup_parameters = []
for i in range(len(data)):
    list_tup_parameters.append((data.loc[i,'Maturity_year'],
                                T,
                                P_T,
                                data.loc[i,'zc_price'],
                                data.loc[i, 'fwd_fx'],
                                data.loc[i, 'fwd_zc']))
    
dict_data['inputs'] = list_tup_parameters

#Dataframe with the inputs
to_show_df = pd.DataFrame(dict_data)

# Define variables to be used in optimization

mat = to_show_df['Maturity_year'].values
tx_zc = to_show_df['zc_price'].values
fwd_fx = to_show_df['fwd_fx'].values
fwd_zc = to_show_df['fwd_zc'].values


#Initialize the parameters

params = {'a' : {"x0" : 0.35,},
          'sigma' : {"x0" : 0.45,}
          }

x0 = np.array([param["x0"] for key, param in params.items()])

#Function we try to minimize
def SqErr(x):
    a, sigma = [param for param in x]    
    diff = tx_zc - Price_ZC(a = a,
                                   sigma = sigma,
                                   t = mat,
                                   T = np.tile(T,len(mat)),
                                   P_T = np.tile(P_T,len(mat)),
                                   P_t = tx_zc,
                                   X_t = fwd_fx,
                                   F_t = fwd_zc)
    squared_mean_diff = (diff**2)   
    squared_mean_diff = squared_mean_diff[np.isnan(squared_mean_diff) == False]
    return sum((squared_mean_diff)/(len(tx_zc) - 2 ))

result = minimize(SqErr, x0, tol = 1e-3, method='SLSQP', 
                  options={'maxiter': 1e5 })

true_alpha, true_sigma = result.x

print("\ntrue_alpha = %f, true_sigma = %f" % (true_alpha, true_sigma))


#Data to plot the graph
model = [Price_ZC(a = true_alpha, sigma = true_sigma, t = mat[i],
         T = T, P_T = P_T,
         P_t = tx_zc [i], X_t = fwd_fx[i], F_t = fwd_zc[i])   for i in range(len(mat))  ]



real_zc = data["zc_price"].values
theo_zc = model
mat= data["Maturity_year"].values 


error_arr = np.array(real_zc - theo_zc, dtype=float)
mat =[str(int(mat[i])) for i in range(len(mat)) ]

plt.figure(figsize=(6,4), dpi=700)
plt.plot(mat,
         theo_zc, label = 'Hull White Model', color = 'red',
         linestyle = '--', marker = 'x')
plt.plot(mat, real_zc, label = 'Market price', color = 'blue', 
         linestyle = '-.', marker = 'x')
plt.plot(mat,
         error_arr, label = 'Error Spread', color = 'green', linestyle = ':',
         alpha = 0.5, marker ='x')

plt.fill_between(mat, error_arr, 0, color='green', alpha=0.15)

plt.axhline(0, 0.05, 0.95, color = 'gray')
plt.ylim(top=1.1)
plt.ylim(bottom=-0.1)
plt.title("Hull White calibration on EUR/USD calls")
plt.xlabel("Maturity")
plt.ylabel("Prices")
plt.legend()
plt.grid(True)
plt.show()

