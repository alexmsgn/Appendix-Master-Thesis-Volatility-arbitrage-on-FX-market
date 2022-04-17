#Librairies

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import datetime as dt


#Function to get the heston characteristic function f(phi)
def heston_characteristic_function(phi, S0, v0, kappa, theta, sigma, rho, lambd, tau, r):
    
    #Initialization of constants a and b
    a = kappa*theta
    b = kappa+lambd
   
    #Definition of the parameter d given b and phi
    d = np.sqrt( (rho*sigma*phi*1j - b)**2 + (phi*1j+phi**2)*sigma**2 )
   
    #Definition of the parameter g given phi, b and d
    g = (b-(rho*sigma*phi*1j)+d)/(b-(rho*sigma*phi*1j)-d)
   
    #calculation of the characteristic function using a decomposition to simplify
    first_term = np.exp(r*phi*1j*tau)
    second_term  = S0**(phi*1j) * ( (1-g*np.exp(d*tau))/(1-g) )**(-2*a/sigma**2)
    third_term = np.exp(a*tau*(b-(rho*sigma*phi*1j)+d)/sigma**2 + v0*(b-(rho*sigma*phi*1j)+d)*( (1-np.exp(d*tau))/(1-g*np.exp(d*tau)) )/sigma**2)
    return first_term*second_term*third_term

#Function to get the function to integrate in the call price formula
def integral_function(phi, S0, v0, kappa, theta, sigma, rho, lambd, tau, r, K):
    
    args = (S0, v0, kappa, theta, sigma, rho, lambd, tau, r)
    numerator = np.exp(r*tau)*heston_characteristic_function(phi-1j,*args) - K*heston_characteristic_function(phi,*args)
    denominator = 1j*phi*K**(1j*phi)
    return numerator/denominator

#Function to obtain the Heston price using the close form solution
def heston_price(S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r):
    args = (S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r)
    
    #Getting the real part if the integrated function using "quad" to integrate
    #between the boundaries [0,10000]
    real_integral, err = np.real(quad(integral_function, 0, 1000, args=args))
    #Hestion price
    return (S0 - K*np.exp(-r*tau))/2 + real_integral/np.pi

#Initial values of the parameters 
params = {"v0": {"x0": 0.1,},
          "kappa": {"x0": 0.3,},
          "theta": {"x0": 0.05,},
          "sigma": {"x0": 0.8,},
          "rho": {"x0": -0.8,},
          "lambd": {"x0": 0.03,},
          }

x0 = np.array([param["x0"] for key, param in params.items()])

# Define variables to be used in optimization
S0_arr = np.tile(1.0914,9)
r_arr = np.array([0.17,0.78,1.22,1.565,1.91,1.94,1.97,2.03, 2.13])
K_arr = np.tile(1.0914,9)
tau_arr = np.array([1/12,1/2,1,2,3,4,5,7,10])
P_arr = np.array([0.022528,0.35246,0.76919,1.043686396,1.0879,1.090934565,1.0913, 1.0914,1.091399999])

#Function to minimize the mean squared error between Market price and Heston price
def SqErr(x):
    v0, kappa, theta, sigma, rho, lambd = [param for param in x]    
    heston_price_vect = np.vectorize(heston_price)
    diff =  P_arr - heston_price_vect(v0 = v0, kappa = kappa, theta = theta,
                                sigma = sigma, lambd = lambd, rho = rho,
                                S0 = S0_arr, K = K_arr, tau = tau_arr, r = r_arr)
 
    mean_squared_err = sum(diff**2)/ len(P_arr)
     
    return mean_squared_err


print(dt.now())
result = minimize(SqErr, x0, tol = 1e-3, method='SLSQP',
                  options={'maxiter': 100 })


true_v0, true_kappa, true_theta, true_sigma, true_rho, true_lambd = result.x
print("\ntrue_v0 = %f, true_kappa = %f, true_theta = %f, true_sigma = %f,true_rho = %f, true_lambd = %f" % (true_v0, true_kappa, true_theta, true_sigma, true_rho, true_lambd))

#Data to plot the graph
list_theoretical_heston = [heston_price(S0 = S0_arr[i],
                                        K = K_arr[i],
                                        v0 = true_v0,
                                        kappa = true_kappa,
                                        theta = true_theta,
                                        sigma = true_sigma,
                                        rho = true_rho,
                                        lambd = true_lambd,
                                        tau = tau_arr[i],
                                        r = r_arr[i]) for i in range(len(S0_arr))]


error_arr = np.array(P_arr) - np.array(list_theoretical_heston)
plt.figure(figsize=(6,4), dpi=700)
plt.plot(tau_arr,
         list_theoretical_heston, label = 'Heston Model', color = 'red', 
         linestyle = '--', marker = 'x')
plt.plot(tau_arr,
         P_arr, label = 'Market price', color = 'blue', linestyle = '-.',
         marker = 'x')
plt.plot(tau_arr,
         error_arr, label = 'Error Spread', color = 'green', linestyle = ':', 
         alpha = 0.5, marker ='x')

plt.fill_between(tau_arr, error_arr, 0, color='green', alpha=0.15)

plt.axhline(0, 0.05, 0.95, color = 'gray')
plt.ylim(top=1.2)
plt.ylim(bottom=-0.5)
plt.title("Heston calibration on EUR/USD calls ATM")
plt.xlabel("Maturity")
plt.ylabel("Prices")
plt.legend()
plt.grid(True)
plt.show()

