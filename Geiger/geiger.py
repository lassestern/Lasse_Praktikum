import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from scipy.signal import find_peaks 
from scipy.signal import peak_widths
import uncertainties.unumpy as unp 
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
from scipy.optimize import curve_fit

U_kenn, N = np.genfromtxt('Kennlinie.dat', unpack = True) #Spannung in V, Impulse, Integrationszeit 60s, Poissonverteilung
N_kenn = unp.uarray(N, np.sqrt(N)) #Impulse mit zugehörigen Fehlern
U_zaehl, I = np.genfromtxt('Zaehlrohrstrom.dat', unpack=True) #Spanung in V , Strom in 10^(-6) Ampere
I_zaehl = unp.uarray(I*10**(-6), 0.05*10**(-6)) #Strom in Ampere mit zugehörigem Fehler

#Geradengleichung definieren
def linear(x, a, b):
    return a*x+b 

#Plataeubereich von 380 bis 590 Volt

#Fit für Gerade durch Plataeubereich
params, cov = curve_fit(linear, U_kenn[6:27], N[6:27])
x = np.linspace(380, 590)
errors = np.sqrt(np.diag(cov))
params_err = unp.uarray(params, np.sqrt(np.diag(cov)))
print(params_err)
m = (1-(params_err[0]*400+params_err[1])/(params_err[0]*500+params_err[1]))*100
print(f""" 
Parameter Gerade:{params}
Fehler Geradenparameter: {errors}
Steigung Gerade in % pro 100V:{m}
""")


#Plot für die Kennlinie
plt.errorbar(U_kenn, N, xerr=0, yerr=np.sqrt(N), fmt='x', label=r"Charakteristik" )
plt.plot(x, params[0]*x+params[1], "g-", label=r"Gerade durch Plataeubereich")
plt.xlabel(r"Spannung [V]")
plt.ylabel(r"Impulse")
plt.legend(loc = "best")
plt.savefig("Charakteristik.pdf")
plt.clf()


#Totzeit bestimmen
N1 = unp.uarray(96041, np.sqrt(96041))  #Impulse bei einer Probe Messzeit t=120s
N12 = unp.uarray(158479, np.sqrt(158479)) #Impulse bei einer Probe Messzeit t=120s
N2 = unp.uarray(76518, np.sqrt(76518)) #Impulse bei zwei Proben Messzeit t=120s
T = (N1+N2-N12)/(2*N1*N2)*120
print(f"Totzeit:{T}")



#Freigesetzte Ladungen pro eingefallenem Teilchen
N_i = ([9837, 9995, 10264, 10151, 10184, 10253, 10493, 11547])
N_I = unp.uarray(N_i, np.sqrt(N_i)) 
Z = I_zaehl/(const.e*N_I/60)
print(f"Zaehlstrom:{I_zaehl}")
print(f"Impulse:{noms(N_I)/60, stds(N_I)/60}")
print("Freigesetzte Ladungen pro eingefallenem Teilchen in e")
print(Z[0])
print(Z[1])
print(Z[2])
print(Z[3])
print(Z[4])
print(Z[5])
print(Z[6])
print(Z[7])


#zugehöriger Plot
plt.errorbar(noms(I_zaehl)*10**(6), noms(Z), xerr=0, yerr=stds(Z), fmt='x', label=r"Freigesetzte Ladungen" )
plt.xlabel(r"Zählrohrstrom$\cdot 10^{-6}$ [A]")
plt.ylabel(r"Freigesetzte Ladung [e]")
plt.legend(loc = "best")
plt.savefig("Ladungen.pdf")
plt.clf()
