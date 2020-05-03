import numpy as np
import scipy as scipy
import matplotlib.pyplot as plt
import scipy.constants as const
from scipy.signal import find_peaks 

theta_Al, rate_Al = np.genfromtxt('ComptonAl.txt', unpack = True) # U = 35kV, Theta(°) und Rate in (Imp/s)
theta_norm, rate_norm = np.genfromtxt('ComptonOhne.txt', unpack = True) #U=35kV, Theta(°) und Rate in (Imp/s)
theta_Cu, rate_Cu = np.genfromtxt('EmissionCu.dat', unpack = True) #Intzeit/Winkel 10s, UBeschl = 35kV, I=1mA, LiF-Kristall, Theta(°) und Rate in (Imp/s)

def lam(theta):
    return (2*201.4*10**(-12)*np.sin(theta*2*np.pi/360))

def Transmission(I, I_0):
    return I/I_0

#Compton Wellenlänge berechnen
print("Compton-Wellenlänge:")
lambda_c = const.h/(const.c * const.m_e)  #Compton-Wellenlänge
print(lambda_c)


#Bragg-Winkel berechnen Quelle Energien http://www.phywe-ru.com/index.php/fuseaction/download/lrn_file/versuchsanleitungen/P2540101/d/p2540101d.pdf
bragg_kalpha = np.arcsin((const.h * const.c/(8.038*10**(3)*const.e))/(2*201.4*10**(-12)))/(2*np.pi)*360
bragg_kbeta = np.arcsin((const.h * const.c/(8.905*10**(3)*const.e))/(2*201.4*10**(-12)))/(2*np.pi)*360
lambda_kalpha = (const.h * const.c)/(8038*const.e)
lambda_kbeta = (const.h * const.c)/(8905*const.e)
print("K-Alpha:")
print(bragg_kalpha)
print(lambda_kalpha)
print("K-Beta:")
print(bragg_kbeta)
print(lambda_kbeta)


#Plot Bremsspektrum Kupfer
plt.plot(theta_Cu, rate_Cu, "-", label=r'Cu bei 35kV')
plt.xlabel(r'Bragg-Winkel [°]')
plt.ylabel(r'Impulsrate [Impulse/Sekunde]')
plt.legend(loc = 'best')
plt.savefig('Spektrum_Cu.pdf')
plt.clf()


#K-Alpha und K-Beta Peak finden
peaks=find_peaks(rate_Cu, height = 1200, distance = 2)
print("K-AlBeta und K-Alpha Peak:")
print(peaks)
print(theta_Cu[122], theta_Cu[145])
#Energien der Peaks
E_beta = const.h * const.c /(2*201.4*10**(-12)*np.sin(theta_Cu[122]*2*np.pi/360))/const.e
E_alpha  = const.h * const.c /(2*201.4*10**(-12)*np.sin(theta_Cu[145]*2*np.pi/360))/const.e
print("Energien der Peaks (Literatur): 8905eV, 8038eV")
print("Energien der Peaks (Experiment):")
print(E_beta, E_alpha)
print('Abweichung der Peaks:')
print(abs(1-E_beta/8905)*100, abs(1-E_alpha/8038)*100)


#Totzeitkorrekturen und Transmission
def korrektur(N):
    return N/(1-90*10**(-6)*N)

I_Al = korrektur(rate_Al)
I_0  = korrektur(rate_norm)

T = I_Al/I_0

T_1 = 1180/2731
T_2 = 1024/2731
print("T1 und T2:")
print(T_1, T_2)


#Plot ComptonAlu

lambda_Alu = lam(theta_Al)

print("Wellenlänge Alu:")
print(lambda_Alu)
print("Intensität Alu:")
print(I_Al)
print("Intensität norm:")
print(I_0)
print("Transmission aufgabe 2:")
print(T)

#Ausgleichsgerade
params, cov_matrix = np.polyfit(lambda_Alu, T, deg=1, cov = True)
x=np.linspace(4.8*10**(-11), 7*10**(-11),1000000)
T_fit = x * params[0] + params[1]
print("Regression: Steigung, Y-Achse")
print(params[0], params[1])

plt.plot(lambda_Alu,T, "x", label=r'Al bei 35kV')
plt.plot(x, x*params[0]+params[1], "r-", label=r'Ausgleichsgerade')
plt.xlabel(r'Wellenlänge [m]')
plt.ylabel(r'Transmission')
plt.legend(loc = 'best')
plt.savefig('Compton_Alu.pdf')
plt.clf()

#lambda1 und lambda 2 bestimmen

lambda1 = (T_1 - params[1])/params[0]
lambda2 = (T_2 - params[1])/params[0]
print("Lambda 1, Lambda 2")
print(lambda1, lambda2)
print('Compton-Wellenlänge (experimentell):')
print(lambda2-lambda1)
print('Abweichung Compton-Wellenlänge:')
print(100*(1-lambda_c/(lambda2-lambda1)))












































