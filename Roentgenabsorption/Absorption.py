import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from scipy.signal import find_peaks 
from scipy.signal import peak_widths
import uncertainties.unumpy as unp 
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
from scipy.optimize import curve_fit
from lmfit import Model
from scipy.interpolate import UnivariateSpline


#Absorptionskoeffizienten


#1 Bragg-Bedingung überprüfen (LiF-Kristall, U=35kV, I=1mA, Integrationszeit pro Winkel t=5s)

alpha_Bragg, rate_Bragg = np.genfromtxt("Bragg.dat", unpack = True) #Alpha von 26 bis 30 Grad, 0,1°-Sprünge, Bragg-Rate in Imp/s
impulse_Bragg = rate_Bragg * 5 #Anzzahl der Impulse in 5 Sekunden
lambda_Bragg = (2*201.4*10**(-12)*np.sin(alpha_Bragg*2*np.pi/360))

alpha_Bragg_u = unp.uarray(alpha_Bragg, 0.1)
lambda_Bragg_u = (2*201.4*10**(-12)*unp.sin(alpha_Bragg_u*2*np.pi/360))


#Maxium Wellenlänge
print(f"""
Max bei:{alpha_Bragg[22]} in °
""")
plt.plot(alpha_Bragg, rate_Bragg, "x", label=r"LiF, $\theta = 14°$")
plt.plot([alpha_Bragg[22], alpha_Bragg[22]], [0, rate_Bragg[22]], "r-", label=r'$Peak (28,2°)$')
plt.xlabel(r"Messwinkel")
plt.ylabel(r"Impulsrate [Impulse/Sekunde]")
plt.legend(loc = "best")
plt.savefig("Bragg.pdf")
plt.clf()


a = const.value("fine-structure constant")
R_inf = const.value("Rydberg constant times hc in J")
print(R_inf)
Z = np.array([29, 30, 31, 35, 37, 38, 40]) #(Cu, Zn, Ga, Br, Rb, Sr, Zr)
E_k = np.array([8.98, 9.65, 10.37, 13.47, 15.20, 16.10, 17.99]) *1000*const.e #Gleiche Reihenfolge in keV
sigma_k = Z - np.sqrt((E_k/R_inf) - (a**2 * Z**4)/4) #Gleiche Reihenfolge
lambda_k = np.array([1380, 1280, 1200, 920, 816, 770, 689]) *10**(-12)# in m
theta_Bragg = np.arcsin(const.h * const.c / E_k /(2*201.4*10**(-12)))*(180/np.pi)# in °
print(f"""
Cu, Zn, Ge, Br, Rb, Sr, Zr
Z:{Z}
E_k:{E_k/const.e}
sigma_k:{sigma_k}
theta_Bragg:{theta_Bragg}
""")

#2 Plot Bremsspektrum Kupfer
theta_Cu, rate_Cu = np.genfromtxt("Emissionsspektrum.dat", unpack = True) #Beschleunigungsspannung U=35kV, Strom I=1mA, LiF-Kristall
impulse_Cu = rate_Cu * 10 #Anzzahl der Impulse in 10 Sekunden

theta_Cu_u = unp.uarray(theta_Cu, 0.1)

plt.plot(theta_Cu, rate_Cu, "k")
plt.plot(theta_Cu, rate_Cu, "-", label=r'Bremsspektrum')

#2.1 Peaks des Cu-Spektrums
peaks=find_peaks(rate_Cu, height = 1200, distance = 2)
print("K-AlBeta und K-Alpha Peak:")
print(peaks)
print(theta_Cu[122], theta_Cu[145])

#2.2 Energien der Peaks
E_beta = const.h * const.c /(2*201.4*10**(-12)*unp.sin(theta_Cu_u[122]*2*np.pi/360))/const.e
E_alpha  = const.h * const.c /(2*201.4*10**(-12)*unp.sin(theta_Cu_u[145]*2*np.pi/360))/const.e
print("Energien der Peaks (Literatur): 8905eV, 8038eV")
print("Energien der Peaks (Experiment):")
print(E_beta, E_alpha)
print('Abweichung der Peaks:')
print(abs(1-E_beta/8905)*100, abs(1-E_alpha/8038)*100)

#2.3 Halbwertsbreite K_alpha und K_beta

peak_Breiten = peak_widths(rate_Cu, peaks[0], rel_height=0.5)
#print(f"""
#Breite Beta, Alpha:{peak_Breiten[0]}
#""")

#Untergrund
def grund(a, b, x):
    return a*x+b 

params, cov = curve_fit(grund,theta_Cu[40:105], rate_Cu[40:105])
print(params)

#2.4 Plot
plt.plot(theta_Cu[120:128], rate_Cu[120:128], "r-", label=r'$K_{\beta}$')
plt.plot(theta_Cu[142:151], rate_Cu[142:151], "-", label=r'$K_{\alpha}$')
plt.xlabel(r'Bragg-Winkel [°]')
plt.ylabel(r'Impulsrate [Impulse/Sekunde]')
plt.legend(loc = 'best')
plt.savefig('Spektrum_Cu.pdf')
plt.clf()


#2.5 Gauß-Fit der Peaks
def gaussian(x, amp, cen, wid):
    return (amp / (np.sqrt(2*np.pi) * wid)) * np.exp(-(x-cen)**2 / (2*wid**2))


#Beta und Breite
x = np.linspace(19.3, 23.8, 45)
x2 = np.linspace(19.3, 23.8, 45)
x_1= theta_Cu[115:138]
y_1 = rate_Cu[115:138]
init_vals = [1590, 20.3, 0.5]
best_vals, cov_matrix = curve_fit(gaussian, x_1, y_1, p0 = init_vals)
Beta_Peak = find_peaks(gaussian(x, best_vals[0], best_vals[1], best_vals[2]), height=1200)
#print("Gauß-Beta-Peak")
#print(Beta_Peak)
Beta_Breite = peak_widths(gaussian(x, best_vals[0], best_vals[1], best_vals[2]), Beta_Peak[0], rel_height=0.5)
#Alpha
x_2= theta_Cu[126:151]
y_2 = rate_Cu[126:151]
init_vals2 = [5050, 22.6, 0.7]
best_vals2, cov_matrix2 = curve_fit(gaussian, x_2, y_2, p0 = init_vals2)
Alpha_Peak = find_peaks(gaussian(x2, best_vals2[0], best_vals2[1], best_vals2[2]), height=5000)
Alpha_Breite = peak_widths(gaussian(x2, best_vals2[0], best_vals2[1], best_vals2[2]), Alpha_Peak[0], rel_height=0.5)
#print("Gauß-Alpha-Peak")
#print(Alpha_Peak)
#print(x[10], x[10]-0.1*Alpha_Breite[0]/2, x[10]+0.1*Alpha_Breite[0]/2)
print("Breiten")
print(Beta_Breite[0], Alpha_Breite[0])


#2 Plot Peaks Kupfer
plt.plot(theta_Cu[115:160], rate_Cu[115:160] - grund(params[1], params[0], theta_Cu[115:160]), "k--", label=r'Bremsspektrum')
plt.plot(theta_Cu[120:128], rate_Cu[120:128] - grund(params[1], params[0], theta_Cu[120:128]), "r--", label=r'$K_{\beta}$')
plt.plot(theta_Cu[142:151], rate_Cu[142:151] - grund(params[1], params[0], theta_Cu[142:151]), "--", label=r'$K_{\alpha}$')
x3 = np.linspace(19.3, 21, 1000)
x4 = np.linspace(21, 23.8, 1000)
#plt.plot(x3, gaussian(x3, best_vals[0], best_vals[1], best_vals[2]), "y-", label=r"Gauß-Fits")
#plt.plot(x4, gaussian(x4, best_vals2[0], best_vals2[1], best_vals2[2]), "y-")
#plt.plot([x[10]-0.1*Beta_Breite[0]/2,x[10]+0.1*Beta_Breite[0]/2], [881.5, 881.5], "g-")
#plt.plot([x[32]-0.1*Alpha_Breite[0]/2,x[32]+0.1*Alpha_Breite[0]/2], [2725.04, 2725.04], "g-")
spline1 = UnivariateSpline(theta_Cu[120:128], rate_Cu[120:128]-np.max(rate_Cu[120:128])/2, s=0)
r1, r2 = spline1.roots() # find the roots
spline2 = UnivariateSpline(theta_Cu[142:151], rate_Cu[142:151]-np.max(rate_Cu[142:151])/2, s=0)
r3, r4 = spline2.roots() # find the roots
plt.axvspan(r1, r2, facecolor='g', alpha=0.5)
plt.axvspan(r3, r4, facecolor='g', alpha=0.5)

print("Untergrund Fit")
print(params)

plt.xlabel(r'Bragg-Winkel [°]')
plt.ylabel(r'Impulsrate [Impulse/Sekunde]')
plt.legend(loc = 'best')
plt.savefig('Peaks_Cu.pdf')
plt.clf()

#Auflösungsvermögen berechnen
#Beta
Auflösung_Beta =  E_beta/((const.h * const.c /(2*201.4*10**(-12)*np.sin(r1*2*np.pi/360))/const.e)-(const.h * const.c /(2*201.4*10**(-12)*np.sin(r2*2*np.pi/360))/const.e))
Auflösung_Alpha =  E_alpha/((const.h * const.c /(2*201.4*10**(-12)*np.sin(r3*2*np.pi/360))/const.e)-(const.h * const.c /(2*201.4*10**(-12)*np.sin(r4*2*np.pi/360))/const.e))
print(f"""
Auflösung Beta:{Auflösung_Beta}
Auflösung Alpha:{Auflösung_Alpha}
""")
print((const.h * const.c /(2*201.4*10**(-12)*np.sin(r1*2*np.pi/360))/const.e)- (const.h * const.c /(2*201.4*10**(-12)*np.sin(r2*2*np.pi/360))/const.e))
print((const.h * const.c /(2*201.4*10**(-12)*np.sin(r3*2*np.pi/360))/const.e)-(const.h * const.c /(2*201.4*10**(-12)*np.sin(r4*2*np.pi/360))/const.e))
#Abschirmkonstanten Kupfer
sigma_k1 = Z[0] - np.sqrt(E_k[0]/R_inf)
sigma_k2 = Z[0] - unp.sqrt((Z[0]-sigma_k1)**(2)*4 - E_alpha*4/(R_inf/const.e))
sigma_k3 = Z[0] - unp.sqrt((Z[0]-sigma_k1)**(2)*9 - E_beta*9/(R_inf/const.e))
print(f"""
sigma_k1:{sigma_k1}
sigma_k2:{sigma_k2}
sigma_k3:{sigma_k3}
""")
Vergleich_k1 = Z[0] - np.sqrt(E_k[0]/R_inf)
Vergleich_k2 = Z[0] - unp.sqrt((Z[0]-sigma_k1)**(2)*4 - 8038*4/(R_inf/const.e))
Vergleich_k3 = Z[0] - unp.sqrt((Z[0]-sigma_k1)**(2)*9 - 8905*9/(R_inf/const.e))
print(f"""
Vergleich_k1:{Vergleich_k1}
Vergleich_k2:{Vergleich_k2} 
Vergleich_k3:{Vergleich_k3}
""")
#Absorptionsspektren

def I_k(min, max):
    return min + (max-min)/2

def s_k (Z, E):
    return Z - unp.sqrt(E*const.e/R_inf - a**2*Z**4/4) 

#Brom
theta_Br, rate_Br = np.genfromtxt("Brom.dat", unpack = True)# Beschleunigungsspannung U=35kV, Strom I=1mA, LiF-Kristall Integrationszeit pro Winkel t=20s
theta_Br_u = unp.uarray(theta_Br, 0.1)
E_Br = const.h * const.c/(2*201.4*10**(-12)*unp.sin(theta_Br_u*2*np.pi/360)) /const.e
plt.plot(theta_Br, rate_Br, "x", label=r"Absorptionsspektrum Br")
plt.xlabel(r"$\theta$ [°]")
plt.ylabel(r"Impulsrate [Imp/s]")
plt.legend(loc="best")
plt.savefig("Brom.pdf")
plt.clf()
I_kBr = I_k(9, 27)
print("I_kBr und zugehörige Energie und Sigma_k")
print(I_kBr)
Br = unp.uarray(13.2, 0.1)
E_kBr = const.h * const.c/(2*201.4*10**(-12)*unp.sin(Br*2*np.pi/360))/const.e
print(E_kBr)
print(s_k(35, E_kBr))






#Gallium
theta_Ga, rate_Ga = np.genfromtxt("Gallium.dat", unpack = True)# Beschleunigungsspannung U=35kV, Strom I=1mA, LiF-Kristall Integrationszeit pro Winkel t=20s
theta_Ga_u = unp.uarray(theta_Ga, 0.1)
E_Ga = const.h * const.c/(2*201.4*10**(-12)*unp.sin(theta_Ga_u*2*np.pi/360))/const.e
plt.plot(theta_Ga, rate_Ga, "x", label=r"Absorptionsspektrum Ga")
plt.xlabel(r"$\theta$ [°]")
plt.ylabel(r"Impulsrate [Imp/s]")
plt.legend(loc="best")
plt.savefig("Gallium.pdf")
plt.clf()
I_kGa = I_k(66, 122)
print("I_kGa und zugehörige Energie und Sigma_k")
print(I_kGa)
Ga = unp.uarray(17.3, 0.1)
E_kGa = const.h * const.c/(2*201.4*10**(-12)*unp.sin(Ga*2*np.pi/360))/const.e
print(E_kGa)
print(s_k(31, E_kGa))


#Zink
theta_Zn, rate_Zn = np.genfromtxt("Zink.dat", unpack = True)# Beschleunigungsspannung U=35kV, Strom I=1mA, LiF-Kristall Integrationszeit pro Winkel t=20s
theta_Zn_u = unp.uarray(theta_Zn, 0.1)
E_Zn = const.h * const.c/(2*201.4*10**(-12)*unp.sin(theta_Zn_u*2*np.pi/360))/const.e
plt.plot(theta_Zn, rate_Zn, "x", label=r"Absorptionsspektrum Zn")
plt.xlabel(r"$\theta$ [°]")
plt.ylabel(r"Impulsrate [Imp/s]")
plt.legend(loc="best")
plt.savefig("Zink.pdf")
plt.clf()
I_kZn = I_k(54, 102)
print("I_kZn und zugehörige Energie und Sigma_k")
print(I_kZn)
Zn = unp.uarray(18.7, 0.1)
E_kZn = const.h * const.c/(2*201.4*10**(-12)*unp.sin(Zn*2*np.pi/360))/const.e
print(E_kZn)
print(s_k(30, E_kZn))


#Strontium
theta_Sr, rate_Sr = np.genfromtxt("Strontium.dat", unpack = True)# Beschleunigungsspannung U=35kV, Strom I=1mA, LiF-Kristall Integrationszeit pro Winkel t=20s
theta_Sr_u = unp.uarray(theta_Sr, 0.1)
E_Sr = const.h * const.c/ (2*201.4*10**(-12)*unp.sin(theta_Sr_u*2*np.pi/360))/const.e
plt.plot(theta_Sr, rate_Sr, "x", label=r"Absorptionsspektrum Sr")
plt.xlabel(r"$\theta$ [°]")
plt.ylabel(r"Impulsrate [Imp/s]")
plt.legend(loc="best")
plt.savefig("Strontium.pdf")
plt.clf()
I_kSr = I_k(40, 196)
print("I_kSr und zugehörige Energie und Sigma_k")
print(I_kSr)
Sr = unp.uarray(11.1, 0.1)
E_kSr = const.h * const.c/(2*201.4*10**(-12)*unp.sin(Sr*2*np.pi/360))/const.e
print(E_kSr)
print(s_k(38, E_kSr))


#Rubidium
theta_Rb, rate_Rb = np.genfromtxt("Rubidium.dat", unpack = True)# Beschleunigungsspannung U=35kV, Strom I=1mA, LiF-Kristall Integrationszeit pro Winkel t=20s
theta_Rb_u = unp.uarray(theta_Rb, 0.1)
E_Rb = const.h * const.c/(2*201.4*10**(-12)*unp.sin(theta_Rb_u*2*np.pi/360))/const.e
plt.plot(theta_Rb, rate_Rb, "x", label=r"Absorptionsspektrum Rb")
plt.xlabel(r"$\theta$ [°]")
plt.ylabel(r"Impulsrate [Imp/s]")
plt.legend(loc="best")
plt.savefig("Rubidium.pdf")
plt.clf()
I_kRb = I_k(12, 64)
print("I_kRb und zugehörige Energie und Sigma_k")
print(I_kRb)
Rb = unp.uarray(11.8, 0.1)
E_kRb = const.h * const.c/(2*201.4*10**(-12)*unp.sin(Rb*2*np.pi/360))/const.e
print(E_kRb)
print(s_k(37, E_kRb))


#Zirkonium
theta_Zr, rate_Zr = np.genfromtxt("Zirkonium.dat", unpack = True)# Beschleunigungsspannung U=35kV, Strom I=1mA, LiF-Kristall Integrationszeit pro Winkel t=20s
theta_Zr_u = unp.uarray(theta_Zr, 0.1)
E_Zr = const.h * const.c/(2*201.4*10**(-12)*unp.sin(theta_Zr_u*2*np.pi/360))/const.e
plt.plot(theta_Zr, rate_Zr, "x", label=r"Absorptionsspektrum Zr")
plt.xlabel(r"$\theta$ [°]")
plt.ylabel(r"Impulsrate [Imp/s]")
plt.legend(loc="best")
plt.savefig("Zirkonium.pdf")
plt.clf()
I_kZr = I_k(112, 301)
print("I_kZr und zugehörige Energie und Sigma_k")
print(I_kZr)
Zr = unp.uarray(10, 0.1)
E_kZr = const.h * const.c/(2*201.4*10**(-12)*unp.sin(Zr*2*np.pi/360))/const.e
print(E_kZr)
print(s_k(40, E_kZr))

#s_Zn
#s_Ga
#s_Br
#s_Rb
#s_Sr

s = np.array([s_k(30, E_kZn), s_k(31, E_kGa), s_k(35, E_kBr), s_k(37, E_kRb), s_k(38, E_kSr), s_k(40, E_kZr)])

#Moseley Wavecheck
E_ks = np.array([E_kZn, E_kGa, E_kBr, E_kRb, E_kSr, E_kZr])
nomsE = noms(E_ks)
param, covm = curve_fit(grund, Z[1:]-noms(s), np.sqrt(nomsE*const.e))
x = np.linspace(26, 40.1)
plt.plot(Z[1:] - noms(s), np.sqrt(noms(E_ks)*const.e), "x", label=r"Messwerte")
plt.plot(x, param[1]*x+param[0], "g-", label=r"Lineare Regression")
plt.xlabel(r"$z_{eff}$")
plt.ylabel(r"$\sqrt{E_K}$ in $\sqrt{J}$")
plt.legend(loc = "best")
plt.savefig("Moseley.pdf")
print("Erst Parameter dann zugehörige Fehler")
print(param)
errors = np.sqrt(np.diag(covm))
m = unp.uarray(param[1], errors[1])
print(errors)
print("Rydberg-Energie und Konstante")
print(m**2/const.e)
print((m**2)/(const.h*const.c))









