import numpy as np
import matplotlib.pyplot as plt

def DUmwandlung(Array):
    i=0
    while i<= 8:
        Array[i] = i * 6
        i += 1
    return Array

spannung, linie = np.genfromtxt("Daten.csv",delimiter=",",unpack=True)

print(linie)
print(spannung)

U =([-19.5, -16.1, -12.4, -9.6, -6.2, -2.4, 1.2, 5.1, 8.3])
D = ([0, 0, 0, 0, 0, 0, 0, 0, 0])

print(linie)

D = DUmwandlung(linie)

print(linie)

#LinRegress

params, cov_matrix = np.polyfit(spannung, linie, deg = 1, cov = True)

print(cov_matrix)

Ufit = np.linspace(-20, 9)

Dfit = Ufit * params[0] + params[1]

#Jetzt der Plot

plt.plot(spannung, linie, "x", label = r"Messwerte")
plt.plot(Ufit, Dfit, "-", label = r"Lineare Regression")
plt.xlabel(r"U [V]")
plt.ylabel(r"D [mm]")
plt.grid()
plt.legend(loc = "best")
plt.savefig("UD_plot.pdf")


#Fehler
errors = np.sqrt(np.diag(cov_matrix))
print("Steigung, y_Achsenabschnitt")
print(params[0], params[1])
print("Fehler")
print(errors)