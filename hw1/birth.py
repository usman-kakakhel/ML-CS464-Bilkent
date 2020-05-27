import random
import numpy as np
import matplotlib.pyplot as plt
import math

n = np.array((2, 3, 5, 10, 20, 30, 40, 50, 75, 100, 365))
fac365 = math.log(math.factorial(365))

facN = []
for i in n:
    facN.append(math.log(math.factorial(365 - i)))

nPow365 = []
for i in n:
    nPow365.append(math.log(365) * i)

y = []

for i in range (len(n)):
    y.append(1 - math.exp(fac365 - (facN[i] + nPow365[i])))

plt.plot(n, y, label='', lw=2)
plt.xlabel('The number of students in the auditorium')
plt.ylabel('The probability corresponding to n')
plt.title('Probability that any 2 people have the same birthday in the auditorium')
plt.legend()
plt.show()
