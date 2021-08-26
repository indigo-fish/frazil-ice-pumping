import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numpy.polynomial import Polynomial

df = pd.read_csv('Default Dataset.csv', header=None)
x = df[0] * 1e3
y = df[1]
plt.scatter(x, y, s=2)

def depth_function(x, c0, c1, c2, c3, c4, c5, c6):
    return c0 + c1 * x + c2 * x**2 + c3 * x**3 + c4 * x**4 + c5 * x**5 + c6 * x**6

results, error = curve_fit(depth_function, x, y)
y1 = []
for x0 in x:
    #y1.append(results[0] + results[1] * x0 + results[2] * x0 ** 2 + results[3] * x0 ** 3 + results[4] * x0 ** 4 + results[5] * x0 ** 5 + results[6] * x0 ** 6)
    y1.append(depth_function(x0, results[0], results[1], results[2], results[3], results[4], results[5], results[6]))
plt.scatter(x, y1, s=2)

# c = Polynomial.fit(x, y, 2)
# y2 = []
# for x0 in x:
#     y2.append(c(0) + c(1) * x0 + c(2) * x0**2)
# plt.scatter(x, y2, s=2)

plt.show()