import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from matplotlib import style
style.use("ggplot")


"""
  #####   ##  ##   #####    ######   #####    ##  ##    ####     #####   ######   #####
 ##       ##  ##   ##  ##   ##       ##  ##   ##  ##     ##     ##       ##       ##  ##
  ####    ##  ##   #####    #####    #####    ##  ##     ##      ####    #####    ##  ##
     ##   ##  ##   ##       ##       ##  ##   ##  ##     ##         ##   ##       ##  ##
 #####     ###     ##       ######   ##  ##     ##      ####    #####    ######   #####
"""

house_price = [245, 312, 279, 308, 199, 219, 405, 324, 319, 255]
size = [1400, 1600, 1700, 1875, 1100, 1550, 2350, 2450, 1425, 1700]

# Reshape the input to your regression
size2 = np.array(size).reshape((-1, 1))

# By using fit module in linear regression, user can fir the data frequently and quickly
regr = linear_model.LinearRegression()
regr.fit(size2, house_price)
print("Coefficients: \n", regr.coef_)
print("Intercept: \n", regr.intercept_)

"""
Output:
Coefficient: [ 0.10976774]
Intercepts: 98.2483296214
"""


# Formula obtained for the trained model
def graph(formula, x_range):
    x = np.array(x_range)
    y = eval(formula)
    plt.plot(x, y)

# Plotting the prediction line
# size_new = 1400
# price = (size_new * regr.coef_) + regr.intercept_
# or;
# regr.predict([[size_new]])


graph('regr.coef_ * x + regr.intercept_', range(1000, 2700))
plt.scatter(size, house_price, color='black')
plt.ylabel('House price', color='C0')
plt.xlabel('Size of house', color='C0')
plt.xticks(color='C0', rotation='vertical')
plt.yticks(color='C0')
plt.show()
