from word2number import w2n
from num2words import num2words
import sympy as sy
from sympy import *
from mpmath import *
from statistics import *
import pandas as pd
import numpy as np
from scipy import stats
from scipy import optimize
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sympy.solvers import solve
from sympy import Symbol
import math
x = Symbol('x')
print(solve("x**2 - 1", x))
#from mpmath import *
#from math import *
#import math
#from sympy.abc import x

#x = symbols('x')
#print(integrate(x**2 * mp.e * cos(x), x))
#print(diff(exp(x**2),x,1))
#print(eval("cos(0)"))
#print(eval("exp(2)"))
def exponenial_func(x, a, b, c):
    return a*np.exp(-b*x)+c
def differ(exp, a):
	return diff(lambda x: exp, a)
def onevarstats(arr):
	return mean(arr), median(arr), mode(arr), stdev(arr), variance(arr)
def linreg(x, y):
	x = np.array(x)
	slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
	plt.plot(x, y, 'o', label='original data')
	plt.plot(x, intercept + slope*x, 'r', label='fitted line')
	plt.legend()
	plt.show()
	return stats.linregress(x,y)#for exp(2), sympy returns exp(2) while math returns decimal value
def f(x):
	return x**2 + 10*np.sin(x)
def graph():
	formula = str(formula)
	x = np.array(range(-10, 10))
	fig = plt.figure(figsize = (6, 4))
	ax = fig.add_subplot(111)
	ax.plot(x, np.sin(x))
	#y = eval(formula)
	#plt.plot(x, y)
	#plt.plot(x, formula)
	plt.show()
def graph2(formula):
	formula = str(formula)
	x = np.array(range(-10, 10))
	y = eval(formula)
	print(y)
	plt.plot(x, y)
	plt.show()
#print(diff(lambda x: x**2 + x, 1))
#print(differ(x**2 + x, 1.0))
#print(onevarstats([1, 2, 3, 3]))
lst = ['Geeks', 'For', 'Geeks', 'is', 
            'portal', 'for', 'Geeks']
 
ax = np.array([399.75, 989.25, 1578.75, 2168.25])
ay = np.array([109,62,39,13])
#ax = np.array([399.75, 989.25, 1578.75, 2168.25, 2757.75, 3347.25, 3936.75, 4526.25, 5115.75, 5705.25])
#ay = np.array([109,62,39,13,10,4,2,0,1,2])
#OptimizeWarning: Covariance of the parameters could not be estimated, category=OptimizeWarning)


popt, pcov = curve_fit(exponenial_func, ax, ay, p0=(1, 1e-6, 1))
yy = exponenial_func(ax, *popt)
root = optimize.root(f, 1)
print(root.x)
#print(graph2("np.sqrt(x)"))
#print(graph())
#plt.plot(ax, ay, 'o', label='original data')
#plt.plot(ax, yy, 'r', label='fitted line')
#plt.legend()
#plt.show()
# Calling DataFrame constructor on list
#df = pd.DataFrame(lst)
#print(df)
#print(str(num2words("0.02 ")))
#print(w2n.word_to_num("point three thousand four"))
time_diff = [1, 2, 3]
#print(np.percentile(time_diff, 50)) 
#x = np.array([[0, 1], [0, 2]])
#r = stats.linregress(x)
#print(linreg([0,1], [0,2]))
