#pip3 install --user num2words
from word2number import w2n
from num2words import num2words
from mpmath import *
from sympy import *
from sympy.solvers import solve
from scipy import stats
from scipy.optimize import curve_fit
from scipy.optimize import root
from statistics import *
import pandas as pd
import numpy as np
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt
import math
import random
from fractions import Fraction
x = Symbol("x")
y = Symbol("y")
alpha = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
def combo(n, r):
	return math.factorial(n)//(math.factorial(r)*math.factorial(n-r))
def perm(n, r):
	return math.factorial(n)//math.factorial(r)
def integral(f, x):
	z = integrate(f, x)
	z = str(z)
	string = ""
	i = 0
	while i < len(z):
		if(i == 4 and z[i] == "2" and z[i+1] == "." and z[i+2] == "7" and 
		z[i+3] == "1" and z[i+4] == "8"): #len 16
			string += "e"
			i += 16
		else:
			string += z[i]
			i+=1
	return string
def defint(f, x, a, b):
	return integrate(f, (x, a, b))
def deriv(exp, x):
	return diff(exp, x)
def differ(exp, x):
	return diff(lambda x: exp, x)
def varstats(arr): #numpy percentile more precise
	df = pd.DataFrame(arr)
	print(df)
	most = 0
	try:
		mode(arr)
		most = mode(arr)
	except:
		most = "None"
	return ("Mean:"+str(mean(arr)),"Sum:"+str(sum(arr)),"SumSquares:"+str(sum(map(lambda i:i*i,arr))),
	"Sx:"+str(stdev(arr)),"Popstd:"+str(pstdev(arr)),"SampVar:"+str(variance(arr)),"Popvar:"+str(pstdev(arr)),
	"Size:"+str(len(arr)),"Min:"+str(min(arr)),"Q1:"+str(np.percentile(arr, 25)),
	"Median:"+str(median(arr)),"Q3:"+str(np.percentile(arr, 75)),"Max:"+str(max(arr)),"Mode:"+str(most))
def matrix(mat):
	return Matrix(mat)
def det(mat):
	return mat.det()
def rref(mat):
	return mat.rref()
def linreg(x, y):
	x = np.array(x)
	slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
	plt.plot(x, y, 'o', label='original data')
	plt.plot(x, intercept + slope*x, 'r', label='fitted line')
	plt.legend()
	plt.show()
	return stats.linregress(x,y)
def polyreg(x, y, n):
	x_new = np.linspace(x[0], x[-1], num=len(x)*10)
	coefs = poly.polyfit(x, y, n)
	ffit = poly.polyval(x_new, coefs)
	plt.plot(x, y, 'o', label='original data')
	plt.plot(x_new, ffit, 'r', label='fitted curve')
	plt.legend()
	plt.show()
	arr = []
	for thing in np.polyfit(x, y, n):
		arr.append(thing)
	return arr
def exponential_func(x, a, b, c):
    return a*np.exp(b*x)+c
def expreg(x, y):
	ax= np.array(x)
	ay = np.array(y)
	popt, pcov = curve_fit(exponential_func, ax, ay, p0=(1, 1e-6, 1))#, maxfev = 10000)
	yy = exponential_func(ax, *popt)
	plt.plot(ax, ay, 'o', label='original data')
	plt.plot(ax, yy, 'r', label='fitted line')
	plt.legend()
	plt.show()
	arr = []
	for thing in popt:
		arr.append(thing)
	return arr
def graph(formula):
	print("he")
	formula = str(formula)
	x = np.array(range(-10, 10))
	print(x)
	y = eval(formula)
	print(y)
	plt.plot(x, y)
	plt.show()
def evaluate(formula, x):
	return eval(str(formula))
def intersection(one, two):
	xcord = solve(one - two, x)[0]
	ycord = evaluate(one, xcord)
	return [xcord, ycord]
def zeros(funct, guess):
	return root(funct, guess).x
	#return intersection(funct, 0)
def rand(low, high, times):
	return[random.uniform(low, high) for x in range(times)]
def randint(low, high, times):
	return[random.randrange(low, high) for x in range(times)]
def fract(num): #kinda weird
	return Fraction(num)
def numtoword(ans):
	ans = str(ans)
	final = ""
	st = ""
	temp = ""
	realtemp = ""
	boo = True
	arr = []
	for world in ans:
		arr.append(world)
	for i in range(len(arr)):
		if arr[i] == "(":
			final = " left parentheses "
		elif arr[i] == ")":
			final = "right parentheses "
		elif arr[i] == "[":
			final = "left bracket "
		elif arr[i] == "]":
			final = "right bracket "
		elif arr[i] == ":" or arr[i] == "=":
			final = " is "
		elif arr[i] == ",":
			final = "comma "
		elif arr[i] == "x":
			if arr[i-1] in alpha:
				final = arr[i]
			else:
				final = "x "
		elif arr[i] == "e":
			if arr[i-1] in alpha:
				final = arr[i]
			else:
				final = "e "
		elif arr[i] == "+":
			final = "plus "
		elif arr[i] == "*":
			if arr[i+1] == "*":
				final = "to the power of "
			elif arr[i-1] == "*":
				final = ""
			else:
				final = "times "
		elif arr[i] == "-":
			final = "minus "
		elif arr[i] == "/":
			final = "divided by "
		elif arr[i] == " ": #or arr[i] == "'":
			final = ""
		elif arr[i] == "'":
			final = " "
		elif arr[i] in alpha:
			final = arr[i]
		temp += arr[i]
		try:
			num2words(arr[i] + "0")
			realtemp = num2words(temp) + " "
			boo = False
		except:
			boo = True	
			temp = ""
			final = realtemp + final
			realtemp = ""
		if(i == len(arr) - 1) and boo == False:
			final = realtemp
			boo = True
		if boo == True:
			st += final
	return st
exp = ""
var = 1
while var == 1:
 exp = input("Math Expression: ")
 st = ""
 temp = ""
 realtemp = ""
 word = ""
 boo = True
 arr = []
 for world in exp.split():
	 arr.append(world)
 for i in range(len(arr)):
   if arr[i] == "plus":
     word = "+"
   elif arr[i] == "minus" or arr[i] == "negative":
     word = "-"
   elif arr[i] == "times":
     word = "*"
   elif arr[i] == "divided":
     word = "/"
   elif arr[i] == "point":
     word = "."
   elif arr[i] == "power":
     word = "**"
   elif arr[i] == "square": #math
     word = "math.sqrt"
   elif arr[i] == "absolute": #math
     word = "math.fabs"
   elif arr[i] == "factorial": #factorial left parentheses... #math
     word = "math.factorial"
   elif arr[i] == "perm":
	   word = "perm"
   elif arr[i] == "combo":
	   word = "combo"
   elif arr[i] == "integrate":
	   word = "integral"
   elif arr[i] == "definite":
	   word = "defint"
   elif arr[i] == "derivative":
	   word = "deriv"
   elif arr[i] == "differentiate": #not working
	   word = "differ"
   elif arr[i] == "log": #math
     word = "math.log"
   elif arr[i] == "cosine": #math
     word = "cos"
   elif arr[i] == "sine": #math
     word = "sin"
   elif arr[i] == "np.sine": #uhhhh
     word = "np.sin"
   elif arr[i] == "tangent": #math
     word = "tan"
   elif (arr[i] == "by" or arr[i] == "parentheses" or arr[i] == "root" or 
   arr[i] == "of" or arr[i] == "value" or arr[i] == "to" or arr[i] == "the" or arr[i] == "base"):
     word = ""
   elif arr[i] == "left":
     word = "("
   elif arr[i] == "right":
     word = ")"
   elif arr[i] == "left-bracket":
	   word = "["
   elif arr[i] == "right-bracket":
	   word = "]"
   elif arr[i] == "comma":
     word = ","
   elif arr[i] == "pi": #math
     word = "math.pi" 
   elif arr[i] == "e": #math
     word = "e"
   elif arr[i] == "x":
	   word = "x"
   elif arr[i] == "var-stats":
	   word = "varstats"
   elif arr[i] == "matrix":
	   word = "matrix"
   elif arr[i] == "determinant":
	   word = "det"
   elif arr[i] == "reduced-row-echelon":
	   word = "rref"
   elif arr[i] == "linreg":
	   word = "linreg"
   elif arr[i] == "polynomial":
	   word = "polyreg"
   elif arr[i] == "exponential":
	   word = "expreg"
   elif arr[i] == "random":
	   word = "rand"
   elif arr[i] == "randint":
	   word = "randint"
   elif arr[i] == "fraction":
	   word = "fract"
   elif arr[i] == "graph":
	   word = "graph"
   elif arr[i] == "evaluate":
	   word = "evaluate"
   elif arr[i] == "intersection":
	   word = "intersection"
   elif arr[i] == "zeros":
	   word = "zeros"
   temp += arr[i] + " "
   try:
	   w2n.word_to_num(arr[i])
	   realtemp = str(w2n.word_to_num(temp))
	   boo = False
   except:
	   boo = True	
	   temp = ""
	   word = realtemp + word
	   realtemp = ""
   if(i == len(arr) - 1) and boo == False:
	   word = realtemp
	   boo = True
   if boo == True:
	   st += word
 print(st)
 a = eval(st)
 print(a)
 print(numtoword(a))
#ax = np.array([399.75, 989.25, 1578.75, 2168.25, 2757.75, 3347.25, 3936.75, 4526.25, 5115.75, 5705.25])
#ay = np.array([109,62,39,13,10,4,2,0,1,2])

#exponential left left-bracket three-hundred-ninety-nine point seventy-five comma 
#nine-hundred-eighty-nine point twenty-five comma one-thousand-five-hundred-seventy-eight point 
#seventy-five comma two-thousand-sixty-eight point twenty-five right-bracket comma left-bracket 
#one-hundred-nine comma sixty-two comma thirty-nine comma thirteen right-bracket right
#expreg([399.75,989.25,1578.75,2068.25],[109,62,39,13])
#[  1.94227476e+02   5.40780932e-04  -4.83497094e+01] a*np.exp(b*x)+c format, different eqn from calc
