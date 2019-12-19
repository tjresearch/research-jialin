from tkinter import *
import math
from math import *  
from sympy import *
import numpy as np
import numpy.polynomial.polynomial as poly
from scipy import stats
from scipy.optimize import root
from scipy.optimize import curve_fit
from statistics import *
import random
from fractions import Fraction
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from word2number import w2n
from num2words import num2words
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk
expression = ""
carry = "" 
big = {}
alpha = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'  
x = Symbol("x")  
y = Symbol("y")
nocalc = 1.0
gui = Tk()
expression_field = Text(gui, height = 20, width = 30)
def press(num): #update expression in text box
    global expression 
    expression = expression + str(num) 
    expression_field.insert(END, num)
def equalpress(): #evaluate final expression
    global nocalc
    global expression
    global carry 
    try:   
        total = str(eval(expression))  
        carry = total
        expression_field.insert(END, "\n")
        expression_field.insert(END, total + "\n")
    except:  
        expression_field.insert(END, "error")
def numtoword():
	global carry
	global nocalc
	ans = str(carry)
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
	expression_field.insert(END, st + "\n")
	expression = ""
	carry = ""
	nocalc += 4 
def clearall(): 
    global expression
    global nocalc 
    expression = ""
    expression_field.delete('1.0',END)
    nocalc = 1.0
def clearline(): 
    global nocalc 
    expression_field.delete(str(nocalc),END)
def bs():
	global expression
	expression_field.delete('%s - 2c' % 'end')
	expression = expression[:-1]  
def graph(formula): #pack???
	formula = str(formula)
	x = np.array(range(-10, 10))
	y = eval(formula)
	plt.plot(x,y)
	plt.show()
def matrix(mat):
	return Matrix(mat)
def det(mat):
	return mat.det()
def rref(mat):
	return mat.rref()
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
def integral(f, x):
	z = integrate(f, x)
	z = str(z)
	string = ""
	i = 0
	while i < len(z)-4:
		if(z[i] == "2" and z[i+1] == "." and z[i+2] == "7" and 
		z[i+3] == "1" and z[i+4] == "8"): #len 16
			string += "e"
			i += 16
		else:
			string += z[i]
			i+=1
	while i < len(z) and i >= len(z)-4:
		string += z[i]
		i+=1 
	return string
def defint(f, x, a, b):
	return integrate(f, (x, a, b))
def deriv(exp, x):
	return diff(exp, x)
def perm(n, r):
	return math.factorial(n)//math.factorial(r)
def combo(n, r):
	return math.factorial(n)//(math.factorial(r)*math.factorial(n-r))
def randint(low, high, times):
	return[random.randrange(low, high) for x in range(times)]
def rand(low, high, times):
	return[random.uniform(low, high) for x in range(times)]
def fract(num): #kinda weird
	return Fraction(num)
def evaluate(formula, x):
	return eval(str(formula))
def help():
	f = open("test.txt", "r")
	expression_field.insert(END, f.read())
	f.close()
def evaluate(formula, x):
	return eval(str(formula))
def intersection(one, two):
	xcord = solve(one - two, x)[0]
	ycord = evaluate(one, xcord)
	return [xcord, ycord]
def zeros(funct): #kinda works
	return intersection(funct, 0)
	#return root(funct, guess).x
def translate():
	global expression
	global nocalc
	expression = expression_field.get(str(nocalc), END)
	arr=[]
	word = ""
	st= ""
	boo = True
	temp = ""
	realtemp = ""
	for world in expression.split():
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
	expression = st
	expression_field.insert(END, "\n")
	expression_field.insert(END, expression)
if __name__ == "__main__":     
    gui.title("Graphing Calculator")   
    gui.geometry("388x800") 
    equation = StringVar()    
    expression_field.focus()
    expression_field.grid(columnspan=4, ipadx=70) #placing widgets in position
    big['1'] = Button(gui, text=' 1 ', fg='black', bg='red', activebackground = 'blue',
                     command=lambda: press(1), height=1, width=7) 
    big['1'].grid(row=2, column=0)   
    big['2'] = Button(gui, text=' 2 ', fg='black', bg='red', activebackground = 'blue',
                     command=lambda: press(2), height=1, width=7) 
    big['2'].grid(row=2, column=1)   
    big['3'] = Button(gui, text=' 3 ', fg='black', bg='red', activebackground = 'blue',
                     command=lambda: press(3), height=1, width=7) 
    big['3'].grid(row=2, column=2) 
    big['4'] = Button(gui, text=' 4 ', fg='black', bg='red', activebackground = 'blue',
                     command=lambda: press(4), height=1, width=7) 
    big['4'].grid(row=3, column=0) 
    big['5'] = Button(gui, text=' 5 ', fg='black', bg='red', activebackground = 'blue',
                     command=lambda: press(5), height=1, width=7) 
    big['5'].grid(row=3, column=1) 
    big['6'] = Button(gui, text=' 6 ', fg='black', bg='red', activebackground = 'blue',
                     command=lambda: press(6), height=1, width=7) 
    big['6'].grid(row=3, column=2) 
    big['7'] = Button(gui, text=' 7 ', fg='black', bg='red', activebackground = 'blue',
                     command=lambda: press(7), height=1, width=7) 
    big['7'].grid(row=4, column=0) 
    big['8'] = Button(gui, text=' 8 ', fg='black', bg='red', activebackground = 'blue',
                     command=lambda: press(8), height=1, width=7) 
    big['8'].grid(row=4, column=1) 
    big['9'] = Button(gui, text=' 9 ', fg='black', bg='red', activebackground = 'blue',
                     command=lambda: press(9), height=1, width=7) 
    big['9'].grid(row=4, column=2) 
    big['0'] = Button(gui, text=' 0 ', fg='black', bg='red', activebackground = 'blue',
                     command=lambda: press(0), height=1, width=7) 
    big['0'].grid(row=5, column=0) 
    big['plus'] = Button(gui, text=' + ', fg='black', bg='red', activebackground = 'blue',
                  command=lambda: press("+"), height=1, width=7) 
    big['plus'].grid(row=2, column=3) 
    big['minus'] = Button(gui, text=' - ', fg='black', bg='red', activebackground = 'blue',
                   command=lambda: press("-"), height=1, width=7) 
    big['minus'].grid(row=3, column=3) 
    big['multiply'] = Button(gui, text=' * ', fg='black', bg='red', activebackground = 'blue',
                      command=lambda: press("*"), height=1, width=7) 
    big['multiply'].grid(row=4, column=3) 
    big['divide'] = Button(gui, text=' / ', fg='black', bg='red', activebackground = 'blue',
                    command=lambda: press("/"), height=1, width=7) 
    big['divide'].grid(row=5, column=3) 
    big['equal'] = Button(gui, text=' = ', fg='black', bg='light blue', activebackground = 'blue',
                   command=equalpress, height=1, width=7) 
    big['equal'].grid(row=15, column=2) 
    big['equalwords'] = Button(gui, text=' =(text) ', fg='black', bg='light blue', activebackground = 'blue',
                   command=numtoword, height=1, width=7) 
    big['equalwords'].grid(row=15, column=3) 
    big['translate'] = Button(gui, text=' translate ', fg='black', bg='light blue', activebackground = 'blue',
                   command=translate, height=1, width=7) 
    big['translate'].grid(row=15, column=1) 
    big['clearall'] = Button(gui, text='Clear All', fg='black', bg='red', activebackground = 'blue',
                   command=clearall, height=1, width=7) 
    big['clearall'].grid(row=5, column='1') 
    big['.'] = Button(gui, text='.', fg='black', bg='red', activebackground = 'blue',
                   command=lambda: press("."), height=1, width=7) 
    big['.'].grid(row=6, column=0) 
    big['<-'] = Button(gui, text='<-', fg='black', bg='red', activebackground = 'blue',
                   command=bs, height=1, width=7) 
    big['<-'].grid(row=6, column=1)
    big['('] = Button(gui, text='(', fg='black', bg='red', activebackground = 'blue',
                   command=lambda: press("("), height=1, width=7) 
    big['('].grid(row=6, column=2)  
    big[')'] = Button(gui, text=')', fg='black', bg='red', activebackground = 'blue',
                   command=lambda: press(")"), height=1, width=7) 
    big[')'].grid(row=6, column=3) 
    big['**'] = Button(gui, text='a^b', fg='black', bg='red', activebackground = 'blue',
                   command=lambda: press("**"), height=1, width=7) 
    big['**'].grid(row=7, column=0) 
    big['abs'] = Button(gui, text='|x|', fg='black', bg='red', activebackground = 'blue',
                   command=lambda: press("abs("), height=1, width=7) 
    big['abs'].grid(row=7, column=1) 
    big['sqrt'] = Button(gui, text='sqrt', fg='black', bg='red', activebackground = 'blue',
                   command=lambda: press("sqrt("), height=1, width=7) 
    big['sqrt'].grid(row=7, column=2) 
    big['pi'] = Button(gui, text='pi', fg='black', bg='red', activebackground = 'blue',
                   command=lambda: press("pi"), height=1, width=7) 
    big['pi'].grid(row=7, column=3) 
    big['e'] = Button(gui, text='e', fg='black', bg='red', activebackground = 'blue',
                   command=lambda: press("e"), height=1, width=7) 
    big['e'].grid(row=8, column=0) 
    big['sin'] = Button(gui, text='sin', fg='black', bg='red', activebackground = 'blue',
                   command=lambda: press("sin("), height=1, width=7) 
    big['sin'].grid(row=8, column=1) 
    big['cos'] = Button(gui, text='cos', fg='black', bg='red', activebackground = 'blue',
                   command=lambda: press("cos("), height=1, width=7) 
    big['cos'].grid(row=8, column=2) 
    big['tan'] = Button(gui, text='tan', fg='black', bg='red', activebackground = 'blue',
                   command=lambda: press("tan("), height=1, width=7) 
    big['tan'].grid(row=8, column=3) 
    big['log'] = Button(gui, text='log(x,base)', fg='black', bg='red', activebackground = 'blue',
                   command=lambda: press("math.log("), height=1, width=7) 
    big['log'].grid(row=9, column=0) 
    big['asin'] = Button(gui, text='sin^(-1)', fg='black', bg='red', activebackground = 'blue',
                   command=lambda: press("asin("), height=1, width=7) 
    big['asin'].grid(row=9, column=1) 
    big['acos'] = Button(gui, text='cos^(-1)', fg='black', bg='red', activebackground = 'blue',
                   command=lambda: press("acos("), height=1, width=7) 
    big['acos'].grid(row=9, column=2) 
    big['atan'] = Button(gui, text='tan^(-1)', fg='black', bg='red', activebackground = 'blue',
                   command=lambda: press("atan("), height=1, width=7) 
    big['atan'].grid(row=9, column=3) 
    big['x'] = Button(gui, text='x', fg='black', bg='red', activebackground = 'blue',
                   command=lambda: press("x"), height=1, width=7) 
    big['x'].grid(row=10, column=0) 
    big['comma'] = Button(gui, text=',', fg='black', bg='red', activebackground = 'blue',
                   command=lambda: press(","), height=1, width=7) 
    big['comma'].grid(row=10, column=1) 
    big['factorial'] = Button(gui, text='!', fg='black', bg='red', activebackground = 'blue',
                   command=lambda: press("factorial("), height=1, width=7) 
    big['factorial'].grid(row=10, column=2) 
    big['Graph'] = Button(gui, text='Graph', fg='black', bg='red', activebackground = 'blue',
                   command=lambda: press("graph("), height=1, width=7) 
    big['Graph'].grid(row=10, column=3) 
    big['integral'] = Button(gui, text='integral(f,x)', fg='black', bg='red', activebackground = 'blue',
                   command=lambda: press("integral("), height=1, width=7) 
    big['integral'].grid(row=11, column=0) 
    big['defint'] = Button(gui, text='def(f,x,a,b)', fg='black', bg='red', activebackground = 'blue',
                   command=lambda: press("defint("), height=1, width=7) 
    big['defint'].grid(row=11, column=1) 
    big['deriv'] = Button(gui, text='deriv(f,x)', fg='black', bg='red', activebackground = 'blue',
                   command=lambda: press("deriv("), height=1, width=7) 
    big['deriv'].grid(row=11, column=2) 
    big['intersect'] = Button(gui, text='intersect(f,g)', fg='black', bg='red', activebackground = 'blue',
                   command=lambda: press("intersection("), height=1, width=7) 
    big['intersect'].grid(row=11, column=3)
    big['clearline'] = Button(gui, text='Clear Line', fg='black', bg='red', activebackground = 'blue',
                   command=clearline, height=1, width=7) 
    big['clearline'].grid(row=5, column=2) 
    big['onevar'] = Button(gui, text='1varstats', fg='black', bg='red', activebackground = 'blue',
                   command=lambda: press("varstats("), height=1, width=7) 
    big['onevar'].grid(row=12, column=0) 
    big['help'] = Button(gui, text='Help', fg='black', bg='green', activebackground = 'blue',
                   command=help, height=1, width=7) 
    big['help'].grid(row=15, column=0)  
    big['perm'] = Button(gui, text='perm(n,r)', fg='black', bg='red', activebackground = 'blue',
                   command=lambda: press("perm("), height=1, width=7) 
    big['perm'].grid(row=12, column=1) 
    big['combo'] = Button(gui, text='combo(n,r)', fg='black', bg='red', activebackground = 'blue',
                   command=lambda: press("combo("), height=1, width=7) 
    big['combo'].grid(row=12, column=2) 
    big['randint'] = Button(gui, text='randint(l,h,n)', fg='black', bg='red', activebackground = 'blue',
                   command=lambda: press("randint("), height=1, width=7) 
    big['randint'].grid(row=12, column=3) 
    big['rand'] = Button(gui, text='rand(l,h,n)', fg='black', bg='red', activebackground = 'blue',
                   command=lambda: press("rand("), height=1, width=7) 
    big['rand'].grid(row=13, column=0) 
    big['fract'] = Button(gui, text='fract(n)', fg='black', bg='red', activebackground = 'blue',
                   command=lambda: press("fract("), height=1, width=7) 
    big['fract'].grid(row=13, column=1) 
    big['zeros'] = Button(gui, text='zeros(f)', fg='black', bg='red', activebackground = 'blue',
                   command=lambda: press("zeros("), height=1, width=7) 
    big['zeros'].grid(row=13, column=2) 
    big['eval'] = Button(gui, text='eval(f,x)', fg='black', bg='red', activebackground = 'blue',
                   command=lambda: press("evaluate("), height=1, width=7) 
    big['eval'].grid(row=13, column=3) 
    big['linreg'] = Button(gui, text='linreg(x,y)', fg='black', bg='red', activebackground = 'blue',
                   command=lambda: press("linreg("), height=1, width=7) 
    big['linreg'].grid(row=14, column=0) 
    big['polyreg'] = Button(gui, text='poly(x,y,n)', fg='black', bg='red', activebackground = 'blue',
                   command=lambda: press("polyreg("), height=1, width=7) 
    big['polyreg'].grid(row=14, column=1) 
    big['expreg'] = Button(gui, text='expreg(x,y)', fg='black', bg='red', activebackground = 'blue',
                   command=lambda: press("expreg("), height=1, width=7) 
    big['expreg'].grid(row=14, column=2) 
    big['det'] = Button(gui, text='det(mtrix(x))', fg='black', bg='red', activebackground = 'blue',
                   command=lambda: press("det("), height=1, width=7) 
    big['det'].grid(row=14, column=3) 
    gui.mainloop()
