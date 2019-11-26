from tkinter import *
import math
from math import *  
from sympy import *
import numpy as np
import matplotlib.pyplot as plt
from word2number import w2n
from num2words import num2words
expression = "" 
big = {}  
x = Symbol("x")  
y = Symbol("y")
def press(num): #update expression in text box
    global expression 
    expression = expression + str(num) 
    equation.set(expression)  
def equalpress(): #evaluate final expression
    try:   
        global expression 
        total = str(eval(expression))  
        equation.set(total) 
        expression = "" 
    except:  
        equation.set(" error ") 
        expression = ""  
def clear(): 
    global expression 
    expression = "" 
    equation.set("") 
def bs():
	global expression
	expression = expression[:-1]  
	equation.set(expression)
def graph(formula):
	formula = str(formula)
	x = np.array(range(-10, 10))
	y = eval(formula)
	plt.plot(x, y)
	plt.show()
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
	return string
def defint(f, x, a, b):
	return integrate(f, (x, a, b))
def deriv(exp, x):
	return diff(exp, x)
def evaluate(formula, x):
	return eval(str(formula))
def intersection(one, two):
	xcord = solve(one - two, x)[0]
	ycord = evaluate(one, xcord)
	return [xcord, ycord]
file1 = open("test.txt","r+") 
express = file1.read()
arr=[]
word = ""
st= ""
boo = True
temp = ""
realtemp = ""
for world in express.split():
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
if __name__ == "__main__": 
    gui = Tk() 
    gui.title("Graphing Calculator")   
    gui.geometry("335x335") 
    equation = StringVar() 
    expression_field = Entry(gui, textvariable=equation) #text entry box 
    expression_field.grid(columnspan=4, ipadx=70) #placing widgets in position
    equation.set('enter your expression') 
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
    big['equal'] = Button(gui, text=' = ', fg='black', bg='red', activebackground = 'blue',
                   command=equalpress, height=1, width=7) 
    big['equal'].grid(row=5, column=2) 
    big['clear'] = Button(gui, text='Clear', fg='black', bg='red', activebackground = 'blue',
                   command=clear, height=1, width=7) 
    big['clear'].grid(row=5, column='1') 
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
    expression = st
    equation.set(expression)
    gui.mainloop()
