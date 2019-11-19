from tkinter import *
import math
from math import *  
from sympy import *
import numpy as np
import matplotlib.pyplot as plt
expression = "" 
big = {}  
x = Symbol("x")  
y = Symbol("y")
# Function to update expressiom 
# in the text entry box 
def press(num): 
    # point out the global expression variable 
    global expression 
  
    # concatenation of string 
    expression = expression + str(num) 
  
    # update the expression by using set method 
    equation.set(expression)  
# Function to evaluate the final expression 
def equalpress(): 
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
if __name__ == "__main__": 
    # create a GUI window 
    gui = Tk() 
  
    # set the background colour of GUI window 
    #gui.configure(background="light green") 
    gui.title("Scientific Calculator")   
    # set the configuration of GUI window 
    gui.geometry("335x325") 
  
    # StringVar() is the variable class 
    # we create an instance of this class 
    equation = StringVar() 
  
    # create the text entry box for 
    # showing the expression . 
    expression_field = Entry(gui, textvariable=equation) 
  
    # grid method is used for placing 
    # the widgets at respective positions 
    # in table like structure . 
    expression_field.grid(columnspan=4, ipadx=70) 
  
    equation.set('enter your expression') 
  
    # create a Buttons and place at a particular 
    # location inside the root window . 
    # when user press the button, the command or 
    # function affiliated to that button is executed . 
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
                   command=lambda: press("y="), height=1, width=7) 
    big['Graph'].grid(row=10, column=3) 
    gui.mainloop()
