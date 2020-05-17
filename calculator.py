from tkinter import *
import math
from math import *
from sympy import *
import numpy as np
import numpy.polynomial.polynomial as poly
from scipy import stats
from scipy.optimize import fsolve
from scipy.optimize import curve_fit
from scipy.optimize import fminbound
from statistics import *
import random
from fractions import Fraction
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from word2number import w2n
from num2words import num2words
import speech_recognition as sr
from gtts import gTTS
import pyaudio
import time
import threading
from multiprocessing import Queue
from playsound import playsound
r = sr.Recognizer()
expression = ""
carry = "" 
big = {}
alpha = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
x = Symbol("x")
y = Symbol("y")
function = ""
nocalc = 1.0
gui = Tk()
expression_field = Text(gui, height = 16, width = 30)
class SpeechRecognizer(threading.Thread):
	def __init__(self):
		super(SpeechRecognizer, self).__init__()
		self.setDaemon(True)
		self.recognized_text = "initial"
	def run(self):
		while True:
			time.sleep(1.0)
			with sr.Microphone() as source:
				#r.adjust_for_ambient_noise(source)
				audio = r.listen(source)
			try:
				a=(r.recognize_google(audio))
				self.recognized_text = a
				print(a)
				if(a == "Translate"):
					translate()
				elif(a == "equals" or a == "equal"):
					equalpress()
				elif(a == "answer"):
					numtoword()
				elif(a == "clear all"):
					clearall()
				elif(a == "clearline"):
					clearline()
				elif(a == "backspace"):
					bs()
				elif(a == "help"):
					help()
				elif(a == "repeat"):
					repeat()
				elif(a == "exit"):
					exit()
				else:
					expression_field.insert(END, a)
			except LookupError:
				a=("Could not understand audio")
				self.recognized_text = a
				print(a)
			except KeyboardInterrupt:
				pass
recognizer = SpeechRecognizer()
recognizer.start()
def press(num): #update expression in text box
	global expression
	expression = expression + str(num)
	expression_field.insert(END, num)
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
	#print(len(arr))
	for i in range(len(arr)):
		if arr[i] == "plus" or arr[i] == "+":
			word = "+"
		elif arr[i] == "minus" or arr[i] == "negative" or arr[i] == "-" or arr[i] == "cookie":
			word = "-"
		elif arr[i] == "times" or arr[i] == "*":
			word = "*"
		elif arr[i] == "divided" or arr[i] == "/":
			word = "/"
		elif arr[i] == "point":
			word = "."
		elif arr[i] == "power" or arr[i] == "^":
			word = "**"
		elif arr[i] == "square": #math
			word = "sqrt"
		elif arr[i] == "absolute": #built-in
			word = "abs"
		elif arr[i] == "factorial" or arr[i] == "factorio": #math
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
		elif arr[i] == "differentiate":
			word = "differ"
		elif arr[i] == "log": #math
			word = "log"
		elif arr[i] == "cosine": #math
			word = "cos"
		elif arr[i] == "sine": #math
			word = "sin"
		elif arr[i] == "tangent": #math
			word = "tan"
		elif (arr[i] == "by" or arr[i] == "parentheses" or arr[i] == "parenthesis" or arr[i] == "root" or 
		arr[i] == "of" or arr[i] == "value" or arr[i] == "to" or arr[i] == "the" or arr[i] == "base"):
			word = ""
		elif arr[i] == "left":
			word = "("
		elif arr[i] == "right":
			word = ")"
		elif arr[i] == "left-bracket" or arr[i] == "l" or arr[i] == "bracket":
			word = "["
		elif arr[i] == "right-bracket" or arr[i] == "r" or arr[i] == "are" or arr[i] == "our":
			word = "]"
		elif arr[i] == "comma" or arr[i] == "continue" or arr[i] == "continued":
			word = ","
		elif arr[i] == "pi" or arr[i] == "Pi": #math
			word = "math.pi" 
		elif arr[i] == "e": #math
			word = "e"
		elif arr[i] == "x" or arr[i] == "X":
			word = "x"
		elif arr[i] == "var-stats" or arr[i] == "variable":
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
		elif arr[i] == "store":
			word = "store"
		elif arr[i] == "minimum":
			word = "minimum"
		elif arr[i] == "maximum":
			word = "maximum"
		elif arr[i] == "function":
			word = "function"
		elif arr[i] == "plot":
			word = "plot"
		elif arr[i] == "plots":
			word = "plots"
		elif arr[i] == "to":
			word = "two"
		elif arr[i] == "for":
			word = "four"
		else:
			word = arr[i]
		temp += arr[i]
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
def equalpress(): #evaluate symbolic expression
	global nocalc
	global expression
	global carry 
	try:
		total = str(eval(expression))
		carry = total
		expression_field.insert(END, "\n")
		expression_field.insert(END, total + "\n")
	except Exception as err:
		expression_field.insert(END, "\n")
		expression_field.insert(END, "error")
		expression_field.insert(END, "\n")
		#expression_field.insert(END, "error")
		print(err)
		raise(err)
def numtoword(): #evaluate final expression
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
	#myobj = gTTS(text=st, lang='en', slow=False)
	#myobj.save("answer.wav")
	#playsound('answer.wav')
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
def graph(formula):
	formula = str(formula)
	init = 0
	start = -10
	if(formula[0:4] == "sqrt"):
		start = 0
	if(formula[0:3] == "log"):
		start = 1
	x = np.arange(start, 10, 0.05)
	y = np.empty_like(x)
	for thing in np.arange(start, 10, 0.05):
		y[init] = evaluate(formula, thing)
		init += 1
	fig = plt.figure()
	plt.plot(x,y)
	plt.xlabel('x')
	plt.ylabel(formula)
	fig.savefig('plot.png')
	plt.show()
	playgraph(formula)
	return "Click equals text to continue"
def plot(x, y):
	fig = plt.figure()
	plt.plot(x, y, 'ro')
	fig.savefig('plot.png')
	plt.show()
def plots(x, typeof):
	fig = plt.figure()
	if(str(typeof) == "1"):
		plt.hist(x)
		plt.suptitle('Histogram')
	if(str(typeof) == "2"):
		plt.boxplot(x)
		plt.suptitle('Box Plot')
	fig.savefig('plot.png')
	plt.show()
def matrix(mat):
	return Matrix(mat)
def det(mat):
	return mat.det()
def rref(mat):
	return mat.rref()
def varstats(arr):
	df = pd.DataFrame(arr)
	expression_field.insert(END, df)
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
		elif(z[i] == "s" and z[i+1] == "i" and z[i+2] == "n"):
			string += "sine"
			i+= 3
		elif(z[i] == "c" and z[i+1] == "o" and z[i+2] == "s"):
			string += "cosine"
			i+=3
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
def differ(exp, x, val):
	epp = deriv(exp, x)
	return lambdify(x, epp)(val)
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
	guide = f.read()
	expression_field.insert(END, guide)
	f.close()
	myobj = gTTS(text=guide, lang='en', slow=False)
	myobj.save("help.wav")
	os.system("ffplay help.wav")
def intersection(one, two):
	xcord = solve(one - two, x)[0]
	ycord = evaluate(one, xcord)
	return [xcord, ycord]
def evalat(s):
	def f(x):
		return eval(str(s))
	return f
def zeros(funct, guess):
	return fsolve(evalat(funct), guess)
def minimum(funct, one, two):
	xonly = fminbound(evalat(funct), one, two)
	return [xonly, evaluate(funct, xonly)]
def maximum(funct, one, two):
	xonly = fminbound(evalat(-1*(funct)), one, two)
	return [xonly, evaluate(funct, xonly)]
def repeat():
	playsound('answer.wav')
def store(funct):
	global function
	function = funct
	return function
def exit():
	sys.exit()
class ToneGenerator(object):
	def __init__(self, samplerate=44100, frames_per_buffer=4410):
		self.p = pyaudio.PyAudio()
		self.samplerate = samplerate
		self.frames_per_buffer = frames_per_buffer
		self.streamOpen = False
	def sinewave(self):
		if self.buffer_offset + self.frames_per_buffer - 1 > self.x_max:
			xs = np.arange(self.buffer_offset, self.x_max)
			tmp = self.amplitude * np.sin(xs * self.omega)
			out = np.append(tmp, np.zeros(self.frames_per_buffer - len(tmp)))
		else:
			xs = np.arange(self.buffer_offset, self.buffer_offset + self.frames_per_buffer)
			out = self.amplitude * np.sin(xs * self.omega)
			self.buffer_offset += self.frames_per_buffer
		return out
	def callback(self, in_data, frame_count, time_info, status):
		if self.buffer_offset < self.x_max:
			data = self.sinewave().astype(np.float32)
			return (data.tostring(), pyaudio.paContinue)
		else:
			return (None, pyaudio.paComplete)
	def is_playing(self):
		if self.stream.is_active():
			return True
		else:
			if self.streamOpen:
				self.stream.stop_stream()
				self.stream.close()
				self.streamOpen = False
			return False
	def play(self, frequency, duration, amplitude):
		self.omega = float(frequency) * (math.pi * 2) / self.samplerate
		self.amplitude = amplitude
		self.buffer_offset = 0
		self.streamOpen = True
		self.x_max = math.ceil(self.samplerate * duration) - 1
		self.stream = self.p.open(format=pyaudio.paFloat32, channels=1, rate=self.samplerate, output=True, 
			 frames_per_buffer=self.frames_per_buffer, stream_callback=self.callback)
def playgraph(formula):
	generator = ToneGenerator()
	frequency_start = 50
	frequency_end = 10000
	amplitude = 0.10
	step_duration = 20
	smallest = 10000
	small_pos = 0
	largest = -10000
	large_pos = 0
	if(formula.isdigit() or (len(formula) > 1 and formula[1] == ".")):
		for numb in np.arange(-10, 10, 0.05):
			print("Playing tone at {0:0.2f} Hz".format(frequency_start))
			generator.play(frequency_start, step_duration, amplitude)
	else:
		starting = -10
		if(formula[0:4] == "sqrt"):
			starting = 0
		if(formula[0:3] == "log"):
			starting = 1
		for numb in np.arange(starting, 10, 0.6):
			elem = evaluate(formula, numb)
			if(elem > largest):
				largest = elem
				large_pos = numb
			if(elem < smallest):
				smallest = elem
				small_pos = numb
		slope=(frequency_end-frequency_start)/(large_pos-small_pos)
		finslope = slope/((largest-smallest)/(large_pos-small_pos))
		start = finslope * smallest
		for numb in np.arange(starting, 10, 0.6):
			elem = evaluate(formula, numb)
			frequency = math.fabs(frequency_start +
								finslope * elem - start)
			print("Playing tone at {0:0.2f} Hz".format(frequency))
			generator.play(frequency, step_duration, amplitude)
			time.sleep(0.02)
if __name__ == "__main__":
	gui.title("Graphing Calculator")
	gui.geometry("388x875")
	expression_field.focus()
	expression_field.grid(columnspan=4, ipadx=70) #placing widgets in position
	big['1'] = Button(gui, text=' 1 ', fg='black', bg='red', activebackground = 'blue', command=lambda: press(1), height=1, width=7) 
	big['1'].grid(row=2, column=0)
	big['2'] = Button(gui, text=' 2 ', fg='black', bg='red', activebackground = 'blue', command=lambda: press(2), height=1, width=7)
	big['2'].grid(row=2, column=1)
	big['3'] = Button(gui, text=' 3 ', fg='black', bg='red', activebackground = 'blue', command=lambda: press(3), height=1, width=7)
	big['3'].grid(row=2, column=2)
	big['4'] = Button(gui, text=' 4 ', fg='black', bg='red', activebackground = 'blue', command=lambda: press(4), height=1, width=7)
	big['4'].grid(row=3, column=0)
	big['5'] = Button(gui, text=' 5 ', fg='black', bg='red', activebackground = 'blue', command=lambda: press(5), height=1, width=7)
	big['5'].grid(row=3, column=1)
	big['6'] = Button(gui, text=' 6 ', fg='black', bg='red', activebackground = 'blue', command=lambda: press(6), height=1, width=7)
	big['6'].grid(row=3, column=2)
	big['7'] = Button(gui, text=' 7 ', fg='black', bg='red', activebackground = 'blue', command=lambda: press(7), height=1, width=7)
	big['7'].grid(row=4, column=0)
	big['8'] = Button(gui, text=' 8 ', fg='black', bg='red', activebackground = 'blue', command=lambda: press(8), height=1, width=7)
	big['8'].grid(row=4, column=1)
	big['9'] = Button(gui, text=' 9 ', fg='black', bg='red', activebackground = 'blue', command=lambda: press(9), height=1, width=7)
	big['9'].grid(row=4, column=2)
	big['0'] = Button(gui, text=' 0 ', fg='black', bg='red', activebackground = 'blue', command=lambda: press(0), height=1, width=7)
	big['0'].grid(row=5, column=0)
	big['plus'] = Button(gui, text=' + ', fg='black', bg='red', activebackground = 'blue',command=lambda: press("+"), height=1, width=7)
	big['plus'].grid(row=2, column=3)
	big['minus'] = Button(gui, text=' - ', fg='black', bg='red', activebackground = 'blue',command=lambda: press("-"), height=1, width=7)
	big['minus'].grid(row=3, column=3)
	big['multiply'] = Button(gui, text=' * ', fg='black', bg='red', activebackground = 'blue',command=lambda: press("*"), height=1, width=7)
	big['multiply'].grid(row=4, column=3)
	big['divide'] = Button(gui, text=' / ', fg='black', bg='red', activebackground = 'blue',command=lambda: press("/"), height=1, width=7)
	big['divide'].grid(row=5, column=3)
	big['equal'] = Button(gui, text=' = ', fg='black', bg='light blue', activebackground = 'blue',command=equalpress, height=1, width=7)
	big['equal'].grid(row=18, column=2)
	big['equalwords'] = Button(gui, text=' =(text) ', fg='black', bg='light blue', activebackground = 'blue',command=numtoword, height=1, width=7)
	big['equalwords'].grid(row=18, column=3)
	big['translate'] = Button(gui, text=' translate ', fg='black', bg='light blue', activebackground = 'blue',command=translate, height=1, width=7)
	big['translate'].grid(row=18, column=1)
	big['clearall'] = Button(gui, text='Clear All', fg='black', bg='red', activebackground = 'blue',command=clearall, height=1, width=7)
	big['clearall'].grid(row=5, column='1')
	big['.'] = Button(gui, text='.', fg='black', bg='red', activebackground = 'blue',command=lambda: press("."), height=1, width=7)
	big['.'].grid(row=6, column=0)
	big['<-'] = Button(gui, text='<-', fg='black', bg='red', activebackground = 'blue',command=bs, height=1, width=7)
	big['<-'].grid(row=6, column=1)
	big['('] = Button(gui, text='(', fg='black', bg='red', activebackground = 'blue',command=lambda: press("("), height=1, width=7)
	big['('].grid(row=6, column=2)
	big[')'] = Button(gui, text=')', fg='black', bg='red', activebackground = 'blue',command=lambda: press(")"), height=1, width=7)
	big[')'].grid(row=6, column=3)
	big['**'] = Button(gui, text='a^b', fg='black', bg='red', activebackground = 'blue',command=lambda: press("**"), height=1, width=7)
	big['**'].grid(row=7, column=0)
	big['abs'] = Button(gui, text='|x|', fg='black', bg='red', activebackground = 'blue',command=lambda: press("abs("), height=1, width=7)
	big['abs'].grid(row=7, column=1)
	big['sqrt'] = Button(gui, text='sqrt', fg='black', bg='red', activebackground = 'blue',command=lambda: press("sqrt("), height=1, width=7)
	big['sqrt'].grid(row=7, column=2)
	big['pi'] = Button(gui, text='pi', fg='black', bg='red', activebackground = 'blue',command=lambda: press("pi"), height=1, width=7)
	big['pi'].grid(row=7, column=3)
	big['e'] = Button(gui, text='e', fg='black', bg='red', activebackground = 'blue',command=lambda: press("e"), height=1, width=7)
	big['e'].grid(row=8, column=0)
	big['sin'] = Button(gui, text='sin', fg='black', bg='red', activebackground = 'blue',command=lambda: press("sin("), height=1, width=7)
	big['sin'].grid(row=8, column=1)
	big['cos'] = Button(gui, text='cos', fg='black', bg='red', activebackground = 'blue',command=lambda: press("cos("), height=1, width=7)
	big['cos'].grid(row=8, column=2)
	big['tan'] = Button(gui, text='tan', fg='black', bg='red', activebackground = 'blue',command=lambda: press("tan("), height=1, width=7)
	big['tan'].grid(row=8, column=3)
	big['log'] = Button(gui, text='log(x,base)', fg='black', bg='red', activebackground = 'blue',command=lambda: press("math.log("), height=1, width=7)
	big['log'].grid(row=9, column=0)
	big['asin'] = Button(gui, text='sin^(-1)', fg='black', bg='red', activebackground = 'blue',command=lambda: press("asin("), height=1, width=7)
	big['asin'].grid(row=9, column=1)
	big['acos'] = Button(gui, text='cos^(-1)', fg='black', bg='red', activebackground = 'blue',command=lambda: press("acos("), height=1, width=7)
	big['acos'].grid(row=9, column=2)
	big['atan'] = Button(gui, text='tan^(-1)', fg='black', bg='red', activebackground = 'blue',command=lambda: press("atan("), height=1, width=7)
	big['atan'].grid(row=9, column=3)
	big['x'] = Button(gui, text='x', fg='black', bg='red', activebackground = 'blue',command=lambda: press("x"), height=1, width=7)
	big['x'].grid(row=10, column=0)
	big['comma'] = Button(gui, text=',', fg='black', bg='red', activebackground = 'blue',command=lambda: press(","), height=1, width=7)
	big['comma'].grid(row=10, column=1)
	big['factorial'] = Button(gui, text='!', fg='black', bg='red', activebackground = 'blue',command=lambda: press("factorial("), height=1, width=7)
	big['factorial'].grid(row=10, column=2)
	big['Graph'] = Button(gui, text='Graph', fg='black', bg='red', activebackground = 'blue',command=lambda: press("graph("), height=1, width=7)
	big['Graph'].grid(row=10, column=3)
	big['integral'] = Button(gui, text='integral(f,x)', fg='black', bg='red', activebackground = 'blue',command=lambda: press("integral("), height=1, width=7)
	big['integral'].grid(row=11, column=0)
	big['defint'] = Button(gui, text='def(f,x,a,b)', fg='black', bg='red', activebackground = 'blue',command=lambda: press("defint("), height=1, width=7)
	big['defint'].grid(row=11, column=1)
	big['deriv'] = Button(gui, text='deriv(f,x)', fg='black', bg='red', activebackground = 'blue',command=lambda: press("deriv("), height=1, width=7)
	big['deriv'].grid(row=11, column=2)
	big['intersect'] = Button(gui, text='intersect(f,g)', fg='black', bg='red', activebackground = 'blue',command=lambda: press("intersection("), height=1, width=7)
	big['intersect'].grid(row=16, column=0)
	big['clearline'] = Button(gui, text='Clear Line', fg='black', bg='red', activebackground = 'blue',command=clearline, height=1, width=7)
	big['clearline'].grid(row=5, column=2)
	big['onevar'] = Button(gui, text='1varstats', fg='black', bg='red', activebackground = 'blue',command=lambda: press("varstats("), height=1, width=7)
	big['onevar'].grid(row=12, column=0)
	big['help'] = Button(gui, text='Help', fg='black', bg='green', activebackground = 'blue',command=help, height=1, width=7)
	big['help'].grid(row=18, column=0)
	big['perm'] = Button(gui, text='perm(n,r)', fg='black', bg='red', activebackground = 'blue',command=lambda: press("perm("), height=1, width=7)
	big['perm'].grid(row=12, column=1)
	big['combo'] = Button(gui, text='combo(n,r)', fg='black', bg='red', activebackground = 'blue',command=lambda: press("combo("), height=1, width=7)
	big['combo'].grid(row=12, column=2)
	big['randint'] = Button(gui, text='randint(l,h,n)', fg='black', bg='red', activebackground = 'blue',command=lambda: press("randint("), height=1, width=7)
	big['randint'].grid(row=12, column=3)
	big['rand'] = Button(gui, text='rand(l,h,n)', fg='black', bg='red', activebackground = 'blue',command=lambda: press("rand("), height=1, width=7)
	big['rand'].grid(row=13, column=0)
	big['fract'] = Button(gui, text='fract(n)', fg='black', bg='red', activebackground = 'blue',command=lambda: press("fract("), height=1, width=7)
	big['fract'].grid(row=13, column=1)
	big['zeros'] = Button(gui, text='zeros(f,g)', fg='black', bg='red', activebackground = 'blue',command=lambda: press("zeros("), height=1, width=7)
	big['zeros'].grid(row=13, column=2)
	big['eval'] = Button(gui, text='eval(f,x)', fg='black', bg='red', activebackground = 'blue',command=lambda: press("evaluate("), height=1, width=7)
	big['eval'].grid(row=13, column=3)
	big['linreg'] = Button(gui, text='linreg(x,y)', fg='black', bg='red', activebackground = 'blue',command=lambda: press("linreg("), height=1, width=7)
	big['linreg'].grid(row=14, column=0)
	big['polyreg'] = Button(gui, text='poly(x,y,n)', fg='black', bg='red', activebackground = 'blue',command=lambda: press("polyreg("), height=1, width=7)
	big['polyreg'].grid(row=14, column=1)
	big['expreg'] = Button(gui, text='expreg(x,y)', fg='black', bg='red', activebackground = 'blue',command=lambda: press("expreg("), height=1, width=7)
	big['expreg'].grid(row=14, column=2)
	big['det'] = Button(gui, text='det(mtrx(x))', fg='black', bg='red', activebackground = 'blue',command=lambda: press("det("), height=1, width=7)
	big['det'].grid(row=14, column=3)
	big['differ'] = Button(gui, text='dif(y,x,val)', fg='black', bg='red', activebackground = 'blue',command=lambda: press("differ("), height=1, width=7)
	big['differ'].grid(row=11, column=3)
	big['rref'] = Button(gui, text='rref(mtrx(x))', fg='black', bg='red', activebackground = 'blue',command=lambda: press("rref("), height=1, width=7)
	big['rref'].grid(row=15, column=2)
	big['repeat'] = Button(gui, text='repeat ans', fg='black', bg='red', activebackground = 'blue',command=repeat, height=1, width=7)
	big['repeat'].grid(row=16, column=1)
	big['store'] = Button(gui, text='sto->funct', fg='black', bg='red', activebackground = 'blue',command=store, height=1, width=7)
	big['store'].grid(row=15, column=3)
	big['min'] = Button(gui, text='min(f,guess)', fg='black', bg='red', activebackground = 'blue',command=lambda: press("min("), height=1, width=7)
	big['min'].grid(row=15, column=0)
	big['max'] = Button(gui, text='max(f,x)', fg='black', bg='red', activebackground = 'blue',command=lambda: press("max("), height=1, width=7)
	big['max'].grid(row=15, column=1)
	big['['] = Button(gui, text='[', fg='black', bg='red', activebackground = 'blue',command=lambda: press("["), height=1, width=7)
	big['['].grid(row=16, column=2)
	big[']'] = Button(gui, text=']', fg='black', bg='red', activebackground = 'blue',command=lambda: press("]"), height=1, width=7)
	big[']'].grid(row=16, column=3)
	big['exit'] = Button(gui, text='exit calc', fg='black', bg='green', activebackground = 'blue',command=exit, height=1, width=7)
	big['exit'].grid(row=17, column=3)
	big['plot'] = Button(gui, text='plot(x,y)', fg='black', bg='red', activebackground = 'blue',command=lambda: press("plot("), height=1, width=7)
	big['plot'].grid(row=17, column=1)
	big['plots'] = Button(gui, text='plots(x,type)', fg='black', bg='red', activebackground = 'blue',command=lambda: press("plots("), height=1, width=7)
	big['plots'].grid(row=17, column=2)
	gui.mainloop()
