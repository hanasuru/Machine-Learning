from itertools import *
import matplotlib.pyplot as plt
import numpy as np

class Perceptron:
	# Initialize a Constructor
	def __init__(self, filename):
		self.data 	  = self.parseCSV(filename)
		self.epoch 	  = 300
		self.target   = self.data[:,-1:]
		self.unit     = self.data[:,:-1]
	
	# Parse CSV file per rows
	def parseCSV(self, filename):
		with open(filename) as raw:
			txt  	  	= raw.read().split('\n')
			W,H 		= len(txt), len(txt[0].split(','))
			data 	  	= np.empty([W, H], dtype='float64')
			for x,y in enumerate(txt):
			  data[x,:] = list(map(float,y.split(',')))
		return data

	def calcActivation(self, units):
		computed = self.bias + sum(i*j for i,j in izip(units, self.weight))
		return 1.0/(1+np.exp(-computed))

	def computeError(self, activation, type_f):
		return (activation - type_f)**2

	def predictActivation(self, num):
		return 1 if num>0.5 else 0

	# Split Dataset for Training & Validation
	def segment(self):
		self.valid = map(lambda x : np.vstack((self.data[x-10:x,:],
					 self.data[x+40:x+50,:])), range(10,60,10))[::-1]
		order  	   = list(combinations(range(5),4))[::-1]
		self.train = map(lambda x : np.array([self.valid])
					 [:,[x]][0][0].reshape((80,5)), order)

	# Train a dataset
	def training(self, data):
		sumError     = 0.0
		count		 = 0

		for row in data:
			A 		= self.calcActivation(row[:-1])
			P 		= self.predictActivation(A)
		 	count  += 1 if P==row[-1] else 0
		 	y       = row[-1]

		 	dweight 	= [2*x*(A-y)*A*(1-A) for x in row[:-1]]
			dbias		= 2.0*(A-y)*A*(1.0-A)
			self.weight = [x-(self.learnRate*z) for x,z in izip(self.weight, dweight)]
			self.bias	= self.bias-(self.learnRate * dbias)
			sumError   += self.computeError(A,y) 
		self.errorT    += sumError/data.shape[0]
		self.accT	   += count/data.shape[0]

	# Validate a dataset
	def validate(self, data):
		sumError = 0.0
		count    = 0
		for row in data:
			A 	     = self.calcActivation(row[:-1])
			P 		 = self.predictActivation(A)
			count   += 1 if P==row[-1] else 0
			sumError+= self.computeError(A,row[-1])
		self.errorV += sumError/data.shape[0]
		self.accV   += count/data.shape[0]

	# Visualize the collection of Accuracy & Loss value
	def plot(self, X, Y, Xname, Yname, title):
		plt.plot(X, label=Xname)
		plt.plot(Y, label=Yname)
		plt.title('{}, Learning Rate : {}'.format(title, self.learnRate))
		plt.ylabel(title)
		plt.legend()
		plt.show()

	# Initialize training within n-epoch times
	def doTrain(self, learnRate, bias, weight):
		self.learnRate 		= learnRate
		self.bias	   		= bias
		self.weight   		= [weight] * 4
		self.epochTrainAcc  = []; self.epochTrainErr  = []
		self.epochValidAcc	= []; self.epochValidErr  = []

		for j in range(self.epoch):
			self.errorT   = 0.0; self.errorV = 0.0
			self.accT     = 0.0; self.accV   = 0.0
			for i in range(len(self.train)):
				self.training(self.train[i])
				self.validate(self.valid[i][:20])
			self.epochTrainAcc.append(self.accT/5.0)
			self.epochTrainErr.append(self.errorT/5.0)
			self.epochValidAcc.append(self.accV/5.0)
			self.epochValidErr.append(self.errorV/5.0)

def main():
	slp = Perceptron('iris.csv')
	slp.segment()
	
	# Do training w/t learning_rate = 0.1
	slp.doTrain(0.1, 0.5, 0.6)
	slp.plot(slp.epochTrainAcc, slp.epochValidAcc,
			 'Training', 'Validation', 'Accuracy')
	slp.plot(slp.epochTrainErr, slp.epochValidErr,
			 'Training', 'Validation', 'Loss')

	# Do training w/t learning_rate = 0.8
	slp.doTrain(0.8, 0.5, 0.6)
	slp.plot(slp.epochTrainAcc, slp.epochValidAcc,
			 'Training', 'Validation', 'Accuracy')
	slp.plot(slp.epochTrainErr, slp.epochValidErr,
			 'Training', 'Validation', 'Loss')

if __name__ == '__main__':
	main()