import numpy as np
from data import Data 
import os
cwd = os.getcwd()


#Node to create the tree
class NOde:
	"""docstring for NOde"""
	def __init__(self, attribute, label, is_label):
		if not is_label:
			sel.is_label = False
			self.attribute = attribute
			self.children = {}
		else:
			self.is_label= True
			self.label = label

		
def read_data(file):
	with open(file) as f:
		for line in f:
			terms =line.strip().split(',')
			print(terms)



training_data = Data(fpath ='car/train.csv')