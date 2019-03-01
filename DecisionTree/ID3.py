import numpy as np
from data import Data 
import math
import os
cwd = os.getcwd()


#Node to create the tree
class Node:
	"""docstring for NOde"""
	def __init__(self, attribute, label, is_label):
		if not is_label:
			self.is_label = False
			self.attribute = attribute
			self.children = {}
		else:
			self.is_label= True
			self.label = label

		



#the mode decides whether to use entropy, majority error or gini index
def best_attribute(examples, attributes, mode):
	total = len(examples)
	unacc = len(examples.get_row_subset('label', 'unacc'))
	acc = len(examples.get_row_subset('label', 'acc'))
	good = len(examples.get_row_subset('label', 'good'))
	vgood = len(examples.get_row_subset('label', 'vgood'))
	sizes = [unacc, acc, good, vgood]
	total_purity = 0
	if mode ==0:
		total_purity = entropy(sizes, total);
	elif mode ==1:
		total_purity = (total-max(sizes))/total
		
	else:
		total_purity = 1 -((unacc/total)*(unacc/total) + (acc/total)*(acc/total)+ (good/total)*(good/total)+ (vgood/total)*(vgood/total))
	best = None
	biggest=0;

	for attr in attributes:

		attr_score =0 # keeps track of enthropy, ME or GINi
		for val in examples.attributes[attr].possible_vals:
			attr_val =examples.get_row_subset(attr, val)
			size_attr_val = len(attr_val)
			if size_attr_val ==0:
				continue

			unacc = len(attr_val.get_row_subset('label', 'unacc'))
			acc = len(attr_val.get_row_subset('label', 'acc'))
			good = len(attr_val.get_row_subset('label', 'good'))
			vgood = len(attr_val.get_row_subset('label', 'vgood'))
			sizes = [unacc, acc, good, vgood]
			if mode == 0: #entropy
				
				val_entropy = entropy(sizes,size_attr_val)
				attr_score  += val_entropy*(size_attr_val/total)

			elif mode ==1:
				
				attr_score += ((size_attr_val-max(sizes))/size_attr_val)*(size_attr_val/total)
				

			else:
				gini_purity = gini(sizes,size_attr_val)
				attr_score += gini_purity*(size_attr_val/total)

		#information gain
		if(total_purity -attr_score) >= biggest:
			biggest = total_purity- attr_score
			best = attr
	return best

def ID3(data, attributes, label, curr_depth, depth_cap, max_depth,mode):
	if len(data) ==0 or len(data.get_row_subset('label', 'unacc')) == len(data) or len(data.get_row_subset('label', 'acc')) == len(data) or len(data.get_row_subset('label', 'good')) == len(data) or len(data.get_row_subset('label', 'vgood')) == len(data):
		return Node(None, label, True)

	if depth_cap and curr_depth == max_depth:
		return Node(None, label, True)
	else:
		#print(curr_depth)
		node_attribute = best_attribute(data, attributes,mode)
		root = Node(node_attribute, None, False)
		node_atr_obj = attributes[node_attribute]

		for val in attributes[node_attribute].possible_vals:
			attr_val = data.get_row_subset(node_attribute, val)
			new_label = find_majority_label(attr_val)

			if len(attr_val) ==0:
				root.children[val] = Node(None, new_label, True)
			else:
				attributes.pop(node_attribute)
				root.children[val] = ID3(attr_val, attributes, new_label, curr_depth + 1, depth_cap, max_depth, mode)
				attributes[node_attribute] = node_atr_obj
	return root

def find_majority_label(examples):
	unacc = len(examples.get_row_subset('label', 'unacc'))
	acc = len(examples.get_row_subset('label', 'acc'))
	good = len(examples.get_row_subset('label', 'good'))
	vgood = len(examples.get_row_subset('label', 'vgood'))
	sizes = [unacc, acc, good, vgood]
	maximum = max(sizes)
	if maximum == unacc:
		return 'unacc'
	elif maximum == acc:
		return 'acc'
	elif maximum == good:
		return 'good'
	else:
		return'vgood'
def sizes(examples):
	unacc = len(examples.get_row_subset('label', 'unacc'))
	acc = len(examples.get_row_subset('label', 'acc'))
	good = len(examples.get_row_subset('label', 'good'))
	vgood = len(examples.get_row_subset('label', 'vgood'))
	sizes = [unacc, acc, good, vgood]
	return sizes
def entropy(sizes, total):
	sum =0
	for p in sizes:
		if p != 0:
			sum +=(-p / total) * math.log(p / total, 2)
	return sum

def gini(sizes, total):
	sum =0
	for p in sizes:
		if p != 0:
			sum += (p/total)*(p/total)
	return 1 - sum


def find_example_label(root, dataset, example):
    #when label node is encountered
    if root.is_label:
    	return example[6] == root.label

    # index the value for the attribute is at in the example
    attr_ind = training_data.attributes[root.attribute].index + 1
    # checking which child the example should traverse to
    for val in root.children:
        if example[attr_ind] == val:
            return find_example_label(root.children[val], dataset, example)
    return False

def find_data_set_acc(data, tree):
    data = data.raw_data
    correct = 0
    # find tree's accuracy for each example in dataset
    for i in range(len(data)):
        row = data[i, :]
        
        if find_example_label(tree, training_data, row):
            correct += 1

    return correct/len(data)





print ("**********ENTROPY****************")
for i in range(1,7):
	training_data = Data(fpath ='car/train.csv')
	test_data = Data(fpath ='car/test.csv')
	tree = ID3(training_data, training_data.attributes, find_majority_label(training_data), 0, True, i, 0)
	train_error = str(find_data_set_acc(training_data, tree))
	train_error2 = str(find_data_set_acc(test_data, tree))
	print ("The decision tree with depth " +str(i)+ " using entropy has train error:")
	print (train_error)
	print ("The decision tree with depth " +str(i)+ " using entropy has test error:")
	print (train_error2)


print ("\n**************MAJORITY ERROR**********")
for i in range(1,7):
	training_data = Data(fpath ='car/train.csv')
	test_data = Data(fpath ='car/test.csv')
	tree1 = ID3(training_data, training_data.attributes, find_majority_label(training_data), 0, True, i, 1)
	m_train_error = str(find_data_set_acc(training_data, tree1))
	m_train_error2 = str(find_data_set_acc(test_data, tree1))
	print ("The decision tree with depth " +str(i)+ " using entropy has train error:")
	print (m_train_error)
	print ("The decision tree with depth " +str(i)+ " using entropy has test error:")
	print (m_train_error2)



print ("\n**************GINI INDEX**********")
for i in range(1,7):
	training_data = Data(fpath ='car/train.csv')
	test_data = Data(fpath ='car/test.csv')
	tree2 = ID3(training_data, training_data.attributes, find_majority_label(training_data), 0, True, i, 2)
	g_train_error = str(find_data_set_acc(training_data, tree2))
	g_train_error2 = str(find_data_set_acc(test_data, tree2))
	print ("The decision tree with depth " +str(i)+ " using entropy has train error:")
	print (g_train_error)
	print ("The decision tree with depth " +str(i)+ " using entropy has test error:")
	print (g_train_error2)