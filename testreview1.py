import pylab as pb
import random
import numpy as np
from matplotlib import pyplot as plt
import scipy as sp

### MEASURING COVARIANCE ###
def compute_deviation_vector(vals):
	mean = pb.mean(vals)
	return [(x-mean) for x in vals]

def covariance(v1, v2):
	v1_deviations = compute_deviation_vector(v1)
	v2_deviations = compute_deviation_vector(v2)
	return pb.dot(v1_deviations, v2_deviations)/len(v1)

## A number between 0 and 1. 0 is no correlation, 1 is
## perfect correlation (-1 is inverse correlation)
def correlation(x,y):
	std_dev_x = x.std()
	std_dev_y = y.std()
	return covariance(x,y)/std_dev_x/std_dev_y

#EXAMPLE OF CORRELATION AND COVERIANCE###
##PAGE SPEED VS PURCHASE AMOUNT##
pageRenderTime = np.random.normal(3.0, 1.0, 100)
purchaseAmount = np.random.normal(50.0, 10.0, 100)/pageRenderTime
fig1 = plt.figure(1)
fig1.suptitle('Page Speed vs. Purchase Amount')
plt.xlabel('Page Render Time \n Coveriance = {c}'.format(c=covariance(pageRenderTime,purchaseAmount)))
plt.ylabel('purchase amount')
plt.grid()
plt.scatter(pageRenderTime, purchaseAmount)
#plt.show()

## using numpy to calculate correlation and covariance:

'''
print 'Numpy Correlation'
print np.corrcoef(pageRenderTime, purchaseAmount)
print 'vs our correlation fn:'
print correlation(pageRenderTime, purchaseAmount)
print '\n Numpy covariance'
print np.cov(pageRenderTime, purchaseAmount)
print 'vs our cov fn'
print covariance(pageRenderTime, purchaseAmount) 
'''

## DATA MODELING AND CURVE FITTING ##


hours = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
requests = np.array([2272, 1386, 1365, 1488, 1337, 1883, 
            2283, 1335, 1025, 1139, 1447, 1203])
if sp.sum(sp.isnan(requests)) > 0:
	print "nans found"
	requests = requests[~sp.isnan(requests)]
	hours = hours[~sp.isnan(requests)]
