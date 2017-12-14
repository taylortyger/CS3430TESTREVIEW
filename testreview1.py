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
'''
pageRenderTime = np.random.normal(3.0, 1.0, 100)
purchaseAmount = np.random.normal(50.0, 10.0, 100)/pageRenderTime
fig1 = plt.figure(1)
fig1.suptitle('Page Speed vs. Purchase Amount')
plt.xlabel('Page Render Time \n Coveriance = {c}'.format(c=covariance(pageRenderTime,purchaseAmount)))
plt.ylabel('purchase amount')
plt.grid()
plt.scatter(pageRenderTime, purchaseAmount)
#plt.show()
'''

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


hours = np.array([1,2,3,4,5,6,7,8,9,10,11,12,
				  1,2,3,4,5,6,7,8,9,10,11,12,
				  1,2,3,4,5,6,7,8,9,10,11,12])
requests = np.array([random.randint(1000,5000)+(2**hours[i]) for i in xrange(len(hours))])

print requests
print correlation(hours, requests)
#this is an example of how you would "clean" data...
if sp.sum(sp.isnan(requests)) > 0:
	print "nans found"
	requests = requests[~sp.isnan(requests)]
	hours = hours[~sp.isnan(requests)]

# what we are trying to do is create a model that will reasonobly
# estimate how many server hits there will be given an hour... 
# so, we will want to know how accurate our prediction model is, so we will
# create a function to calculate the model error:
def error(model, x, y):
	# error is the sum of the squared differences 
	# between the model predictions and the actual data
	return sp.sum((model(x)-y)**2)

# now let's try to fit a curve to our data, we can try higher
# and higher polynomials, (x^2) vs (x^3) ... vs(x^9) etc
#   			...
# sp.polyfit(x, y, polynomial degree, full=True)
poly_coeffs, error, rank, sv, rcond = sp.polyfit(hours, requests, 3, full=True)

#linear model function:
f1 = sp.poly1d(sp.polyfit(hours,requests,1))

# Quadratic MODEL:
f2 = sp.poly1d(sp.polyfit(hours,requests,2))

# linear interpolation: generating line values
# sp.linspace(start, stop, n)
# n is the number of partition vals
# sp.linspace(0,1,3) = array([0.,0.5, 1.])

plt.scatter(hours, requests)
plt.title('WEB TRAFFIC')
plt.xlabel('Time')
plt.ylabel('requests/hr')
plt.autoscale(tight=True)
plt.grid()
xvals = sp.linspace(0, hours[-1], 1000)
plt.plot(xvals,f1(xvals), linewidth=3)
plt.plot(xvals,f2(xvals), linewidth=4, color='g')
plt.legend(['d=%d' % f1.order], loc='upper left')
plt.show()

# A function might underfit or overfit. The goal is to reduce the error, thus
# having the least amount of deviation from the curve to our data points as 
# possible. 

