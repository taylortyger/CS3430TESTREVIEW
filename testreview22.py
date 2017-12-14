# CONDITIONAL PROBABILITY
# TRAIN/TEST DATA SPLITS
from numpy import random
random.seed(0)

# Example of people age groups 20,30,40,50,60,70
# 20's are most likely to purchase

#dictionary of age groups
peopleInAgeGroup = {20:0,30:0,40:0,50:0,60:0,70:0}
purchasesInAgeGroup = {20:0,30:0,40:0,50:0,60:0,70:0}

#total people and purchases
numOfPurchases = 0
numOfPeople = 100000

for _ in xrange(numOfPeople):
	# randomly choose age group
	ageGroup = random.choice([20,30,40,50,60,70])
	# the younger you are the less likely to purchase
	purchaseProb = float(ageGroup)/100.0
	peopleInAgeGroup[ageGroup] += 1
	if(random.random() < purchaseProb):
		numOfPurchases += 1
		purchasesInAgeGroup[ageGroup] += 1

## P(AG)
def probOfAgeGroup(ageGroup):
	return float(peopleInAgeGroup[ageGroup])/numOfPeople

## P(Purchase)
def probOfPurchase():
	return float(numOfPurchases)/numOfPeople

## P(Purchase, AG)
def probOfPurchaseAndAgeGroup(ageGroup):
	return float(purchasesInAgeGroup[ageGroup])/numOfPeople

## Common sense computation of P(Purchase | AG)
def probOfPurchaseGivenAgeGroup(ageGroup):
	return float(purchasesInAgeGroup[ageGroup])/peopleInAgeGroup[ageGroup]

## P(Purchase | AG) = P(Purchase, AG)/P(AG)
def condProbOfPurchaseGivenAgeGroup(ageGroup):
	return probOfPurchaseAndAgeGroup(ageGroup)/probOfAgeGroup(ageGroup)

## we can view
##		P(P, AG)
##		P(AG)
##		P(P | AG)
##			There are two ways to do this with:
##			probOfPurchaseGivenAgeGroup
##			and condProbOfPurchaseGivenAgeGroup
## as follows:
for ag in xrange(20,80,10):
	print 'AGE GROUP: {}'.format(ag)
	print('P(Purchase,AG={age}) = {prob}'.format(age=ag,prob=probOfPurchaseAndAgeGroup(ag)))
	print('P(AG={age}) = {prob}'.format(age=ag,prob=probOfAgeGroup(ag)))
	print('P(Purchase|AG={age}) = {prob}'.format(age=ag,prob=probOfPurchaseGivenAgeGroup(ag)))
	print('P(Purchase|AG={age}) = {prob}'.format(age=ag,prob=condProbOfPurchaseGivenAgeGroup(ag)))
	

## TRAIN/TEST DATA SPLIT
## This is a simple concept
## Split your data into a training set and a testing set
## Then, train your model using the training set data,
## and test your model on the testing set data

'''
K-fold Cross Validation:
Basically, 
you split your data into K random segments (folds).
You reserve one fold for testing and the other k-1 for training
Train your model on the k-1 folds and test it on the test fold
Go to step one and repeat.
This method hopefully minimizes overfitting and underfitting.
'''

import numpy as np
import matplotlib.pyplot as plt
pageRenderTime = np.random.normal(3.0,1.0,100)
purchaseAmount = np.random.normal(50.0,10.0,100)

## Train and Test Data Split (80/20)
trainPageTime = pageRenderTime[:80]
trainPageTime = np.array(trainPageTime)
testPageTime = pageRenderTime[80:]
testPageTime = np.array(testPageTime)
trainPurchaseAmount = purchaseAmount[:80]
trainPurchaseAmount = np.array(trainPurchaseAmount)
testPurchaseAmount = purchaseAmount[80:]
testPurchaseAmount = np.array(testPurchaseAmount)

#Fit line to training data:
trained_model = np.poly1d(np.polyfit(trainPageTime, trainPurchaseAmount, 8))

t = np.linspace(0,7,100)

fig1 = plt.figure(1)
fig1.suptitle('Page Render Time vs Purchase Amount')
plt.xlabel('Page Render Time')
plt.ylabel('Purchase Amount')
plt.grid()
ax = plt.axes()
ax.set_xlim([0,7])
ax.set_ylim([0,100])
plt.scatter(pageRenderTime, purchaseAmount)
plt.scatter(trainPageTime, trainPurchaseAmount, color='r')
plt.plot(t, trained_model(t), c='g')
plt.show()

