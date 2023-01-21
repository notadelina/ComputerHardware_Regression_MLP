# Neural Network Regression for a Computer Hardware database
 

DataBase : 'https://archive.ics.uci.edu/ml/datasets/Computer+Hardware'

## Description

Implemented a MLP-based regression
model for predicting computer hardware performance, using Python and scikit-learn library, which involved
preprocessing and feature engineering



### Data size

209x9

### Input example

Input : ‘apollo,dn320,400,1000,3000,0,1,2,38,23’, where:

        *	‘apollo’ is the name of the manufacturer
        *	‘dn320’ is the name of the computer model
        *	‘400’ is the time cycle of the computer in nanoseconds.
        *	‘1000’ is the minimum of memory in kB.
        *	‘3000’ is the maximum of memory in kB.
        *	‘0’ is the cache memory in kB.
        *	‘1’ is the minimum number of channels.
        *	‘2’ is the maximum number of channels.
        *	’38’ is the relative performance that was published.
        *	’23’ is the relative performance estimated in the original article.


### Data splitting

 75% for training 
 25% for testing

### Parameters 
  * One or two hidden layers.
  * Number of neuron in the hidden layers.
  * Learning rate 0.1 or 0.01
