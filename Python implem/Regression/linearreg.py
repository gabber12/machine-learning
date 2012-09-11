import numpy as np
import matplotlib.pyplot as mp

def load(filename, delim=","):

    dstring = file( filename, 'r')
    x = np.loadtxt(dstring,delimiter=delim)
    y = np.array([x[:,-1]]).T
    x = x[:,:-1]
    return x, y

def plotData(X,Y):
    
    mp.scatter(X,Y,marker="x",color="y")

def costfunction(X,Y,theta,verbose=False):
    """init_theta --is a column matrix.
        X --each row consist of one training example.
    """
    m = np.shape(X)[0]
    
    X = np.hstack((np.ones((m,1)),X))

    h = np.dot(X,theta)
    grad = (h-Y) * X
    J =  np.sum((h-Y) ** 2)/(2*m)
    if verbose :print "J>> ",J
    return J, grad

def gradientDescent(X, Y, init_theta, alpha, iteration, verbose=False):

    m = np.shape(X)[0]
    while iteration:
        grad_sum = np.sum(costfunction(X,Y,init_theta,verbose)[1],0)
        init_theta = init_theta - (alpha/m) * np.array([grad_sum]).T
        
        iteration -= 1

    return init_theta.T

def predict(X, Theta):

    """X is a row matrix 
    theta is given by gradient descent a row matrix"""

    X=np.hstack((np.ones((1,1)),X))
    return np.dot(X, Theta.T)

def lineareg(filename, Delim=",",plot=False):

    """file should contain first the column of x and then y
        values will be taken to delimit by ',' unless specified othewise"""

    x,y = load(filename, Delim)
    m = np.shape(x)[0]
    n = np.shape(x)[1]+1
    if plot:plotData(x,y)
    init_theta = np.zeros((n, 1))
    Theta = gradientDescent(x, y, init_theta, 0.01, 1500)
    if plot:
        mp.plot(np.dot( np.hstack((np.ones((m,1)),x)), Theta.T),y,"ro")
        mp.plot(np.dot( np.hstack((np.ones((25,1)),np.array([list(range(0,25))]).T )),Theta.T),"g")
    return Theta


###Test for data File Bivariate
Theta = lineareg("data",plot=True)
print "Theta Value>> ", Theta
print "Predicted Value >> ",predict(np.array([[3.5000]]),Theta)   
##################
mp.show()

###Test for D file
Theta = lineareg("d")
print "Theta value>> ", Theta

print "Predicted value for 6 ,6 >> ",predict(np.array([[6,6]]),Theta)
