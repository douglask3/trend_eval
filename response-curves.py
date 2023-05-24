import numpy as np
import matplotlib.pyplot as plt
'''
def hinge_1(x0, y0, a, b):

    """ fits a hinge curve function
    x -- numpy array 
     -- hinge point
    """
    
    if np.all(x1 > x0):
        y = a*x1 + b
   
    else:
        y = y0
    
    return y
   

#calling the function

x1 = np.round(np.linspace(0, 1, num=10), decimals=1)

hin1 = hinge_1(0.5, 0.7, 2, 3)

print(hin1)
 
#notes
 
# np.linspace is a function from NumPy that generates a sequence of evenly spaced numbers within a specified range
'''
'''
def hinge_2(x, a1, b1, a2, b2):

    
    x0 = (b2-b1)/(a1-a2)
    print("x0 = ", x0)
    y=[]
    
    for xi in x:
        if xi > x0:
            yi = a2*xi + b2
   
        else:
            yi = a1*xi + b1
    
        y.append(yi)
    return y

#calling the function

x = np.linspace(0, 1, num=100)
print(x)
hin2 = hinge_2(x, -1, 1, 2, 0.5)

print("y = ", hin2)
plt.plot(x, hin2)
plt.show()
'''

def exp(x, p):
    
    y = x**p
    
    return y
 

#calling the function

x = np.linspace(0, 1, num=100)
e = exp(x,2)

print(e)
plt.plot(x, e)
plt.show()




