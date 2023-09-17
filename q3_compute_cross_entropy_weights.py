"""
Lab10 - MultiLR and LogReg
3) (coding) Logistic Regression: Using the data provided (Logistic_regression_ls.csv), 
find the decision boundary (linear) using Optimization of the sigmoid function. 

Plot the following
(a) binary cross-entropy cost function and its contour plot  
(b) squared-error cost function and contour plot 
(c) data with its decision boundary. 
NOTE: You can use a 'constant' alpha value for the learning rate (instead of line search). 
"""

"""
Please note:
Obtaining of optimised weights w0,w1,w2 is implemenented in this .py file
Results take time to dispaly due to low alpha and more iterations

Part a,b,c is implemented in attached .ipynb file using the obtained 
w0,w1,w2 values from this file
"""

#Using Cross Entropy cost function 

import pandas as pd
import math 
	
# header names
header = ['x1', 'x2', 'label']
	
# define name of csv file
filename = "Logistic_regression_ls.csv"

#Read csv file Using Pandas
df = pd.read_csv(filename)
#print(df)

#list to store x1, x2 column data from csv file
x1_data = []
x2_data = []
y_data = [] #list to store y column data from csv file

#read data from csv file and store in corresponding lists
for ind in df.index: #get index value from data frame
    #print data row wise
    #print(ind, df[header[0]][ind], df[header[1]][ind], df[header[2]][ind]) 
    x1_data.append(df[header[0]][ind])
    x2_data.append(df[header[1]][ind])
    y_data.append(df[header[2]][ind])
print("\n")

# m = number of sample points available in ground truth data
#m = len(x_data) 
m = len(x1_data) 
#print(m)



##################################################### 2
"""
Setting up functions for finding grad J in Gradiend Descent Algorithm 
"""
#FUNCTION 1  
#to compute hypothesis function i.e. sigmoid function at (w_0,w_1,w_2,w_2,x1,x2) 
def h_for_w(w_0, w_1, w_2, x1, x2, i=0):
    
    z = w_0 + (w_1 *x1) + (w_2 * x2)
    
    #compute result for sigmoid function for input data 
    sigma = 1 / (1 + (math.exp(-z)))
    
    #return values of w_0, w_1, w_2, sigma
    if i == 1:
        return w_0, w_1, w_2, sigma  
    
    #return value of sigma
    else:
        return sigma


#FUNCTION 2
#to compute first order partial derivative of cross entropy loss function J 
#wrt w_0 at w_0, w_1, w_2, x1, x2 
def J_grad_w1(w_0,w_1, w_2):
    sum_0 = 0
    
    for i in range(0,m):
        sum_0 = sum_0 + (h_for_w(w_0, w_1, w_2, x1_data[i], x2_data[i]) - y_data[i])
    
    sum_0 = sum_0 / m             
    return sum_0

#FUNCTION 3 
#to compute first order partial derivative of cross entropy loss function J 
#wrt w_1 at w_0, w_1, w_2, x1, x2 def J_grad_w2(w_0,w_1, w_2):
def J_grad_w2(w_0,w_1, w_2):
    sum_1 = 0
    
    for i in range(0,m):
        sum_1 = sum_1 + ((h_for_w(w_0, w_1, w_2, x1_data[i], x2_data[i]) - y_data[i]) * x1_data[i])
    
    sum_1 = sum_1 / m             
    return sum_1

#FUNCTION 4 
#to compute first order partial derivative of cross entropy loss function J 
#wrt w_2 at w_0, w_1, w_2, x1, x2 def J_grad_w3(w_0,w_1, w_2):
def J_grad_w3(w_0,w_1, w_2):
    sum_2 = 0
    
    for i in range(0,m):
        sum_2 = sum_2 + ((h_for_w(w_0, w_1, w_2, x1_data[i], x2_data[i]) - y_data[i]) * x2_data[i])
    
    sum_2 = sum_2 / m             
    return sum_2
##################################################### 2


##################################################### 4
#FUNCTION 5
#get hypothesis = sigmoid value for computing Cost function 
def get_sigmoid_value(w_0, w_1, w_2, x1, x2):
    
    #h(w) = 1 / (1 + e^(-w'x))
    z = w_0 + (w_1 * x1) + (w_2 * x2)
    sigmoid = 1.000 / (1.000 + math.exp(-z))
    #print("sigmoid value", sigmoid)
    return sigmoid
    

#FUNCTION 6
#function defined but not used 
#function to compute value of cost function for given values of w_0, w_1, w_2
def Compute_J_1(alpha, w_0, w_1, w_2, i=0):
       
    sum_0 = 0

    for j in range(0,m):
        r = get_sigmoid_value(w_0, w_1, w_2, x1_data[j], x2_data[j])
        s =  math.log(r) 
        #print(s)
        
        # in absence of 0.0001, the interpreter throws 
        # math domain error as it cannot handle log(0)
        # hence a small value 0.0001 is added for computation purpose 
        s_minus_1 = math.log(1.00-r+0.0001)  
        
        sum_0 = sum_0 + (   (-1 * y_data[j] * s)  -  (   (1 - y_data[j]) * s_minus_1)  )                        
        
    sum_0 = sum_0 / m  

    #return value of w_0, w_1, w_2, J 
    if i == 1:
        return w_0, w_1, w_2, sum_0
        
    #return value of J only 
    else:    
        return sum_0
##################################################### 4





"""
START OF EXECUTION OF FULL ALGORITHM ---------------
"""
############################## 3   
#START OF ALGORITHM : 
#LOGISTIC REGRESSION USING STEEPEST GRADIENT DESCENT USING CONSTANT ALPHA      

#Hypothesis equation for univariate linear regression
#h(w) = sigma(z) = 1 / (1 + (math.exp(-z)))
#where z = w_0 + (w_1 *x1) + (w_2 * x2)
    
#w_0 = bias or weight 
#w_1,w_2 = weight

#cost function for logistic regression
#J(w_0, w_1) = (1/m) sum(1 to m) [ (- y log(h(w))) - ((1-y) log(1 - h(w))) ]
#x1_data = ground truth x1 data
#x2_data = ground truth x2 data
#y_data = ground truth y data
#m = number of available sample points


#STEP 1 : Choose initial values
#initial guess for weights
w_0 = 5 #initial weight (bias) value  
w_1 = 1  #initial weight value
w_2 = 1  #initial weight value

stop_value = 0.0001 #define limit of search or convergence for norm of gradient
k = 1               #iteration counter 
iteration = 10000   #set no. of iterations

#flag = 0 means no critical point / weight can be found 
flag = 0 
alpha =0.0001 #fixed alpha 

while(k < iteration): 
    #STEP 2:compute first order partial derivate of J wrt w_0, w_1,w_2  at every w_0, w_1,w_2 values 
    j_grad_w1 = J_grad_w1(w_0,w_1, w_2) #PD wrt w_0
    j_grad_w2 = J_grad_w2(w_0,w_1, w_2) #PD wrt w_1
    j_grad_w3 = J_grad_w3(w_0,w_1, w_2) #PD wrt w_2
    
    #STEP 3: compute next value of weights from step 2
    w1_next = w_0 - alpha * j_grad_w1 
    w2_next = w_1 - alpha * j_grad_w2 
    w3_next = w_2 - alpha * j_grad_w3
            
    #compute first order partial derivate of J at w1_next, w2_next, w3_next wrt w1 w2 , w3
    j_grad_wnext_1 = J_grad_w1(w1_next, w2_next, w3_next)  #PD wrt w1
    j_grad_wnext_2 = J_grad_w2(w1_next, w2_next, w3_next)  #PD wrt w2
    j_grad_wnext_3 = J_grad_w3(w1_next, w2_next, w3_next)  #PD wrt w3
    
    #start computing norm of grad J vector
    j_grad_wnext_1 = j_grad_wnext_1 * j_grad_wnext_1
    j_grad_wnext_2 = j_grad_wnext_2 * j_grad_wnext_2
    j_grad_wnext_3 = j_grad_wnext_3 * j_grad_wnext_3
    
    #check if norm of grad J is within stop_value
    if math.sqrt(j_grad_wnext_1 + j_grad_wnext_2 + j_grad_wnext_3) < stop_value:
    #if abs(Compute_J_1(alpha, w_0, w_1, w_2) - Compute_J_1(alpha, w1_next, w2_next, w3_next)) < stop_value:
        flag = 1 #norm of grad J is within stop_value
        break    #break from loop as critical point is found
    else: #updtae w1,w2,w3  and proceed to next iteration
        w_0 = w1_next
        w_1 = w2_next
        w_2 = w3_next
        
        if k % 1000 == 0:#these values are used for plotting on contour of J(w1, w2,w3)
            print("Updated value of w0:", w_0)
            print("Updated value of w1:", w_1)
            print("Updated value of w2:", w_2)
            print("\n")
        
        k = k+1

print("\n")        
if flag == 1:  #critical value / weights found  
    print("The bias value w_0 is:", w1_next)
    print("The slope value w_1 is:", w2_next)
    print("The slope value w_2 is:", w3_next)
    print("\nBest fit line for given dataset using Logistic Regression is: ")
    print("y = " + str(w1_next) + " + " + str(w2_next) + " * x1" + " + " + str(w3_next) + " * x2")
if flag == 0:  #critical value / weights could not be found  
    print("Start with some other initial guess")
############################## 3   


"""
Result:

Updated value of w0: 4.9500162438886575
Updated value of w1: 0.8980384866291381
Updated value of w2: 0.9023169840435693


Updated value of w0: 4.900038307833832
Updated value of w1: 0.7960863755703399
Updated value of w2: 0.8046423825406647


Updated value of w0: 4.850068744087361
Updated value of w1: 0.6941484752510411
Updated value of w2: 0.7069806210690934


Updated value of w0: 4.80011144409518
Updated value of w1: 0.5922325824666059
Updated value of w2: 0.6093389475166738


Updated value of w0: 4.750172517285374
Updated value of w1: 0.49035201548239726
Updated value of w2: 0.5117298805142232


Updated value of w0: 4.700261993753909
Updated value of w1: 0.38853145195445055
Updated value of w2: 0.41417690338872765


Updated value of w0: 4.650397551843447
Updated value of w1: 0.28682227173501285
Updated value of w2: 0.3167295325807219


Updated value of w0: 4.6006138622689035
Updated value of w1: 0.18534759092766526
Updated value of w2: 0.2195077300578577


Updated value of w0: 4.550988852951739
Updated value of w1: 0.08444428556927047
Updated value of w2: 0.1228422045263151




Start with some other initial guess
"""




























"""
#PRINTING x column data 
print(x_data)#prints entire x xolumn only 
print("\n")

print(type(x_data))#prints data type as list

print(x_data[0])#prints first element of x column

print(type(x_data[0])) #prints type of ele in x column <class 'numpy.float64'>

print(len(x_data))#200

#x_data is a list of elements of type numpy.float64

for ele in x_data:
    print(ele)


#PRINTING y column data 
print(y_data)#prints entire y xolumn only 
print("\n")

print(type(y_data))#prints data type as list

print(y_data[0])#prints first element of y column

print(type(y_data[0])) #prints type of ele in y column <class 'numpy.float64'>

print(len(y_data))#200

#y_data is a list of elements of type numpy.float64


print("\n")
##being np , elements can be added using np.add(a,b)
print(np.add(x_data, y_data))
print(len(np.add(x_data, y_data)))
"""








