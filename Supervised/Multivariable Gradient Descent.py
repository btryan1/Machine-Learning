#"This file contains everything necessary to calculate an arbitary amount of theta coefficients using gradient descent."
#"We should also note that this file is working in progress and there are two versions of it, this one which contains all the functions we have programmed i.e. some functions calculate the same thing however use different modules, notation to derive the answer"
#"On top of this, my aim is that this becomes a class method once it has been finalized and working as well as to produce plots of everything hence there are history arrays which hopefully will help us do this"

#"Imports all the usual modules"
import numpy as np
import pandas as pd

#"Imports the prediciton data set we used for simple linear regression (for testing purposes and then we shall move up in the number of inputted features"
prediction_df=pd.read_csv(r'C:\Users\benti\Group Project 2\Data\Prediction_DF.csv')

#"extracts these data frames into respective lists"
prediction_x,prediction_y=prediction_df['X'].values.tolist(),prediction_df['Y'].values.tolist()


#"Since this file contains multiple functions that do the same thing, we have defined two main call function where we interchange our fucntions called to see the differences in output and whether one would be preferred over the other"

#"This function generates the arrays needed by the gradient descent function in order for it to be mathematically sound"

def Array_Gen(df):

    test_array=pd.DataFrame.to_numpy(df)
    W=(np.shape(test_array)[1])
    Y_col_vec=test_array[:, [0]]
    theta_col_vec=np.ones((W,1))
    X_Array=test_array[:,1:W]
    X_1_Matrix = np.c_[np.ones((len(test_array),1)),X_Array]
    
    return (X_1_Matrix,Y_col_vec,theta_col_vec)

#"Our first method of calculating the cost function (in our case we have selected the MSE) at the given theta."
def  Cost_Calc(x,y,theta):
    n = len(y)
    hypothesis_function = np.dot(x,theta)
    cost_function = (1/(2*n)) * np.sum(np.square(hypothesis_function-y))
    return cost_function

#"This is our second method for calculating the cost function, the key difference between the previous one is that we utilise different defintions in our calculations which results in the output of a 1x1 array instead of a constant which is given by above"
def  Cost_Calc2(x,y,theta):
    
    n = len(y)
    hypothesis_function = x.dot(theta)
    cost_function = (1/(2*n)) * (hypothesis_function-y).T.dot(hypothesis_function-y)
    return cost_function

#"Our method for the Gradient Descent algorithm, it takes in the generated arrays as arguements as well as the learning rate and the desired number of iterations"
def Grad_Des(x,y,theta,LR,i):
    #"The length of our array will be the number of data points"
    n = len(y)
    #"sets our inital history arrays to the correct shape with zeros as their inital values"
    theta_value_history = np.zeros((i,2))
    cost_value_history = np.zeros(i)
    #"Enter our only for loop which calculates new theta column vector using the learning rate and the transpose dot product between the difference of the hyp func and the true y col vec/"
    #"Also updates our history arrays with current value for each iteration"
    for j in range(i):
        hypothesis_func_col_vec = x.dot(theta)
        theta = theta -LR*((1/n))*( x.T.dot((hypothesis_func_col_vec - y)))
        theta_value_history[j] =theta.T
        cost_value_history[j]  = Cost_Calc2(x,y,theta)
        #"returns our theta and history arrays"
    return theta, cost_value_history, theta_value_history

#"Second main call, this is also for testing purposes utilizing different methods to get the to same answer"

def main():   
    X,Y,theta=Array_Gen(prediction_df)
    LR =0.000002
    I = 1000000


    theta,cost_value_history,theta_value_history = Grad_Des(X,Y,theta,LR,I)

    print(cost_value_history)

main()
