import numpy as np
import pandas as pd

df=pd.read_csv(r'C:\Users\benti\Group Project 2\Data\Housing Data.csv')

class MultipleRegression():
    def __init__(self):
        self.train_df=df.sample(frac=0.8,random_state=419)
        self.prediction_df=df.drop(self.train_df.index)

    def main(self):   
            X,Y,theta=self.Array_Gen(self.train_df)
            LR =0.002
            I = 5000
            theta_col_vec,cost_value_history,theta_value_history = self.Grad_Des(X,Y,theta,LR,I)
            for i in range(len(theta_col_vec)):
                print('Theta_{}'.format(i),theta_col_vec[i][0])


    def Array_Gen(self,df):
        test_array=pd.DataFrame.to_numpy(df)
        W=(np.shape(test_array)[1])
        Y_col_vec=test_array[:, [0]]
        theta_col_vec=np.ones((W,1))
        X_Array=test_array[:,1:W]
        X_1_Matrix = np.c_[np.ones((len(test_array),1)),X_Array]
        
        return (X_1_Matrix,Y_col_vec,theta_col_vec)

    def  Cost_Calc(self,x,y,theta):
        
        n = len(y)
        hypothesis_function = x.dot(theta)
        cost_function = (1/(2*n)) * (hypothesis_function-y).T.dot(hypothesis_function-y)
        return cost_function

    def Grad_Des(self,x,y,theta,LR,i):
        n = len(y)
        theta_value_history = np.zeros((i,2))
        cost_value_history = np.zeros(i)
        for j in range(i):
            hypothesis_func_col_vec = x.dot(theta)
            theta = theta -LR*((1/n))*( x.T.dot((hypothesis_func_col_vec - y)))
            theta_value_history[j] =theta.T
            cost_value_history[j]  = self.Cost_Calc(x,y,theta)
            
        return theta, cost_value_history, theta_value_history

    def Make_Predictions(self,theta):
        X,Y,Z=MultipleRegression.Array_Gen(self.prediction_df)
        return(X.dot(theta))
