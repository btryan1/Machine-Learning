import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from matplotlib import pyplot as plt


class HC:
    def __init__(self,df):
        self.df=df

    def Array_To_List_Col_Vec(self,df):
        test_array=pd.DataFrame.to_numpy(df)
        W=(np.shape(test_array)[1])
        col_vec_array=[]
        for i in range(W):
            col_vec_array.append(test_array[:,[i]])
        
        return(col_vec_array)
    
    def Distance_Matrix(self,array):
        M=np.zeros((len(array),len(array)),dtype=float)
        for i in range(len(array)):
            for j in range(len(array)):
                if i!=j :
                    X_i=array[i]
                    X_j=array[j]
                    distance=float(np.sqrt((X_i-X_j).T.dot(X_i-X_j)))
                    M[i][j]=distance
                else:
                    M[i][j]=250
        return(M)
    
    
        
    def Min_Distance_Calc(self,M):
        min=np.min(M)
        min_array=[]
   
        for i in range(np.shape(M)[0]):
            for j in range(np.shape(M)[0]):
                if min==M[i][j]:
                    return i,j,min
                else:
                    continue
       
        return min_array


    def Feature_Alphabet(self):
        Feature_Alphabet_List=list()
        for i in range(ord('A'), ord('Z') + 1):
            Feature_Alphabet_List.append(chr(i))
        return Feature_Alphabet_List
    
    def Cluster_Forrest(self,W):
        Cluster_Forrest=[]
        W=W-1
        for i  in range(10000):
            W=W+1
            Cluster_Forrest.append(W)
        return Cluster_Forrest

    def Return_Cluster(self,I,J,clusters,Feature_List,choice):
        clusters_new=[]
        if choice == 1:
            feature_1=clusters[I]
            feature_2=clusters[J]
            if I==0 or J==0:
                clusters_new.append((feature_1,feature_2))
                for i in range(len(clusters)):
                    if i != I and i !=J and i!=0:
                        clusters_new.append(clusters[i])
            else:
                clusters_new.append((feature_1,feature_2))
                clusters_new.append(clusters[0])
                for i in range(len(clusters)):
                    if i != I and i !=J and i!=0:
                        clusters_new.append(clusters[i])
            return clusters_new
        else:
            feature_1=Feature_List[I]
            feature_2=Feature_List[J]
            clusters_new.append((feature_1,feature_2))
            for i in range(len(Feature_List)):
                if i != I and i !=J:
                    features=Feature_List[i]
                    clusters_new.append(features)
            return clusters_new

    def Update_Distance_Matrix2(self,M,update):
        new_shape=len(M)-1
        M_update=np.zeros((new_shape,new_shape),dtype=float)
        y=0
        for i in range(new_shape-1):
            y+=1
            it=0
            for j in range (y,new_shape):
                if i!=j:
                    M_update[i][j]=update[it]
                    it+=1
                else:
                    continue
            del update[:it]
        for i in range(new_shape):
            for j in range(new_shape):
                if i==j:
                    M_update[i][j]=250
                else:
                    continue
        
        M_update= M_update + M_update.T -np.diag(np.diag(M_update))
        return M_update


    def Z_clusters(self,clusters,Cluster_Forrest):
         del clusters[:0]
         clusters[0]=Cluster_Forrest[0]
         del Cluster_Forrest[0]
         return clusters,Cluster_Forrest

    def Z_matrix(self,cluster1,cluster2,minimum):
        Z=[]
        for i in range(4):
                    if i<=0:
                        Z.append(min(cluster1[i],cluster1[i+1]))
                    elif i==1:
                        Z.append(max(cluster1[i],cluster1[i-1]))
                    elif i == 2:
                        Z.append(minimum)
                    else:
                        Z.append(len(cluster2))
        return Z

    def flatten(self,cluster):
        out = []
        for item in cluster:
            if isinstance(item, (list, tuple)):
                out.extend(HC.flatten(item))
            else:
                out.append(item)
        return out

