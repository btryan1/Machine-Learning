import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from matplotlib import pyplot as plt
from Super import HC as HCSuper

df = pd.read_csv(r'C:\Users\benti\Group Project 2\HC\Data\HC test14.csv')

class Complete_Linkage(HCSuper):
    def __init__(self,df):
        super().__init__(df)

    def Find_Min_Distance(self,M,I,J,choice,minimum):
            if choice ==0:
                update=[]
                update2=[]
                update3=[]
                it=0
                for i in range(np.shape(M)[0]):
                    if i!=I and i!=J:
                        maximum=max(M[J][i],M[I][i])
                        update.append(maximum)
                        if maximum==M[J][i]:
                            update3.append(M[I][i])
                        else:
                            update3.append(M[J][i])
                for i in range(np.shape(M)[0]-1):
                    it+=1
                    for j in range(it,np.shape(M)[0]):
                        if M[i][j] not in update and M[i][j] not in update3 and M[i][j] != minimum:
                            update2.append(M[i][j])   
                        else:
                            continue 
                UPDATE=update+update2 
                return UPDATE
            else:
                y=0
                update=[M[0][1],M[0][2],M[1][2]]
                new_update = np.delete(update, np.where(update==minimum))
                return max(new_update)    

    def flatten(self,l):
        out = []
        for item in l:
            if isinstance(item, (list, tuple)):
                out.extend(HC.flatten(item))
            else:
                out.append(item)
        return out

    def run2(self):
        array=HC.Array_To_List_Col_Vec(self.df)
        W=len(array)
        branch_lengths=[]
        Feature_List=[i for i in range(W)]
        clusters_history=[]
        for i in range(W):
            if i<=0:
                Cluster_Forrest=HC.Cluster_Forrest(W)
                M=HC.Distance_Matrix(array)
                I,J,minimum=HC.Min_Distance_Calc(M)
                branch_lengths.append(minimum)
                clusters_new=HC.Return_Cluster(I,J,0,Feature_List,0)
                for i in range(len(clusters_new)):
                    clusters_history.append(clusters_new[i])
                clusters,Cluster_Forrest=HC.Z_clusters(clusters_new,Cluster_Forrest)
                cluster=clusters_history[0]
                Z=HC.Z_matrix(cluster,cluster,minimum)
    
            else:
                if len(clusters_history)==2:
                    clusters2=[]
                    clusters2.append((clusters_history[0],clusters_history[1]))
                    clusters2=HC.flatten(clusters2)
                    minimum=HC.Find_Min_Distance(M,I,J,1,minimum)
                    Z_new=HC.Z_matrix(clusters,clusters2,minimum)
                    Z=np.vstack([Z,Z_new])
                    return(Z)
                UPDATE=HC.Find_Min_Distance(M,I,J,0,minimum)
                M=HC.Update_Distance_Matrix2(M,UPDATE)
                I,J,minimum=HC.Min_Distance_Calc(M)
                branch_lengths.append(minimum)
                clusters_new=HC.Return_Cluster(I,J,clusters_history,Feature_List,1)
                clusters_history=[]
                for i in range(len(clusters_new)):
                    clusters_history.append(clusters_new[i])
                clusters_new=HC.Return_Cluster(I,J,clusters,0,1)
                cluster1=clusters_new[0]
                cluster2=clusters_history[0]
                cluster2 = HC.flatten(cluster2)    
                Z_new=HC.Z_matrix(cluster1,cluster2,minimum)   
                Z=np.vstack([Z,Z_new])
                clusters,Cluster_Forrest=HC.Z_clusters(clusters_new,Cluster_Forrest)
                print(np.shape(M))
        
HC=Complete_Linkage(df)
Z= HC.run2()
fig = plt.figure(figsize=(25, 10))
dn=dendrogram(Z)
plt.title('Our Complete Linkage Dendrogram')
plt.show()
