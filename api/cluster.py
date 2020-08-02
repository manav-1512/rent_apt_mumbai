# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd

rcParams['xtick.color'] = '#b0e0e6'
rcParams['ytick.color'] = '#b0e0e6'
rcParams['axes.labelcolor'] = '#b0e0e6'
rcParams['axes.edgecolor'] = '#b0e0e6'
rcParams['axes.facecolor'] = 'rgba(0.118,0.118,0.184)'
rcParams['legend.facecolor'] = 'rgba(0.118,0.118,0.184)'
rcParams['figure.facecolor'] = 'rgba(0.118,0.118,0.184)'
rcParams['savefig.facecolor'] = 'rgba(0.118,0.118,0.184)'
rcParams['text.color'] = '#b0e0e6'
# print(rcParams.keys())

# Importing the dataset
dataset = pd.read_csv('api/static/data/prop_data_clean1.csv')
X = dataset.iloc[:, [11,13,16]].values

def process(X):
    X = X.reshape(-1,2)
    return X

def process2(X):
    return X.reshape(-1,2)

a = np.where(X[:, 0]==X[:,1])
for i in a:
    X[i,1]=0
x = np.where(X[:,0]>30)
for i in x:
    X[i,0],X[i,1] = X[i,1],X[i,0]
x = np.where(X[:,0]>21)
for i in x:
    X[i,0]-=10    

from sklearn.impute import SimpleImputer
missingvalues0 = SimpleImputer(missing_values = 1, strategy = 'constant', verbose = 0, fill_value=0)
X[:, 0:1] = missingvalues0.fit_transform(X[:, 0:1])
missingvalues0 = SimpleImputer(missing_values = 7, strategy = 'constant', verbose = 0, fill_value=0)
X[:, 1:2] = missingvalues0.fit_transform(X[:, 1:2])
missingvalues = SimpleImputer(missing_values = 0, strategy = 'constant', verbose = 0, fill_value=np.nan)
X[:, 0:2] = missingvalues.fit_transform(X[:, 0:2])
missingvaluesm = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose = 0)
X[:, 0:2] = missingvaluesm.fit_transform(X[:, 0:2])




def filter(param1,param2,param3):
    #Applying k-means to the dataset
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters= 3, init='k-means++', max_iter=300, n_init=1, random_state=0)
    y_kmeans = kmeans.fit_predict(X[:, :])


    def clust1(n):
        place_cluster = [1,0,2] 
        if n in [0,1,2]:
            return np.where(y_kmeans == place_cluster[n])
        elif n == 3:
            return np.where(y_kmeans!= None)
        

    place = dataset.iloc[:, [0,2,1,16]].values
    arri = clust1(param1)
    c = np.count_nonzero(arri)
    arri = arri[0]
    if arri[0] == 0:
        c+=1       
    place = place[arri, :]
    place = place.reshape(c,4)

    from sklearn.impute import SimpleImputer
    missingvalues = SimpleImputer(missing_values = np.nan, strategy = 'median', verbose = 0)
    place[:, 3:4] = missingvalues.fit_transform(place[:, 3:4]) 

    kmeans2 = KMeans(n_clusters= 3, init='k-means++', max_iter=300, n_init=1, random_state=0)
    y_kmeans2 = kmeans2.fit_predict(place[:,-1:])

    def clust2(n,m):
        final_cluster = [[2,0,1],[0,1,2],[0,2,1],[1,0,2]]
        if m==3:
            return np.where(y_kmeans2 != None)
        else:
            return np.where(y_kmeans2 == final_cluster[n][m]) 

    arri2 = clust2(param1, param2)        
    c2 = np.count_nonzero(arri2)
    # print(np.amin(place[arri2, -1]))
    # print(np.amax(place[arri2, -1]))
    arri2 = arri2[0]
    if arri2[0]==0:
        c2+=1

    place = np.array(place, dtype = 'int')

    data = dataset.iloc[:, [5,6,7,8,12,15,17,9]].values 

    data = data[arri,:]
    place1 = place[arri2, :]

    data = np.append(place1[:, :], data[arri2, :], axis = 1) 

    data = data.reshape(c2,12)

    df = pd.DataFrame(data)


    # print(dataset.columns)
    ax = [0,2,1,16,5,6,7,8,12,15,17,9]
    col = []
    for i in ax:
        col.append(dataset.columns[i])
    df.columns = (col)
    if param3 != 0: 
        df.query('bedroom_num == '+str(param3), inplace = True)
    s = 'api/static/data/'+str(param1)+','+str(param2)+','+str(param3)+'.csv'
    df.to_csv(s, index = False)    

def map(param1,param2,map):


    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters= 3, init='k-means++', max_iter=300, n_init=1, random_state=0)
    y_kmeans = kmeans.fit_predict(X[:, :])


    def clust1(n):
        place_cluster = [1,0,2] 
        if n in [0,1,2]:
            return np.where(y_kmeans == place_cluster[n])
        elif n == 3:
            return np.where(y_kmeans)

    place = dataset.iloc[:, [0,2,1,16]].values
    arri = clust1(param1)
    c = np.count_nonzero(arri)
    arri = arri[0]
    if arri[0] == 0:
        c+=1       
    # place = place[arri, :]
    place = place.reshape(34315,4)

    from sklearn.impute import SimpleImputer
    missingvalues = SimpleImputer(missing_values = np.nan, strategy = 'median', verbose = 0)
    place[:, 3:4] = missingvalues.fit_transform(place[:, 3:4]) 

    kmeans2 = KMeans(n_clusters= 3, init='k-means++', max_iter=300, n_init=1, random_state=0)
    y_kmeans2 = kmeans2.fit_predict(place[:,-1:])

    def clust2(n,m):
        final_cluster = [[2,0,1],[0,1,2],[0,2,1],[1,0,2]]
        if m==3:
            return np.where(y_kmeans2)
        else:
            return np.where(y_kmeans2 == final_cluster[n][m]) 

    arri2 = clust2(param1,param2)
    arri2 = arri2[0]
    X2 = X[arri, :]
    if map == 1:
        y_kmeans = kmeans.fit_predict(X2)
        # Visualising the clusters
        fig = plt.figure()
        ax = fig.add_subplot('111',projection = '3d')
        if param1 == 3:
            ax.scatter(X2[y_kmeans == 0, 0], X2[y_kmeans == 0, 1], X2[y_kmeans == 0, -1], s = 100, c = 'blue', label = 'Suburbs')
            ax.scatter(X2[y_kmeans == 2, 0], X2[y_kmeans == 2, 1], X2[y_kmeans == 2, -1], s = 100, c = 'green', label = 'Downtown')
            ax.scatter(X2[y_kmeans == 1, 0], X2[y_kmeans == 1, 1], X2[y_kmeans == 1, -1], s = 100, c = 'red', label = 'Posh')
        elif param1 == 0:
            ax.scatter(X2[y_kmeans == 2, 0], X2[y_kmeans == 2, 1], X2[y_kmeans == 2, -1], s = 100, c = 'blue', label = 'Economical')
            ax.scatter(X2[y_kmeans == 0, 0], X2[y_kmeans == 0, 1], X2[y_kmeans == 0, -1], s = 100, c = 'green', label = 'Comfortable')
            ax.scatter(X2[y_kmeans == 1, 0], X2[y_kmeans == 1, 1], X2[y_kmeans == 1, -1], s = 100, c = 'red', label = 'Luxurious')
            ax.title.set_text('Suburbs')
        elif param1 == 1:
            ax.scatter(X2[y_kmeans == 0, 0], X2[y_kmeans == 0, 1], X2[y_kmeans == 0, -1], s = 100, c = 'blue', label = 'Economical')
            ax.scatter(X2[y_kmeans == 1, 0], X2[y_kmeans == 1, 1], X2[y_kmeans == 1, -1], s = 100, c = 'green', label = 'Comfortable')
            ax.scatter(X2[y_kmeans == 2, 0], X2[y_kmeans == 2, 1], X2[y_kmeans == 2, -1], s = 100, c = 'red', label = 'Luxurious')
            ax.title.set_text('Downtown')
        elif param1 == 2:    
            ax.scatter(X2[y_kmeans == 0, 0], X2[y_kmeans == 0, 1], X2[y_kmeans == 0, -1], s = 100, c = 'blue', label = 'Economical')
            ax.scatter(X2[y_kmeans == 2, 0], X2[y_kmeans == 2, 1], X2[y_kmeans == 2, -1], s = 100, c = 'green', label = 'Comfortable')
            ax.scatter(X2[y_kmeans == 1, 0], X2[y_kmeans == 1, 1], X2[y_kmeans == 1, -1], s = 100, c = 'red', label = 'Luxurious')
            ax.title.set_text('Posh')
        # ax.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], X[y_kmeans == 3, -1], s = 100, c = 'cyan', label = 'Cluster 4')
        # ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:,-1], s = 100, c = 'yellow', label = 'Centroids')
        ax.legend()
        # ax.tick_params(axis = 'x', color = (0.69,0.878,0.902,1))
        # ax.tick_params(axis = 'y', color = (0.69,0.878,0.902,1))
        # ax.tick_params(axis = 'z', color = (0.69,0.878,0.902,1))
        # # ax.set_facecolor('#1e1e2f')
        # ax.w_xaxis.set_pane_color((0.69,0.878,0.902,1))
        # ax.xaxis.label.set_color((0.69,0.878,0.902,1))
        # ax.w_yaxis.set_pane_color((0.69,0.878,0.902,1))
        # ax.w_zaxis.set_pane_color((0.69,0.878,0.902,1))
        ax.set_xlabel('Latitude')
        ax.set_ylabel('Longitude')
        ax.set_zlabel('Price')
        # plt.show()

    elif map ==2:
        # Visualising the clusters
        fig = plt.figure()
        ax = fig.add_subplot('111')
        if param2 == 3:
            ax.scatter(place[y_kmeans2 == 1, 0], place[y_kmeans2 == 1, -1],  c = 'blue', label = 'Economical')
            ax.scatter(place[y_kmeans2 == 2, 0], place[y_kmeans2 == 2, -1],  c = 'green', label = 'Comfortable')
            ax.scatter(place[y_kmeans2 == 0, 0], place[y_kmeans2 == 0, -1],  c = 'red', label = 'Luxurious')
            ax.legend()
        elif param2 == 0:
            ax.scatter(place[y_kmeans2 == 1, 0], place[y_kmeans2 == 1, -1],  c = 'blue')
            ax.title.set_text('Economical')
        elif param2 == 1:
            ax.scatter(place[y_kmeans2 == 0, 0], place[y_kmeans2 == 0, -1],  c = 'red')
            ax.title.set_text('Comfortable')
        elif param2 == 2:
            ax.scatter(place[y_kmeans2 == 2, 0], place[y_kmeans2 == 2, -1],  c = 'green')
            ax.title.set_text('Luxurious')
        # ax.scatter(place[y_kmeans2 == 3, 1], place[y_kmeans2 == 3, 2], place[y_kmeans2 == 3, -1],  c = 'cyan', label = 'Cluster 4')
        # ax.scatter(kmeans2.cluster_centers_[:, 0], kmeans2.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
        ax.set_xlabel('Area In SQ. Feet')
        ax.set_ylabel('Price')
        # plt.show()
    plt.savefig('media/img/'+str(param2)+str(map)+'.jpg')

# for i in range(4):
#     map(i,3,1)

for i in range(4):
    for j in range(4):
        for k in range(6):
            filter(i,j,k) 
 

