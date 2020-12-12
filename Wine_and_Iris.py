import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine, load_iris

#Load the wine and iris data then convert them to data frame
wine = load_wine()
iris = load_iris()

wine_df = pd.DataFrame(wine.data, columns = wine.feature_names)
iris_df = pd.DataFrame(iris.data, columns = iris.feature_names)

#Run K-Means for a range of clusters using a for loop and collecting the distortions into a list
wine_distortions = []
for k in range(1,15):
    wine_kmean_Model = KMeans(n_clusters=k).fit(wine_df)
    wine_distortions.append(wine_kmean_Model.inertia_)
#Plot the distortions of K-Means
plt.figure(figsize=(16,8))
plt.plot(range(1,15), wine_distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Heuristic showing Wine')
plt.savefig('Wine.png')
plt.show()


#Run K-Means for a range of clusters using a for loop and collecting the distortions into a list
iris_distortions = []
for k in range(1,15):
    iris_kmean_Model =  KMeans(n_clusters=k).fit(iris_df)
    iris_distortions.append(iris_kmean_Model.inertia_)
#Plot the distortions of K-Means
plt.figure(figsize=(16, 8))
plt.plot(range(1, 15), iris_distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Heuristic showing Iris')
plt.savefig('Wine.png')
plt.show()

#From the two graphes we get, we can observe that the “elbow” is the number 3 which is optimal for this case.