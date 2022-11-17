import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
cust = pd.read_csv("/content/Mall_Customers.csv")
cust.shape
cust.head()
cust.info()
cust.rename(columns = {"Genre":"Gender"}, inplace = True)
cust.describe()
cust.drop(labels = 'CustomerID', axis = 1 , inplace = True)
cust.isnull().sum()
cust.dtypes
cust['Gender'].value_counts()
cust["Gender"].replace({"Male":1, "Female":0}, inplace = True)
sns.heatmap(data = cust.corr(), annot = True, fmt = ".2f", cmap = "cividis_r")
font = {"family":"Sherif", "size":16}
plt.subplots_adjust(left = 1, bottom = 1,right = 2.5, top = 2, wspace = 0.5, h
space = None)
plt.subplot(1,2,1)
plt.pie(x = [len(cust[cust.Gender == 1]) , len(cust[cust.Gender == 0])] , labe
ls = ['Male' , 'Female'], shadow = True , startangle = -
30 , explode = [0.1,0] , autopct = '%.0f%%')
plt.title("Customers gender", fontdict = font)
plt.subplot(1,2,2)
male_avg_score = cust[cust.Gender == 1]['Spending Score (1-100)'].mean()
female_avg_score = cust[cust.Gender == 0]['Spending Score (1-100)'].mean()
plt.bar(x = ['Male' , 'Female'] , height = [male_avg_score , female_avg_score]
, color = ['tab:cyan' , 'tab:green'])
plt.title('Customers spending score' , fontdict = font)
plt.ylabel('Average spending score' , fontdict = font)
plt.xlabel('Gender' , fontdict = font)
plt.text(-0.3 , 40 , 'Average = {:.2f}'.format(male_avg_score))
plt.text(0.7 , 40 , 'Average = {:.2f}'.format(female_avg_score))
plt.show()
age_list = cust.Age.unique()
age_list.sort()
avg_list = []
for age in age_list:
avg_list.append(cust[cust.Age == age]['Spending Score (1-100)'].mean())
plt.plot(age_list,avg_list)
plt.xlabel('Age' , fontdict = font)
plt.ylabel('Average spending score' , fontdict = {'family':'serif' , 'size':14
})
plt.title('Spending score in different ages')
plt.plot([20,70] , [40,40] , linestyle = '--' , c = 'tab:green' , alpha = 0.8)
plt.plot([35,35] , [10,90] , linestyle = '--' , c = 'tab:red' , alpha = 0.8)
plt.text(31,7,'Age = 35')
plt.show()
sc = StandardScaler()
data_scaled = sc.fit_transform(cust)
pca = PCA(n_components = 2)
data_pca = pca.fit_transform(data_scaled)
print("data shape after PCA :",data_pca.shape)
wcss_list = []
for i in range(1, 15):
kmeans = KMeans(n_clusters = i , init = 'k-means++' , random_state = 1)
kmeans.fit(data_pca)
wcss_list.append(kmeans.inertia_)
plt.plot(range(1,15) , wcss_list)
plt.plot([4,4] , [0 , 500] , linestyle = '--' , alpha = 0.7)
plt.text(4.2 , 300 , 'Elbow = 4')
plt.xlabel('K' , fontdict = font)
plt.ylabel('WCSS' , fontdict = font)
plt.show()
kmeans = KMeans(n_clusters = 4 , init = 'k-means++' , random_state = 1)
kmeans.fit(data_pca)
cluster_id = kmeans.predict(data_pca)
result_data = pd.DataFrame()
result_data['PC1'] = data_pca[:,0]
result_data['PC2'] = data_pca[:,1]
result_data['ClusterID'] = cluster_id
cluster_colors = {0:'tab:red' , 1:'tab:green' , 2:'tab:blue' , 3:'tab:pink'}
cluster_dict = {'Centroid':'tab:orange','Cluster0':'tab:red' , 'Cluster1':'tab
:green'
, 'Cluster2':'tab:blue' , 'Cluster3':'tab:pink'}
plt.scatter(x = result_data['PC1'] , y = result_data['PC2'] , c = result_data[
'ClusterID'].map(cluster_colors))
handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=v, label=k,
markersize=8) for k, v in cluster_dict.items()]
plt.legend(title='color', handles=handles, bbox_to_anchor=(1.05, 1), loc='uppe
r left')
plt.scatter(x = kmeans.cluster_centers_[:,0] , y = kmeans.cluster_centers_[:,1
] , marker = 'o' , c = 'tab:orange', s = 150 , alpha = 1)
plt.title("Clustered by KMeans" , fontdict = font)
plt.xlabel("PC1" , fontdict = font)
plt.ylabel("PC2" , fontdict = font)
plt.show()
