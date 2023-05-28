import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.datasets import make_blobs

#sidebar
st.sidebar.markdown("Clustering의 :red[변수] ")
number = st.sidebar.number_input('How many center do you want?',1,6,3)
st.sidebar.write('The current center is ', number)
#center = st.slider('How many center do you want?', 1, 10, 5)
#st.write("The current center is ", center)
k = st.sidebar.slider('How many K do you want?', 1, 6, 3)
st.sidebar.write("The current k is ", k)

#first title
col1, col2 = st.columns(2)
col1.title("Pattern_Final_Test")
col2.subheader("[202302801 김다운]")
st.title("")
st.title("")
#second title
st.title("Clustering - K-medoids / K-Means")
st.title("")
st.header("**Clustering - 군집분석**")
st.markdown("")
st.markdown("'Clustering(군집분석)'은 :red[비지도학습](unsupervised learning)의 일종으로 기준이 없는 상태에서 주어진 데이터의 속성값들을 고려해 :blue[유사한 데이터끼리 그룹화를 시키는 학습 모델]을 말한다. 각 데이터의 유사성을 측정하여, 유사성이 높은 집단끼리  분류하고 군집간에 상이성을 규명하는 방법이다.")
st.markdown("")
img = Image.open("clustering.jpg")
st.image(
    img,
    caption="clustering 방법",
    width=650,
    channels="RGB"
)

#K-medoids

st.title("")
st.header("K-medoids")
st.markdown("")
st.markdown("군집의 무게 중심을 구하기 위해서 데이터의 :red[평균 대신 중간점](medoids)을 사용하는 알고리즘이다.")

data,true_labels=make_blobs(n_samples=450,centers=number,cluster_std=0.6,random_state=0)

kmedoids=KMedoids(n_clusters=k,random_state=0).fit(data)

for i in range(k):
    cluster_data=data[kmedoids.labels_==i]
    plt.scatter(cluster_data[:,0],cluster_data[:,1],label=f"Cluster{i+1}")
    
plt.scatter(kmedoids.cluster_centers_[:,0],kmedoids.cluster_centers_[:,1],marker="x",color="k",s=100,label="Medoids")
plt.legend()
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("K_medoids Clustering")
#plt.show()
st.pyplot(plt)

#K-Means

st.title("")
st.header("K-means")
st.markdown("")
st.markdown("각 군집에 할당된 포인트들의 :red[평균 좌표를 이용]해서 중심점을 반복적으로 업데이트하며 군집을 형성하는 알고리즘이다.")
X,y=make_blobs(n_samples=450, centers=number,cluster_std=0.9)
np.random.seed(42)

plt.scatter(X[:,0],X[:,1],marker='.')

k_means=KMeans(init="k-means++",n_clusters=k,n_init=12)
k_means.fit(X)

k_means_labels=k_means.labels_
print('k_means_label :',k_means_labels)

k_means_cluster_centers=k_means.cluster_centers_
print('k_means_cluster_center : ',k_means_cluster_centers)

fig=plt.figure(figsize=(6,4))

colors=plt.cm.Spectral(np.linspace(0,1,len(set(k_means_labels))))

ax=fig.add_subplot(1,1,1)

for j, col in zip(range(k),colors):
    my_members=(k_means_labels==j)
    cluster_center=k_means_cluster_centers[j]

    ax.plot(X[my_members,0],X[my_members,1],'w',markerfacecolor=col,marker='.')
    ax.plot(cluster_center[0],cluster_center[1],'o',markerfacecolor=col,markeredgecolor='k',markersize=6)

    ax.set_title('K_Means Clustering')
    ax.set_xticks(())
    ax.set_yticks(())
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    #plt.show()
    
    st.pyplot(fig)
