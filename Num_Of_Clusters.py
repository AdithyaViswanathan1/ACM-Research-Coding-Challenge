import pandas as pd
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from matplotlib import pyplot as plt

data = pd.read_csv("ClusterPlot.csv")

model = KMeans()

elbow_visualizer = KElbowVisualizer(model, k=(1,10), timings=False)

elbow_visualizer.fit(data)
print("Number of Clusters: ", elbow_visualizer.elbow_value_)
elbow_visualizer.show()

x = data.copy()
kmeans = KMeans(elbow_visualizer.elbow_value_)
kmeans.fit(x)

clusters = x.copy()
clusters["cluster_pred"] = kmeans.fit_predict(x)
plt.scatter(data["V1"], data["V2"], c=clusters["cluster_pred"], cmap = "rainbow")
plt.xlabel("V1")
plt.ylabel("V2")
plt.show()






