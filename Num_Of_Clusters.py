import pandas as pd
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

data = pd.read_csv("ClusterPlot.csv")

model = KMeans()

elbow_visualizer = KElbowVisualizer(model, k=(1,10), timings=False)

elbow_visualizer.fit(data)
print("Number of Clusters: ", elbow_visualizer.elbow_value_)
elbow_visualizer.show()


