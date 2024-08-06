### Anomaly detection based on Kubernetes Metrics

Implementing Anomaly detection on a dataset is one of the easy things
that can be done in  Machine Learning . It's usually done on application data 
to gain insights on the various business use cases . However, it can also be applied
to DevOps use case . 

Example shows Anomaly detection by using two data points from the Kubernetes metrics 
server
 - CPU
 - Memory

How can this be applied  

- High CPU without any decrease in load is usually due to bugs in code
- High Memory without scaling down is due to memory leak 

Anomaly detection can automatically train from the input data and provide alerts if there are 
anomalies

#### Python Setup
I am using Python 3.11.7 and  pip 23.3.1

Install required libraries
```
pip install -U pysad
pip install kubernetes matplotlib numpy
pip install kubernetes matplotlib numpy pysad scikit-learn
ins
```
I am using Minikube , but you can set up your environment accordingly to expose the metrics
```
minikube start
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
```

#### kubernetes-fetch-metrics.py

Gets the metrics form k8s and Displays you should see output like the on execution

```
CPU Usage: 0.000750 cores, Memory Usage: 59.08 MB, Ratio: 0.000013
CPU Usage: 0.000228 cores, Memory Usage: 66.15 MB, Ratio: 0.000003
CPU Usage: 0.000750 cores, Memory Usage: 59.08 MB, Ratio: 0.000013
CPU Usage: 0.000228 cores, Memory Usage: 66.15 MB, Ratio: 0.000003
CPU Usage: 0.000750 cores, Memory Usage: 59.08 MB, Ratio: 0.000013
CPU Usage: 0.000228 cores, Memory Usage: 66.15 MB, Ratio: 0.000003
CPU Usage: 0.000750 cores, Memory Usage: 59.08 MB, Ratio: 0.000013
CPU Usage: 0.000228 cores, Memory Usage: 66.15 MB, Ratio: 0.000003
CPU Usage: 0.000750 cores, Memory Usage: 59.08 MB, Ratio: 0.000013
CPU Usage: 0.000228 cores, Memory Usage: 66.15 MB, Ratio: 0.000003
CPU Usage: 0.000750 cores, Memory Usage: 59.08 MB, Ratio: 0.000013
```

#### kubernetes-fetch-metrics-graph.py

Real time Unsupervised Learning from the initial dataset and provides anomalies going forward. 
It uses Isolation forest that works well with Unsupervised data.

Data from metrics server is used , for Minikube you can enable by 
```minikube addons enable metrics-server```

Also this is from default namespace , update your setup accordingly 

My setup I am using the consumememory application that is used for HPA to get the metrics . 
Here it is for reference https://github.com/arun2dot0/consumememory
You can start increasing the load after the data is trained ( 100 limit) to find anomalies

I was too many anomalies on initial try , so how to fix this ?
- Rolling median 

```
# Example: Using rolling median
  from pandas import Series

rolling_median = Series(data).rolling(window=window_size).median()
``````

- Require Consecutive Anomalies to be considered

```
# Example: Require 2 consecutive anomalies
consecutive_anomalies = 2
is_anomaly = all(anomaly_flags[-consecutive_anomalies:])
```
![Alt text](anomaly_k8s_metrics.png?raw=true "Demo")
