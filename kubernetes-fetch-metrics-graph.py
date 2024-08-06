from kubernetes import client, config
import time
import numpy as np
from collections import deque
import re
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.ensemble import IsolationForest
from pandas import Series

# Load kube config
config.load_kube_config()

# Create API instance
api = client.CustomObjectsApi()

# Initialize deques to store recent values
cpu_values = deque(maxlen=100)
memory_values = deque(maxlen=100)

# Initialize lists to store data for plotting
cpu_data = []
memory_data = []
cpu_anomaly_markers = []
memory_anomaly_markers = []

# Initialize the Isolation Forest models for anomaly detection
cpu_isolation_forest = IsolationForest(contamination=0.1, random_state=42)
memory_isolation_forest = IsolationForest(contamination=0.1, random_state=42)

# Rolling median window size
window_size = 100

# Consecutive anomalies requirement
consecutive_anomalies_required = 2
cpu_anomaly_flags = deque(maxlen=consecutive_anomalies_required)
memory_anomaly_flags = deque(maxlen=consecutive_anomalies_required)

def parse_k8s_resource(resource_string):
    """Parse Kubernetes resource string to float value in standard units."""
    match = re.match(r'^(\d+(\.\d+)?)(n|u|m|k|Ki|Mi|Gi|Ti|Pi|Ei|K|M|G|T|P|E)?$', resource_string)
    if not match:
        return 0.0

    value, _, unit = match.groups()
    value = float(value)

    if unit == 'n':
        return value * 1e-9
    elif unit == 'u':
        return value * 1e-6
    elif unit == 'm':
        return value * 1e-3
    elif unit in ['K', 'Ki']:
        return value * 1024
    elif unit in ['M', 'Mi']:
        return value * 1024**2
    elif unit in ['G', 'Gi']:
        return value * 1024**3
    elif unit in ['T', 'Ti']:
        return value * 1024**4
    elif unit in ['P', 'Pi']:
        return value * 1024**5
    elif unit in ['E', 'Ei']:
        return value * 1024**6
    else:
        return value

def get_metrics():
    try:
        metrics = api.list_namespaced_custom_object(
            group="metrics.k8s.io",
            version="v1beta1",
            namespace="default",
            plural="pods"
        )
        return metrics
    except Exception as e:
        print(f"Error fetching metrics: {e}")
        return None

def preprocess_metrics(metrics):
    if not metrics or 'items' not in metrics:
        return []

    data = []
    for pod in metrics['items']:
        if 'containers' in pod and pod['containers']:
            cpu_usage = pod['containers'][0]['usage'].get('cpu', '0')
            memory_usage = pod['containers'][0]['usage'].get('memory', '0')

            # Parse CPU and memory usage
            cpu_cores = parse_k8s_resource(cpu_usage)
            memory_bytes = parse_k8s_resource(memory_usage)

            # Convert memory to MB for readability
            memory_mb = memory_bytes / (1024 * 1024)

            data.append([cpu_cores, memory_mb])
    return data

def detect_anomalies(cpu, memory):
    cpu_anomaly = False
    memory_anomaly = False

    # Add current values to deques
    cpu_values.append(cpu)
    memory_values.append(memory)

    # Check if we have enough data to detect anomalies
    if len(cpu_values) == cpu_values.maxlen:
        # Apply rolling median
        cpu_rolling_median = Series(cpu_values).rolling(window=window_size).median().iloc[-1]
        memory_rolling_median = Series(memory_values).rolling(window=window_size).median().iloc[-1]

        # Reshape for sklearn
        cpu_data_reshaped = np.array(list(cpu_values)).reshape(-1, 1)
        memory_data_reshaped = np.array(list(memory_values)).reshape(-1, 1)

        # Fit and predict
        cpu_isolation_forest.fit(cpu_data_reshaped)
        memory_isolation_forest.fit(memory_data_reshaped)

        cpu_anomaly = cpu_isolation_forest.predict([[cpu_rolling_median]]) == -1
        memory_anomaly = memory_isolation_forest.predict([[memory_rolling_median]]) == -1

    return cpu_anomaly, memory_anomaly

def update_plot(frame):
    print("Updating plot...")
    metrics = get_metrics()
    if metrics:
        data = preprocess_metrics(metrics)
        if data:
            for x in data:
                cpu_usage, memory_usage = x

                # Detect anomalies
                cpu_anomaly, memory_anomaly = detect_anomalies(cpu_usage, memory_usage)

                # Update anomaly flags
                cpu_anomaly_flags.append(cpu_anomaly)
                memory_anomaly_flags.append(memory_anomaly)

                # Check for consecutive anomalies
                if len(cpu_anomaly_flags) == consecutive_anomalies_required and all(cpu_anomaly_flags):
                    cpu_anomaly_markers.append(len(cpu_data))

                if len(memory_anomaly_flags) == consecutive_anomalies_required and all(memory_anomaly_flags):
                    memory_anomaly_markers.append(len(memory_data))

                # Update data for plotting
                cpu_data.append(cpu_usage)
                memory_data.append(memory_usage)

                print(f"CPU Usage: {cpu_usage:.6f} cores, Memory Usage: {memory_usage:.2f} MB")

                if cpu_anomaly:
                    print("CPU ANOMALY DETECTED!")
                if memory_anomaly:
                    print("MEMORY ANOMALY DETECTED!")
        else:
            print("No valid metrics data found")
    else:
        print("Failed to fetch metrics")

    # Update the plots
    ax1.clear()
    ax2.clear()
    ax1.plot(cpu_data, label='CPU Usage (cores)')
    ax2.plot(memory_data, label='Memory Usage (MB)')

    for marker in cpu_anomaly_markers:
        ax1.axvline(x=marker, color='r', linestyle='--', alpha=0.7)
    for marker in memory_anomaly_markers:
        ax2.axvline(x=marker, color='r', linestyle='--', alpha=0.7)

    ax1.legend()
    ax2.legend()

# Set up the plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
ani = FuncAnimation(fig, update_plot, interval=10000)  # Update every 10 seconds

plt.tight_layout()
plt.show()

print("Script completed.")
