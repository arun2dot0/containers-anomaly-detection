from kubernetes import client, config
import time
import numpy as np
from collections import deque
import re

# Load kube config
config.load_kube_config()

# Create API instance
api = client.CustomObjectsApi()

# Initialize deques to store recent values for calculating rolling statistics
cpu_values = deque(maxlen=100)
memory_values = deque(maxlen=100)
cpu_memory_ratios = deque(maxlen=100)

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

def calculate_zscore(value, values):
    if len(values) < 2:
        return 0
    mean = np.mean(values)
    std = np.std(values)
    if std == 0:
        return 0
    return (value - mean) / std

def is_anomaly(zscore, threshold=3):
    return abs(zscore) > threshold

def detect_cpu_memory_anomaly(cpu, memory, cpu_memory_ratio):
    cpu_zscore = calculate_zscore(cpu, cpu_values)
    memory_zscore = calculate_zscore(memory, memory_values)
    ratio_zscore = calculate_zscore(cpu_memory_ratio, cpu_memory_ratios)
    
    high_cpu_low_memory = cpu_zscore > 2 and memory_zscore < 1 and ratio_zscore > 2
    high_memory_low_cpu = memory_zscore > 2 and cpu_zscore < 1 and ratio_zscore < -2
    
    return high_cpu_low_memory, high_memory_low_cpu

def main():
    while True:
        metrics = get_metrics()
        if metrics:
            data = preprocess_metrics(metrics)
            if data:
                for x in data:
                    cpu_usage, memory_usage = x
                    
                    # Calculate CPU/Memory ratio
                    cpu_memory_ratio = cpu_usage / (memory_usage + 1e-10)  # Avoid division by zero
                    
                    # Add current values to deques
                    cpu_values.append(cpu_usage)
                    memory_values.append(memory_usage)
                    cpu_memory_ratios.append(cpu_memory_ratio)
                    
                    # Detect anomalies
                    high_cpu_low_memory, high_memory_low_cpu = detect_cpu_memory_anomaly(cpu_usage, memory_usage, cpu_memory_ratio)
                    
                    print(f"CPU Usage: {cpu_usage:.6f} cores, Memory Usage: {memory_usage:.2f} MB, Ratio: {cpu_memory_ratio:.6f}")
                    
                    if high_cpu_low_memory:
                        print("ANOMALY DETECTED: High CPU usage without corresponding memory increase")
                    elif high_memory_low_cpu:
                        print("ANOMALY DETECTED: High memory usage without corresponding CPU load")
            else:
                print("No valid metrics data found")
        else:
            print("Failed to fetch metrics")
        time.sleep(10)  # Wait for 10 seconds before next fetch

if __name__ == "__main__":
    main()
