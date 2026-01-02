import matplotlib.pyplot as plt
import re
import os

def parse_report(filepath):
    scenarios = []
    current_scenario = {}
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if line.startswith("SCENARIO:"):
            if current_scenario:
                scenarios.append(current_scenario)
            current_scenario = {'name': line.split(":")[1].strip()}
        
        if not current_scenario:
            continue
            
        if line.startswith("Throughput:"):
            current_scenario['throughput'] = float(re.search(r"([\d\.]+)", line).group(1))
            
        if "INSERT Metrics" in line:
            current_mode = 'insert'
        elif "QUERY Metrics" in line:
            current_mode = 'query'
            
        if line.startswith("Average Latency:"):
            lat = float(re.search(r"([\d\.]+)", line).group(1))
            current_scenario[f'{current_mode}_latency'] = lat

    if current_scenario:
        scenarios.append(current_scenario)
    return scenarios

def plot_metrics(scenarios):
    names = [s['name'] for s in scenarios]
    throughputs = [s['throughput'] for s in scenarios]
    insert_lats = [s.get('insert_latency', 0) for s in scenarios]
    query_lats = [s.get('query_latency', 0) for s in scenarios]

    # Plot 1: Throughput
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, throughputs, color='#4CAF50')
    plt.title('Optimized Benchmark Throughput (127.0.0.1)', fontsize=14)
    plt.ylabel('Requests per Second (req/s)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom')
    
    plt.savefig('throughput_chart.png')
    print("Generated throughput_chart.png")

    # Plot 2: Latency
    plt.figure(figsize=(10, 6))
    x = range(len(names))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], insert_lats, width, label='Insert', color='#2196F3')
    plt.bar([i + width/2 for i in x], query_lats, width, label='Query', color='#FF9800')
    
    plt.title('Optimized Average Latency (ms)', fontsize=14)
    plt.ylabel('Latency (ms)', fontsize=12)
    plt.xticks(x, names)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.ylim(0, max(max(insert_lats), max(query_lats)) * 1.2)
    
    plt.savefig('latency_chart.png')
    print("Generated latency_chart.png")

if __name__ == "__main__":
    data = parse_report("benchmark_report_final.txt")
    plot_metrics(data)
