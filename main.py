import numpy as np
import matplotlib.pyplot as plt
from network import Network
from energy import EnergyManager
from traffic import TrafficManager

def main():
    # 初始化网络
    network = Network(num_stations=10, area_size=(1000, 1000))
    energy_manager = EnergyManager(network)
    traffic_manager = TrafficManager(network)
    
    # 模拟参数
    simulation_hours = 24
    metrics = {
        'energy_cost': [],
        'network_performance': [],
        'battery_levels': [],
        'renewable_energy': []  # 新增：记录可再生能源
    }
    
    # 运行模拟
    for time_step in range(simulation_hours):
        # 更新可再生能源
        network.update_renewable_energy(time_step)
        
        # 更新流量负载
        traffic_manager.update_traffic_load(time_step)
        
        # 优化流量分布
        traffic_manager.optimize_traffic_distribution()
        
        # 优化能源分配
        energy_manager.optimize_energy_allocation(time_step)
        
        # 更新电池电量
        network.update_battery_levels()
        
        # 更新电网电价
        energy_manager.update_grid_price(time_step)
        
        # 记录指标
        metrics['energy_cost'].append(energy_manager.calculate_energy_cost(time_step))
        metrics['network_performance'].append(traffic_manager.calculate_network_performance())
        metrics['battery_levels'].append([s.battery_level for s in network.stations])
        metrics['renewable_energy'].append([s.renewable_energy for s in network.stations])
    
    # 绘制结果
    plot_results(metrics, simulation_hours)
    plot_network_topology(network)  # 新增：绘制网络拓扑

def plot_network_topology(network):
    """绘制网络拓扑结构"""
    plt.figure(figsize=(10, 10))
    
    # 绘制聚合器
    agg_x, agg_y = network.aggregator.location
    plt.plot(agg_x, agg_y, 'ro', markersize=15, label='Aggregator')
    
    # 绘制基站
    for station in network.stations:
        x, y = station.location
        plt.plot(x, y, 'bo', markersize=10, label=f'Station {station.id}' if station.id == 0 else "")
        
        # 绘制到聚合器的连接线
        plt.plot([x, agg_x], [y, agg_y], 'k--', alpha=0.3)
    
    plt.title('Network Topology')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.legend()
    plt.savefig('network_topology.png')
    plt.close()

def plot_results(metrics, simulation_hours):
    """绘制模拟结果"""
    hours = range(simulation_hours)
    
    # 创建图形
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 绘制能源成本
    ax1.plot(hours, metrics['energy_cost'], 'b-', label='Energy Cost')
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('Cost (¥)')
    ax1.set_title('Energy Cost Over Time')
    ax1.grid(True)
    
    # 绘制网络性能
    performance = metrics['network_performance']
    load_balance = [p['load_balance_index'] for p in performance]
    ax2.plot(hours, load_balance, 'g-', label='Load Balance Index')
    ax2.set_xlabel('Hour')
    ax2.set_ylabel('Balance Index')
    ax2.set_title('Network Load Balance Over Time')
    ax2.grid(True)
    
    # 绘制电池电量
    battery_levels = np.array(metrics['battery_levels'])
    for i in range(battery_levels.shape[1]):
        ax3.plot(hours, battery_levels[:, i], label=f'Station {i+1}')
    ax3.set_xlabel('Hour')
    ax3.set_ylabel('Battery Level (kWh)')
    ax3.set_title('Battery Levels Over Time')
    ax3.grid(True)
    ax3.legend()
    
    # 绘制可再生能源
    renewable_energy = np.array(metrics['renewable_energy'])
    avg_renewable = np.mean(renewable_energy, axis=1)
    ax4.plot(hours, avg_renewable, 'r-', label='Average Renewable Energy')
    ax4.set_xlabel('Hour')
    ax4.set_ylabel('Power (W)')
    ax4.set_title('Average Renewable Energy Over Time')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('simulation_results.png')
    plt.close()

if __name__ == '__main__':
    main() 