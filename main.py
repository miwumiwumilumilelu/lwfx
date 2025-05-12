import numpy as np
import matplotlib.pyplot as plt
from network import Network
from energy import EnergyManager
from traffic import TrafficManager
from algorithms import CentralizedAlgorithms
from typing import Tuple, Dict

def run_simulation(use_aggregator: bool, num_stations: int = 10, area_size: Tuple[float, float] = (1000, 1000)) -> Dict:
    """运行单次模拟"""
    # 初始化网络
    network = Network(num_stations=num_stations, area_size=area_size)
    energy_manager = EnergyManager(network)
    traffic_manager = TrafficManager(network)
    
    # 设置是否使用聚合器优化
    traffic_manager.set_aggregator_optimization(use_aggregator)
    
    # 模拟参数
    simulation_hours = 24
    metrics = {
        'battery_levels': [],
        'renewable_energy': [],
        'energy_cost': [],
        'service_quality': [],
        'aggregator_utilization': [],
        'load_balance_index': []
    }
    
    # 运行模拟
    for time_step in range(simulation_hours):
        # 更新可再生能源（根据是否使用聚合器切换分配模式）
        network.update_renewable_energy(time_step, use_aggregator)
        
        # 更新流量负载
        traffic_manager.update_traffic_load(time_step)
        
        # 使用集中式算法优化
        result = CentralizedAlgorithms.solve_jesls(
            stations=network.stations,
            time_step=time_step,
            lambda_1=0.5,
            lambda_2=0.5
        )
        
        # 更新网络状态
        for station in network.stations:
            station.current_power = result['energy_config']['power_allocation'][station.id]
        
        # 更新电池电量
        network.update_battery_levels()
        
        # 根据聚合器使用情况调整电网电价
        if use_aggregator:
            energy_manager.grid_price = 0.8  # 聚合器场景下较低电网电价
        else:
            energy_manager.grid_price = 1.2  # 无聚合器场景下较高电网电价
        
        # 计算能源成本
        energy_cost = energy_manager.calculate_energy_cost(time_step)
        
        # 计算网络性能
        performance = traffic_manager.calculate_network_performance()
        
        # 记录指标
        metrics['battery_levels'].append([s.battery_level for s in network.stations])
        metrics['renewable_energy'].append([s.renewable_energy for s in network.stations])
        metrics['energy_cost'].append(energy_cost)
        metrics['service_quality'].append(result['energy_config']['service_quality'])
        metrics['aggregator_utilization'].append(performance['aggregator_utilization'])
        metrics['load_balance_index'].append(performance['load_balance_index'])
    
    return metrics

def main():
    """主函数：运行对比模拟并展示结果"""
    # 运行两种场景的模拟
    metrics_with_aggregator = run_simulation(use_aggregator=True)
    metrics_without_aggregator = run_simulation(use_aggregator=False)
    
    # 仅生成能耗对比图
    plot_energy_cost_comparison(metrics_with_aggregator, metrics_without_aggregator)

def plot_energy_cost_comparison(metrics_with: Dict, metrics_without: Dict):
    """绘制能耗对比结果"""
    hours = range(24)
    plt.figure(figsize=(10, 5))
    plt.plot(hours, metrics_with['energy_cost'], 'b-', label='With Aggregator')
    plt.plot(hours, metrics_without['energy_cost'], 'r--', label='Without Aggregator')
    plt.xlabel('Hour')
    plt.ylabel('Energy Cost (¥)')
    plt.title('Energy Cost Comparison')
    plt.grid(True)
    plt.legend()
    plt.savefig('energy_cost_comparison.png')
    plt.close()

def plot_comparison_results(metrics_with: Dict, metrics_without: Dict):
    """绘制对比结果"""
    hours = range(24)
    
    # 创建图形，包含六个子图
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 18))
    
    # 1. 电池电量对比
    battery_with = np.array(metrics_with['battery_levels'])
    battery_without = np.array(metrics_without['battery_levels'])
    avg_battery_with = np.mean(battery_with, axis=1)
    avg_battery_without = np.mean(battery_without, axis=1)
    ax1.plot(hours, avg_battery_with, 'b-', label='With Aggregator')
    ax1.plot(hours, avg_battery_without, 'r--', label='Without Aggregator')
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('Average Battery Level (kWh)')
    ax1.set_title('Battery Levels Comparison')
    ax1.grid(True)
    ax1.legend()
    
    # 2. 可再生能源对比
    renewable_with = np.array(metrics_with['renewable_energy'])
    renewable_without = np.array(metrics_without['renewable_energy'])
    avg_renewable_with = np.mean(renewable_with, axis=1)
    avg_renewable_without = np.mean(renewable_without, axis=1)
    ax2.plot(hours, avg_renewable_with, 'g-', label='With Aggregator')
    ax2.plot(hours, avg_renewable_without, 'r--', label='Without Aggregator')
    ax2.set_xlabel('Hour')
    ax2.set_ylabel('Average Renewable Energy (W)')
    ax2.set_title('Renewable Energy Comparison')
    ax2.grid(True)
    ax2.legend()
    
    # 3. 能源成本对比
    ax3.plot(hours, metrics_with['energy_cost'], 'b-', label='With Aggregator')
    ax3.plot(hours, metrics_without['energy_cost'], 'r--', label='Without Aggregator')
    ax3.set_xlabel('Hour')
    ax3.set_ylabel('Energy Cost (¥)')
    ax3.set_title('Energy Cost Comparison')
    ax3.grid(True)
    ax3.legend()
    
    # 4. 服务质量对比
    ax4.plot(hours, metrics_with['service_quality'], 'b-', label='With Aggregator')
    ax4.plot(hours, metrics_without['service_quality'], 'r--', label='Without Aggregator')
    ax4.set_xlabel('Hour')
    ax4.set_ylabel('Service Quality Score')
    ax4.set_title('Service Quality Comparison')
    ax4.grid(True)
    ax4.legend()
    
    # 5. 聚合器利用率
    ax5.plot(hours, metrics_with['aggregator_utilization'], 'b-', label='With Aggregator')
    ax5.plot(hours, metrics_without['aggregator_utilization'], 'r--', label='Without Aggregator')
    ax5.set_xlabel('Hour')
    ax5.set_ylabel('Aggregator Utilization')
    ax5.set_title('Aggregator Utilization Comparison')
    ax5.grid(True)
    ax5.legend()
    
    # 6. 负载均衡指数对比
    ax6.plot(hours, metrics_with['load_balance_index'], 'b-', label='With Aggregator')
    ax6.plot(hours, metrics_without['load_balance_index'], 'r--', label='Without Aggregator')
    ax6.set_xlabel('Hour')
    ax6.set_ylabel('Load Balance Index')
    ax6.set_title('Load Balance Comparison')
    ax6.grid(True)
    ax6.legend()
    
    plt.tight_layout()
    plt.savefig('comparison_results.png')
    plt.close()

if __name__ == '__main__':
    main() 