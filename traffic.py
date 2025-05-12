import numpy as np
from typing import List, Tuple, Dict
from network import BaseStation, Aggregator
from algorithms import CentralizedAlgorithms, DistributedAlgorithms

class TrafficManager:
    """流量管理器类，负责网络流量管理和负载均衡"""
    def __init__(self, network):
        self.network = network
        self.traffic_pattern = self._generate_traffic_pattern()
        self.alpha = 0.5  # 负载均衡权重
        self.beta = 0.5   # 能源效率权重
        self.users = []  # 用户列表
        self.use_aggregator = True  # 是否使用聚合器优化
    
    def _generate_traffic_pattern(self) -> np.ndarray:
        """生成24小时流量模式"""
        hours = np.arange(24)
        # 使用高斯混合模型模拟流量模式
        morning_peak = 0.8 * np.exp(-(hours - 9)**2 / 2)  # 早高峰
        evening_peak = 0.9 * np.exp(-(hours - 19)**2 / 2)  # 晚高峰
        base_load = 0.3 * np.ones_like(hours)  # 基础负载
        return morning_peak + evening_peak + base_load
    
    def update_traffic_load(self, time_step: int):
        """更新网络流量负载，考虑聚合器状态"""
        # 1. 获取当前流量模式
        current_traffic = self.traffic_pattern[time_step % 24]
        
        # 2. 如果不使用聚合器，基站各自只承担当前流量
        if not self.use_aggregator:
            for station in self.network.stations:
                station.current_power = current_traffic
            # 更新聚合器负载
            self.network.aggregator.update_load(self.network.stations)
            return
        
        # 3. 使用聚合器时，全局优化负载分布
        aggregator = self.network.aggregator
        available_capacity = aggregator.get_available_capacity()
        # 使用LDP算法解决负载分布问题
        load_distribution = CentralizedAlgorithms.solve_ldp(self.network.stations)
        # 考虑聚合器容量限制
        total_load = sum(load_distribution.values())
        if total_load > available_capacity and available_capacity > 0:
            scale_factor = available_capacity / total_load
            load_distribution = {k: v * scale_factor for k, v in load_distribution.items()}
        # 4. 使用分布式算法进行用户关联和负载调整
        if self.users:
            user_associations = DistributedAlgorithms.user_association(
                self.network.stations,
                self.users
            )
            for station in self.network.stations:
                associated = sum(1 for u, sid in user_associations.items() if sid == station.id)
                agg_factor = 1 - (aggregator.current_load / aggregator.max_power)
                station.current_power = load_distribution[station.id] * (1 + associated * 0.1) * (1 + agg_factor * 0.2)
        else:
            # 无用户，仅按比例分配
            for station in self.network.stations:
                agg_factor = 1 - (aggregator.current_load / aggregator.max_power)
                station.current_power = load_distribution[station.id] * (1 + agg_factor * 0.2)
        # 更新聚合器负载
        self.network.aggregator.update_load(self.network.stations)
    
    def optimize_traffic_distribution(self) -> List[float]:
        """优化流量分布，考虑聚合器状态"""
        stations = self.network.stations
        aggregator = self.network.aggregator
        
        # 对每个基站使用BPRA算法进行优化
        for station in stations:
            # 考虑聚合器状态进行优化
            if self.use_aggregator:
                aggregator_factor = 1 - (aggregator.current_load / aggregator.max_power)
                # 调整基站最大功率
                original_max_power = station.max_power
                station.max_power *= (1 + aggregator_factor * 0.2)
                
                # 使用BPRA算法
                bpra_result = DistributedAlgorithms.solve_bpra(station)
                station.current_power = bpra_result['power']
                
                # 恢复原始最大功率
                station.max_power = original_max_power
            else:
                # 不使用聚合器优化时的简单优化
                bpra_result = DistributedAlgorithms.solve_bpra(station)
                station.current_power = bpra_result['power']
        
        # 更新聚合器负载
        aggregator.update_load(stations)
        return [s.current_power for s in stations]
    
    def calculate_network_performance(self) -> Dict:
        """计算网络性能指标，包括聚合器性能"""
        stations = self.network.stations
        aggregator = self.network.aggregator
        
        # 计算基站性能指标
        total_load = sum(s.current_power for s in stations)
        avg_load = total_load / len(stations)
        load_variance = np.var([s.current_power for s in stations])
        
        # 计算聚合器性能指标
        aggregator_utilization = aggregator.current_load / aggregator.max_power
        aggregator_efficiency = 1 - (load_variance / (avg_load ** 2)) if avg_load > 0 else 0
        
        return {
            'total_load': total_load,
            'average_load': avg_load,
            'load_variance': load_variance,
            'load_balance_index': 1 - (load_variance / (avg_load ** 2)) if avg_load > 0 else 0,
            'aggregator_utilization': aggregator_utilization,
            'aggregator_efficiency': aggregator_efficiency
        }
    
    def add_user(self, user_id: int, location: Tuple[float, float]):
        """添加用户到网络"""
        self.users.append({
            'id': user_id,
            'location': location
        })
    
    def remove_user(self, user_id: int):
        """从网络中移除用户"""
        self.users = [u for u in self.users if u['id'] != user_id]
    
    def set_aggregator_optimization(self, use_aggregator: bool):
        """设置是否使用聚合器优化"""
        self.use_aggregator = use_aggregator 