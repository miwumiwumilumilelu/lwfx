import numpy as np
from typing import List, Tuple, Dict
from network import BaseStation
from algorithms import CentralizedAlgorithms, DistributedAlgorithms

class TrafficManager:
    """流量管理器类"""
    def __init__(self, network):
        self.network = network
        self.traffic_pattern = self._generate_traffic_pattern()
        self.alpha = 0.5  # 负载均衡权重
        self.beta = 0.5   # 能源效率权重
        self.users = []  # 用户列表
    
    def _generate_traffic_pattern(self) -> np.ndarray:
        """生成24小时流量模式"""
        hours = np.arange(24)
        # 使用高斯混合模型模拟流量模式
        morning_peak = 0.8 * np.exp(-(hours - 9)**2 / 2)
        evening_peak = 0.9 * np.exp(-(hours - 19)**2 / 2)
        base_load = 0.3 * np.ones_like(hours)
        return morning_peak + evening_peak + base_load
    
    def update_traffic_load(self, time_step: int):
        """更新网络流量负载"""
        # 1. 使用LDP算法解决负载分布问题
        load_distribution = CentralizedAlgorithms.solve_ldp(self.network.stations)
        
        # 2. 使用分布式算法进行用户关联
        if self.users:
            user_associations = DistributedAlgorithms.user_association(
                self.network.stations,
                self.users
            )
            
            # 根据用户关联更新基站负载
            for station in self.network.stations:
                associated_users = [
                    user for user, station_id in user_associations.items()
                    if station_id == station.id
                ]
                station.current_power = load_distribution[station.id] * (1 + len(associated_users) * 0.1)
        else:
            # 如果没有用户，直接使用LDP结果
            for station in self.network.stations:
                station.current_power = load_distribution[station.id]
    
    def optimize_traffic_distribution(self) -> List[float]:
        """优化流量分布"""
        stations = self.network.stations
        
        # 对每个基站使用BPRA算法进行优化
        for station in stations:
            bpra_result = DistributedAlgorithms.solve_bpra(station)
            station.current_power = bpra_result['power']
        
        return [s.current_power for s in stations]
    
    def calculate_network_performance(self) -> dict:
        """计算网络性能指标"""
        stations = self.network.stations
        total_load = sum(s.current_power for s in stations)
        avg_load = total_load / len(stations)
        load_variance = np.var([s.current_power for s in stations])
        
        return {
            'total_load': total_load,
            'average_load': avg_load,
            'load_variance': load_variance,
            'load_balance_index': 1 - (load_variance / (avg_load ** 2))
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