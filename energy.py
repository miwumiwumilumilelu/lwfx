import numpy as np
from typing import List, Tuple, Dict
from network import BaseStation
from algorithms import CentralizedAlgorithms, DistributedAlgorithms

class EnergyManager:
    """能源管理器类"""
    def __init__(self, network):
        self.network = network
        self.grid_price = 1.0  # 电网电价，单位：元/kWh
        self.battery_efficiency = 0.9  # 电池充放电效率
        self.lambda_1 = 0.5  # 能源成本权重
        self.lambda_2 = 0.5  # 服务质量权重
    
    def optimize_energy_allocation(self, time_step: int) -> List[float]:
        """优化能源分配"""
        # 1. 使用JESLS算法进行集中式优化
        jesls_result = CentralizedAlgorithms.solve_jesls(
            self.network.stations,
            time_step,
            self.lambda_1,
            self.lambda_2
        )
        
        # 2. 使用IPC算法进行功率控制
        ipc_result = CentralizedAlgorithms.solve_ipc(self.network.stations)
        
        # 3. 使用SWF算法进行可再生能源分配
        swf_result = CentralizedAlgorithms.solve_swf(self.network.stations)
        
        # 4. 使用分布式算法进行能源合作
        energy_transfers = DistributedAlgorithms.energy_cooperation(self.network.stations)
        
        # 更新基站状态
        for i, station in enumerate(self.network.stations):
            # 更新功率
            station.current_power = ipc_result[station.id]
            
            # 更新可再生能源
            station.renewable_energy = swf_result[station.id]
            
            # 更新电池电量
            if station.id in jesls_result['energy_config']['p_battery']:
                station.battery_level -= jesls_result['energy_config']['p_battery'][i]
                station.battery_level = max(0, station.battery_level)
            
            # 处理能源转移
            for (from_id, to_id), amount in energy_transfers.items():
                if station.id == from_id:
                    station.battery_level -= amount
                elif station.id == to_id:
                    station.battery_level += amount
        
        return [s.current_power for s in self.network.stations]
    
    def calculate_energy_cost(self, time_step: int) -> float:
        """计算能源成本"""
        total_cost = 0.0
        for station in self.network.stations:
            # 计算从电网获取的能量
            grid_energy = max(0, station.current_power - station.renewable_energy - station.battery_level)
            # 计算成本
            total_cost += grid_energy * self.grid_price
        
        return total_cost
    
    def update_grid_price(self, time_step: int):
        """更新电网电价（模拟分时电价）"""
        hour = time_step % 24
        if 8 <= hour <= 22:  # 高峰时段
            self.grid_price = 1.2
        else:  # 低谷时段
            self.grid_price = 0.8 