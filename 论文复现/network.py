import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Aggregator:
    """聚合器类"""
    id: int
    location: Tuple[float, float]
    connected_stations: List[int] = None
    
    def __init__(self, id: int, location: Tuple[float, float]):
        self.id = id
        self.location = location
        self.connected_stations = []

@dataclass
class BaseStation:
    """基站类"""
    id: int
    location: Tuple[float, float]
    max_power: float
    aggregator_id: int  # 连接的聚合器ID
    current_power: float = 0.0
    renewable_energy: float = 0.0
    battery_level: float = 0.0
    max_battery: float = 100.0

class Network:
    """网络模型类"""
    def __init__(self, num_stations: int, area_size: Tuple[float, float]):
        self.num_stations = num_stations
        self.area_size = area_size
        self.stations: List[BaseStation] = []
        self.aggregator = None
        self.initialize_network()
    
    def initialize_network(self):
        """初始化网络，放置聚合器和基站"""
        # 在区域中心放置聚合器
        center_x = self.area_size[0] / 2
        center_y = self.area_size[1] / 2
        self.aggregator = Aggregator(id=0, location=(center_x, center_y))
        
        # 在聚合器周围随机放置基站
        for i in range(self.num_stations):
            # 在聚合器周围随机生成位置
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(0, min(self.area_size) / 3)  # 限制基站到聚合器的距离
            x = center_x + distance * np.cos(angle)
            y = center_y + distance * np.sin(angle)
            
            station = BaseStation(
                id=i,
                location=(x, y),
                max_power=100.0,  # 最大功率，单位：W
                aggregator_id=0,  # 连接到中心聚合器
                battery_level=50.0  # 初始电池电量，单位：kWh
            )
            self.stations.append(station)
            self.aggregator.connected_stations.append(i)
    
    def update_renewable_energy(self, time_step: int):
        """更新可再生能源供应（基于正态分布的太阳能）"""
        hour = time_step % 24
        
        # 使用正态分布模拟太阳能强度
        # 假设正午12点为峰值，标准差为3小时
        solar_intensity = np.exp(-(hour - 12)**2 / (2 * 3**2))
        
        for station in self.stations:
            # 添加一些随机波动
            random_factor = np.random.normal(1.0, 0.1)
            # 计算太阳能发电量
            station.renewable_energy = station.max_power * solar_intensity * random_factor
    
    def update_battery_levels(self):
        """更新所有基站的电池电量"""
        for station in self.stations:
            # 计算净能量变化
            net_energy = station.renewable_energy - station.current_power
            # 更新电池电量
            station.battery_level = np.clip(
                station.battery_level + net_energy,
                0,
                station.max_battery
            )
    
    def get_network_state(self) -> dict:
        """获取网络当前状态"""
        return {
            'aggregator': {
                'id': self.aggregator.id,
                'location': self.aggregator.location,
                'connected_stations': self.aggregator.connected_stations
            },
            'stations': [
                {
                    'id': s.id,
                    'location': s.location,
                    'power': s.current_power,
                    'renewable': s.renewable_energy,
                    'battery': s.battery_level,
                    'aggregator_id': s.aggregator_id
                }
                for s in self.stations
            ]
        } 