import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict

@dataclass
class Aggregator:
    """聚合器类，作为能源管理和流量协调的核心"""
    id: int
    location: Tuple[float, float]
    max_power: float = 500.0  # 聚合器最大功率（降低以体现限制）
    current_load: float = 0.0  # 当前负载
    connected_stations: List[int] = None
    
    def __init__(self, id: int, location: Tuple[float, float]):
        self.id = id
        self.location = location
        self.connected_stations = []
        self.current_load = 0.0
    
    def update_load(self, stations: List['BaseStation']):
        """更新聚合器负载"""
        self.current_load = sum(s.current_power for s in stations if s.aggregator_id == self.id)
    
    def get_available_capacity(self) -> float:
        """获取可用容量"""
        return self.max_power - self.current_load

@dataclass
class BaseStation:
    """基站类"""
    id: int
    location: Tuple[float, float]
    max_power: float
    aggregator_id: int  # 连接的聚合器ID
    current_power: float = 0.0
    renewable_energy: float = 0.0  # 太阳能发电量
    battery_level: float = 0.0
    max_battery: float = 100.0
    solar_panel_area: float = 10.0  # 太阳能板面积（平方米）
    solar_efficiency: float = 0.2  # 太阳能转换效率

class Network:
    """网络模型类"""
    def __init__(self, num_stations: int, area_size: Tuple[float, float]):
        self.num_stations = num_stations
        self.area_size = area_size
        self.stations: List[BaseStation] = []
        self.aggregator = None
        self.initialize_network()
    
    def initialize_network(self):
        """初始化网络，以聚合器为中心构建网络拓扑"""
        # 在区域中心放置聚合器
        center_x = self.area_size[0] / 2
        center_y = self.area_size[1] / 2
        self.aggregator = Aggregator(id=0, location=(center_x, center_y))
        
        # 在聚合器周围按环形分布放置基站
        radius = min(self.area_size) / 3  # 基站分布半径
        for i in range(self.num_stations):
            # 计算基站位置（环形分布）
            angle = 2 * np.pi * i / self.num_stations
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            
            # 创建基站实例
            station = BaseStation(
                id=i,
                location=(x, y),
                max_power=100.0,  # 最大功率，单位：W
                aggregator_id=0,  # 连接到中心聚合器
                battery_level=50.0,  # 初始电池电量，单位：kWh
                solar_panel_area=10.0,  # 太阳能板面积
                solar_efficiency=0.2  # 太阳能转换效率
            )
            self.stations.append(station)
            self.aggregator.connected_stations.append(i)
        
        # 更新聚合器初始负载
        self.aggregator.update_load(self.stations)
    
    def update_renewable_energy(self, time_step: int, use_aggregator: bool = True):
        """更新可再生能源供应（基于24小时正态分布的太阳能）"""
        hour = time_step % 24
        
        # 使用正态分布模拟24小时太阳能强度
        # 正午12点为峰值，标准差为3小时
        solar_intensity = np.exp(-(hour - 12)**2 / (2 * 3**2))
        
        # 标准太阳辐射强度（W/m²）
        standard_solar_irradiance = 1000.0
        
        for station in self.stations:
            # 计算太阳能发电量
            station.renewable_energy = (
                station.solar_panel_area * 
                station.solar_efficiency * 
                standard_solar_irradiance * 
                solar_intensity
            )
            # 添加一些随机波动（±10%）
            station.renewable_energy *= np.random.normal(1.0, 0.1)
        
        # 聚合器调度决策
        if use_aggregator:
            self.aggregator.update_load(self.stations)
            available_capacity = self.aggregator.get_available_capacity()
            total_renewable = sum(s.renewable_energy for s in self.stations)
            # 只在总可再生超过聚合器容量时进行比例缩减
            if total_renewable > available_capacity > 0:
                for station in self.stations:
                    station.renewable_energy = (station.renewable_energy / total_renewable) * available_capacity
        else:
            # 无聚合器时，每个基站独立管理可再生能源
            for station in self.stations:
                station.renewable_energy = min(station.renewable_energy, station.max_power)
    
    def update_battery_levels(self):
        """更新所有基站的电池电量"""
        for station in self.stations:
            # 计算净能量变化
            net_energy = station.renewable_energy - station.current_power
            
            # 更新电池电量，考虑充放电效率
            if net_energy > 0:  # 充电
                station.battery_level = np.clip(
                    station.battery_level + net_energy * 0.9,  # 充电效率90%
                    0,
                    station.max_battery
                )
            else:  # 放电
                station.battery_level = np.clip(
                    station.battery_level + net_energy / 0.9,  # 放电效率90%
                    0,
                    station.max_battery
                )
        
        # 更新聚合器负载
        self.aggregator.update_load(self.stations)
    
    def get_network_state(self) -> Dict:
        """获取网络当前状态"""
        return {
            'aggregator': {
                'id': self.aggregator.id,
                'location': self.aggregator.location,
                'current_load': self.aggregator.current_load,
                'available_capacity': self.aggregator.get_available_capacity(),
                'connected_stations': self.aggregator.connected_stations
            },
            'stations': [
                {
                    'id': s.id,
                    'location': s.location,
                    'power': s.current_power,
                    'renewable': s.renewable_energy,
                    'battery': s.battery_level,
                    'aggregator_id': s.aggregator_id,
                    'solar_panel_area': s.solar_panel_area,
                    'solar_efficiency': s.solar_efficiency
                }
                for s in self.stations
            ]
        } 