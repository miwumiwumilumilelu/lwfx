import numpy as np
import cvxpy as cp
from typing import List, Dict, Tuple
from network import BaseStation, Aggregator

class CentralizedAlgorithms:
    @staticmethod
    def solve_jesls(stations: List[BaseStation], 
                   time_step: int,
                   lambda_1: float = 0.5,
                   lambda_2: float = 0.5,
                   max_iterations: int = 100,
                   convergence_threshold: float = 1e-6) -> Dict:
        n = len(stations)
        
        # 初始化基站的传输功率为最大值
        for station in stations:
            station.current_power = station.max_power
        
        prev_objective = float('inf')
        energy_sufficient = False  # 添加可再生能源充足指示变量
        
        for iteration in range(max_iterations):
            # 1. 解决负载分布问题 (LDP)
            load_distribution = CentralizedAlgorithms.solve_ldp(stations)
            
            # 2. 解决能源配置问题 (ECP)
            energy_config = CentralizedAlgorithms.solve_ecp(stations, load_distribution)
            
            # 3. 计算当前目标函数值
            current_objective = (
                lambda_1 * energy_config['grid_cost'] - 
                lambda_2 * energy_config['service_quality']
            )
            
            # 4. 检查可再生能源是否充足
            renewable_energy_sum = sum(s.renewable_energy for s in stations)
            total_energy_demand = sum(load_distribution[s.id] for s in stations)
            if renewable_energy_sum >= total_energy_demand:
                energy_sufficient = True
            else:
                energy_sufficient = False
            
            # 5. 检查收敛性
            if abs(current_objective - prev_objective) < convergence_threshold:
                break
                
            prev_objective = current_objective
            
        return {
            'load_distribution': load_distribution,
            'energy_config': energy_config,
            'energy_sufficient': energy_sufficient,  # 返回可再生能源是否充足
            'iterations': iteration + 1
        }
    @staticmethod
    def solve_ldp(stations: List[BaseStation]) -> Dict[int, float]:
        n = len(stations)
        
        # 获取当前状态
        current_loads = np.array([s.current_power for s in stations])
        battery_levels = np.array([s.battery_level for s in stations])
        renewable_energy = np.array([s.renewable_energy for s in stations])
        
        # 定义优化变量
        x = cp.Variable(n)
        phi = cp.Variable()  # 新增变量 phi
        
        # 目标函数：最大化最小冗余传输速率
        objective = cp.Maximize(phi)
        
        # 约束条件
        constraints = [
            x >= 0,
            x <= current_loads,
            x <= battery_levels + renewable_energy,
            current_loads - x >= phi  # 修改：直接使用向量比较
        ]
        
        # 求解优化问题
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.ECOS)
            return {station.id: float(x.value[i]) for i, station in enumerate(stations)}
        except:
            return {station.id: station.current_power for station in stations}
    
    @staticmethod
    def solve_ecp(stations: List[BaseStation], 
                load_distribution: Dict[int, float]) -> Dict:
        """能源配置问题 (ECP)"""
        n = len(stations)
        
        # 1. 解决功率控制问题 (IPC)
        power_config = CentralizedAlgorithms.solve_ipc(stations)
        
        # 2. 解决能源合作问题 (SWF)
        energy_config = CentralizedAlgorithms.solve_swf(stations)
        
        # 3. 计算目标函数值
        energy_cost = sum(energy_config['p_grid']) * 0.5  # 假设电网电价为0.5元/单位
        service_quality = sum(energy_config['service_level']) / n  # 归一化服务等级
        
        # 4. 计算总功率分配
        power_allocation = {
            station.id: (
                energy_config['p_renewable'][i] +
                energy_config['p_battery'][i] +
                energy_config['p_grid'][i]
            )
            for i, station in enumerate(stations)
        }
        
        return {
            'p_grid': energy_config['p_grid'],
            'p_battery': energy_config['p_battery'],
            'p_renewable': energy_config['p_renewable'],
            'service_level': energy_config['service_level'],
            'grid_cost': energy_cost,
            'service_quality': service_quality,
            'power_allocation': power_allocation  # 添加功率分配
        }
    
    @staticmethod
    def solve_ipc(stations: List[BaseStation], 
                  max_iterations: int = 100,
                  convergence_threshold: float = 1e-6) -> Dict[int, float]:
        """迭代功率控制算法 (IPC)"""
        n = len(stations)
        prev_power = np.array([s.current_power for s in stations])
        
        for iteration in range(max_iterations):
            # 计算干扰函数
            interference = np.zeros(n)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        interference[i] += prev_power[j]
            
            # 更新功率
            current_power = np.array([
                min(s.max_power, s.current_power / (1 + interference[i]))
                for i, s in enumerate(stations)
            ])
            
            # 检查收敛性
            if np.max(np.abs(current_power - prev_power)) < convergence_threshold:
                break
                
            prev_power = current_power
            
        return {station.id: float(current_power[i]) for i, station in enumerate(stations)}
    
    @staticmethod
    def solve_swf(stations: List[BaseStation]) -> Dict:
        """序贯注水算法 (SWF)"""
        n = len(stations)
        renewable_energy = np.array([s.renewable_energy for s in stations])
        current_load = np.array([s.current_power for s in stations])
        battery_levels = np.array([s.battery_level for s in stations])
        
        # 初始化能源分配
        p_renewable = np.zeros(n)  # 可再生能源分配
        p_battery = np.zeros(n)    # 电池能源分配
        p_grid = np.zeros(n)       # 电网能源分配
        service_level = np.zeros(n)  # 服务等级
        
        # 按负载排序
        sorted_indices = np.argsort(current_load)
        remaining_renewable = np.sum(renewable_energy)
        
        # 序贯分配能源
        for idx in sorted_indices:
            if remaining_renewable <= 0:
                break
                
            # 1. 优先使用可再生能源
            renewable_to_use = min(
                current_load[idx],
                renewable_energy[idx],
                remaining_renewable
            )
            p_renewable[idx] = renewable_to_use
            remaining_renewable -= renewable_to_use
            
            # 2. 其次使用电池能源
            remaining_load = current_load[idx] - p_renewable[idx]
            battery_to_use = min(remaining_load, battery_levels[idx])
            p_battery[idx] = battery_to_use
            
            # 3. 最后使用电网能源
            p_grid[idx] = max(0, remaining_load - battery_to_use)
            
            # 计算服务等级
            service_level[idx] = min(1.0, (p_renewable[idx] + p_battery[idx] + p_grid[idx]) / current_load[idx])
        
        return {
            'p_renewable': p_renewable.tolist(),
            'p_battery': p_battery.tolist(),
            'p_grid': p_grid.tolist(),
            'service_level': service_level.tolist()
        }

class DistributedAlgorithms:
    """分布式算法实现"""
    
    @staticmethod
    def user_association(stations: List[BaseStation], 
                        users: List[Dict]) -> Dict[int, int]:
        """基于负载的用户关联算法"""
        associations = {}
        
        for user in users:
            # 计算到每个基站的距离和负载
            station_metrics = []
            for station in stations:
                distance = np.sqrt(
                    (user['location'][0] - station.location[0])**2 +
                    (user['location'][1] - station.location[1])**2
                )
                load_factor = station.current_power / station.max_power
                metric = 1 / (distance * (1 + load_factor))
                station_metrics.append((station.id, metric))
            
            # 选择最佳基站
            best_station = max(station_metrics, key=lambda x: x[1])[0]
            associations[user['id']] = best_station
            
        return associations
    
    @staticmethod
    def solve_bpra(station: BaseStation) -> Dict[str, float]:
        """基于二分法的功率和资源分配算法 (BPRA)"""
        # 二分法参数
        left = 0
        right = station.max_power
        tolerance = 1e-6
        max_iterations = 100
        
        for _ in range(max_iterations):
            mid = (left + right) / 2
            
            # 计算当前功率下的性能
            performance = DistributedAlgorithms._calculate_performance(station, mid)
            
            if performance > station.current_power:
                left = mid
            else:
                right = mid
                
            if right - left < tolerance:
                break
                
        optimal_power = (left + right) / 2
        return {
            'power': optimal_power,
            'performance': DistributedAlgorithms._calculate_performance(station, optimal_power)
        }
        
    @staticmethod
    def _calculate_performance(station: BaseStation, power: float) -> float:
        """计算给定功率下的性能"""
        # 简化的性能计算模型
        return power * (1 - station.current_power / station.max_power)
    
    @staticmethod
    def energy_cooperation(stations: List[BaseStation]) -> Dict[Tuple[int, int], float]:
        """小区间能源合作算法"""
        n = len(stations)
        energy_transfers = {}
        
        # 计算每个基站的能源盈余/缺口
        energy_balance = np.array([
            s.renewable_energy + s.battery_level - s.current_power
            for s in stations
        ])
        
        # 找出能源盈余和缺口的基站
        surplus_stations = np.where(energy_balance > 0)[0]
        deficit_stations = np.where(energy_balance < 0)[0]
        
        # 进行能源交易
        for i in surplus_stations:
            for j in deficit_stations:
                if energy_balance[i] <= 0 or energy_balance[j] >= 0:
                    continue
                    
                # 计算可转移的能源量
                transfer_amount = min(
                    energy_balance[i],
                    -energy_balance[j]
                )
                
                if transfer_amount > 0:
                    energy_transfers[(i, j)] = transfer_amount
                    energy_balance[i] -= transfer_amount
                    energy_balance[j] += transfer_amount
        
        return energy_transfers