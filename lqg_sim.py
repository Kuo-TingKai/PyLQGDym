import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.special import gamma
from dataclasses import dataclass
from typing import Dict, List, Tuple
import itertools

@dataclass
class SpinFoam:
    """自旋泡沫結構"""
    vertices: List[np.ndarray]  # 頂點位置
    edges: List[Tuple]         # 邊的連接
    faces: List[List]          # 面的定義
    spins: Dict               # 邊上的自旋標籤
    intertwiners: Dict        # 頂點上的交織子

class LQGSimulation:
    def __init__(self):
        # 基本物理常數
        self.PLANCK_LENGTH = 1.616255e-35  # 米
        self.PLANCK_TIME = 5.391247e-44    # 秒
        self.NEWTON_G = 6.67430e-11        # 牛頓引力常數
        self.HBAR = 1.054571817e-34        # 普朗克常數
        self.C = 299792458                 # 光速
        
        # Immirzi參數 (BBM值)
        self.IMMIRZI = 0.2375
        
        self.foam = self._initialize_spin_foam()
        self.holonomy_cache = {}
        self.animation_frames = []  # 添加用於存儲動畫幀的列表
        
    def _initialize_spin_foam(self) -> SpinFoam:
        """初始化自旋泡沫結構"""
        # 增加頂點數量
        vertices = [np.array([0,0,0,0])]  # 中心點
        n_vertices = 10  # 增加到10個頂點
        
        # 在4維超球面上生成隨機頂點
        for _ in range(n_vertices - 1):
            # 生成隨機4維向量
            v = np.random.randn(4)
            # 標準化到planck長度
            v = v / np.linalg.norm(v) * self.PLANCK_LENGTH
            vertices.append(v)
            
        # 創建更多的邊和面
        edges = list(itertools.combinations(range(n_vertices), 2))
        faces = []
        for i, j, k in itertools.combinations(range(n_vertices), 3):
            faces.append([i,j,k])
            
        # 初始化自旋和交織子
        spins = {edge: self._random_spin() for edge in edges}
        intertwiners = {i: self._calculate_intertwiner(i, edges, spins) 
                       for i in range(n_vertices)}
        
        return SpinFoam(vertices, edges, faces, spins, intertwiners)
    
    def _random_spin(self) -> float:
        """生成有效的自旋標籤，增加可能的自旋值"""
        # 增加自旋值的範圍和精細度
        spins = np.arange(0.5, 5.1, 0.5)  # 從0.5到5.0，步長0.5
        # 使用指數分佈的權重，使得較小的自旋值更常見
        weights = np.exp(-spins)
        weights = weights / np.sum(weights)
        return float(np.random.choice(spins, p=weights))
    
    def _calculate_intertwiner(self, vertex: int, 
                             edges: List[Tuple], 
                             spins: Dict) -> np.ndarray:
        """計算頂點的交織子"""
        # 獲取與頂點相連的邊的自旋
        vertex_spins = []
        for e in edges:
            if vertex in e:
                vertex_spins.append(spins[e])
                
        # 使用 Clebsch-Gordan 係數計算交織子
        # 這裡使用簡化版本
        n = len(vertex_spins)
        if n == 0:
            return np.array([1.0])
        
        dim = int(2 * sum(vertex_spins) + 1)
        return np.random.randn(dim)
    
    def calculate_area(self, face: List) -> float:
        """計算面的量子化面積"""
        # 使用 LQG 面積算子公式
        spins = []
        for i in range(len(face)):
            for j in range(i+1, len(face)):
                edge = tuple(sorted([face[i], face[j]]))
                if edge in self.foam.spins:
                    spins.append(self.foam.spins[edge])
        
        area = 0
        for j in spins:
            area += self.PLANCK_LENGTH**2 * self.IMMIRZI * \
                   np.sqrt(j * (j + 1))
            
        return 8 * np.pi * area
    
    def calculate_volume(self, vertex: int) -> float:
        """計算頂點的量子化體積"""
        # 獲取與頂點相關的所有自旋
        adjacent_spins = []
        for edge in self.foam.edges:
            if vertex in edge:
                adjacent_spins.append(self.foam.spins[edge])
                
        if len(adjacent_spins) < 4:  # 需要至少4個自旋
            return 0.0
            
        # 使用 LQG 體積算子公式的簡化版本
        volume = self.PLANCK_LENGTH**3 * self.IMMIRZI**(3/2)
        volume *= np.abs(np.prod(adjacent_spins))**(1/4)
        return volume
    
    def calculate_holonomy(self, loop: List[int]) -> np.ndarray:
        """計算環跡全純量"""
        loop_key = tuple(loop)
        if loop_key in self.holonomy_cache:
            return self.holonomy_cache[loop_key]
            
        # 計算 SU(2) 全純量
        holonomy = np.eye(2, dtype=complex)
        for i in range(len(loop)):
            j = (i + 1) % len(loop)
            edge = tuple(sorted([loop[i], loop[j]]))
            if edge in self.foam.spins:
                spin = self.foam.spins[edge]
                # 使用 SU(2) 表示
                theta = 2 * np.pi * spin
                h = np.array([[np.cos(theta/2), -np.sin(theta/2)],
                             [np.sin(theta/2), np.cos(theta/2)]])
                holonomy = holonomy @ h
                
        self.holonomy_cache[loop_key] = holonomy
        return holonomy
    
    def hamiltonian_constraint(self) -> float:
        """計算哈密頓約束"""
        H = 0
        # 對每個頂點
        for v in range(len(self.foam.vertices)):
            volume = self.calculate_volume(v)
            # 計算相關的環跡全純量
            for face in self.foam.faces:
                if v in face:
                    hol = self.calculate_holonomy(face)
                    H += volume * np.real(np.trace(hol))
        
        return H * self.HBAR / self.PLANCK_TIME
    
    def evolve(self, steps: int = 100, dt: float = 1e-44):
        """時間演化"""
        energies = []
        self.animation_frames = []
        
        for _ in range(steps):
            # 記錄當前狀態用於動畫
            current_state = {
                'vertices': [v.copy() for v in self.foam.vertices],
                'edges': self.foam.edges.copy(),
                'spins': self.foam.spins.copy()
            }
            self.animation_frames.append(current_state)
            
            # 記錄能量
            E = self.hamiltonian_constraint()
            energies.append(E)
            
            # 更新自旋配置
            for edge in self.foam.edges:
                if np.random.random() < 0.2:  # 增加自旋更新的概率
                    old_spin = self.foam.spins[edge]
                    new_spin = self._random_spin()
                    # 只接受能量降低的變化
                    self.foam.spins[edge] = new_spin
                    new_energy = self.hamiltonian_constraint()
                    
                    if new_energy > E:  # 如果能量增加，有機會拒絕這個變化
                        if np.random.random() > np.exp(-(new_energy - E)/(self.HBAR/self.PLANCK_TIME)):
                            self.foam.spins[edge] = old_spin
                    
                    # 更新相關交織子
                    for v in edge:
                        self.foam.intertwiners[v] = \
                            self._calculate_intertwiner(v, 
                                                      self.foam.edges, 
                                                      self.foam.spins)
            
            # 清除全純量緩存
            self.holonomy_cache.clear()
            
        return np.array(energies)
    
    def create_animation(self, filename='lqg_evolution.gif', fps=10):
        """創建演化動畫"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        def update(frame):
            ax.clear()
            state = self.animation_frames[frame]
            vertices = np.array(state['vertices'])[:,:3]  # 只取前三維
            
            # 繪製頂點
            ax.scatter(vertices[:,0], vertices[:,1], vertices[:,2], 
                      c='red', s=100)
            
            # 繪製邊
            for edge in state['edges']:
                v1, v2 = vertices[edge[0]], vertices[edge[1]]
                spin = state['spins'][edge]
                ax.plot([v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]], 
                       linewidth=spin*2, alpha=0.6)
            
            # 去掉網格線和座標軸
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            
            ax.set_title(f"Loop Quantum Gravity Spin Foam (Step {frame})")
            ax.set_xlim([vertices[:,0].min(), vertices[:,0].max()])
            ax.set_ylim([vertices[:,1].min(), vertices[:,1].max()])
            ax.set_zlim([vertices[:,2].min(), vertices[:,2].max()])
        
        from matplotlib.animation import FuncAnimation, PillowWriter
        anim = FuncAnimation(fig, update, frames=len(self.animation_frames), 
                           interval=1000/fps)
        
        # 保存為 GIF
        writer = PillowWriter(fps=fps)
        anim.save(filename, writer=writer)
        plt.close()
    
    def visualize(self):
        """視覺化自旋泡沫"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 繪製頂點
        vertices = np.array(self.foam.vertices)[:,:3]  # 只取前三維
        ax.scatter(vertices[:,0], vertices[:,1], vertices[:,2], 
                  c='red', s=100)
        
        # 繪製邊
        for edge in self.foam.edges:
            v1, v2 = vertices[edge[0]], vertices[edge[1]]
            spin = self.foam.spins[edge]
            ax.plot([v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]], 
                   linewidth=spin*2, alpha=0.6)
            
        # 去掉網格線和座標軸
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        
        ax.set_title("Loop Quantum Gravity Spin Foam")
        plt.show()

# 主程序
if __name__ == "__main__":
    # 創建模擬
    lqg = LQGSimulation()
    
    # 視覺化初始狀態
    print("初始自旋泡沫結構：")
    lqg.visualize()
    
    # 演化系統並創建動畫
    print("系統演化中...")
    energies = lqg.evolve(steps=200)
    print("創建動畫中...")
    lqg.create_animation()
    
    # 繪製能量演化圖
    plt.figure(figsize=(10, 6))
    plt.plot(energies)
    plt.title("Loop Quantum Gravity Energy Evolution")
    plt.xlabel("Time Steps")
    plt.ylabel("Energy (J)")
    plt.yscale('log')
    plt.grid(True)
    plt.show()
    
    # 視覺化最終狀態
    print("最終自旋泡沫結構：")
    lqg.visualize()