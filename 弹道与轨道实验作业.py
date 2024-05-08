import numpy as np

# 万有引力常数 (m^3/kg/s^2)
G = 6.67430e-11

# 地球参数
M = 5.972e24  # 地球质量 (kg)
R = 6371e3  # 地球半径 (m)

# 观测数据
rA = np.array([3626e3, -6111e3, 3626e3])  # 位置 (m)
vA = np.array([3.828e3, 4.543e3, 3.828e3])  # 速度 (m/s)
t0 = 0  # 初始时间 (s)

# 计算加速度的函数
def acceleration(r):
    r_norm = np.linalg.norm(r)
    return -G * M / r_norm**3 * r

# 执行龙格库塔积分的函数
def runge_kutta_step(r, v, dt):
    k1v = dt * acceleration(r)
    k1r = dt * v
    k2v = dt * acceleration(r + k1r / 2)
    k2r = dt * (v + k1v / 2)
    k3v = dt * acceleration(r + k2r / 2)
    k3r = dt * (v + k2v / 2)
    k4v = dt * acceleration(r + k3r)
    k4r = dt * (v + k3v)
    v_new = v + (k1v + 2*k2v + 2*k3v + k4v) / 6
    r_new = r + (k1r + 2*k2r + 2*k3r + k4r) / 6
    return r_new, v_new

# 时间步长
dt = 1  # 时间步长 (s)
num_steps = 600  # 要计算多少时间后的位置和速度

# 用于存储状态历史记录的数组
r_history = [rA.copy()]
v_history = [vA.copy()]

# 执行积分
r = rA.copy()
v = vA.copy()
for i in range(num_steps):
    r, v = runge_kutta_step(r, v, dt)
    r_history.append(r.copy())
    v_history.append(v.copy())

# 最终位置和速度
r_final = r
v_final = v

# 计算轨道参数的函数
def orbital_elements(r, v):
    h = np.cross(r, v)
    h_norm = np.linalg.norm(h)
    
    n = np.cross([0, 0, 1], h)
    n_norm = np.linalg.norm(n)
    
    e_vec = ((np.linalg.norm(v)**2 - G * M / np.linalg.norm(r)) * r - np.dot(r, v) * v) / (G * M)
    e = np.linalg.norm(e_vec)
    
    a = 1 / (2 / np.linalg.norm(r) - np.linalg.norm(v)**2 / (G * M))
    
    inc = np.arccos(h[2] / h_norm)
    
    if n_norm == 0:
        raan = 0
    else:
        raan = np.arccos(n[0] / n_norm)
        if n[1] < 0:
            raan = 2 * np.pi - raan
    
    if e == 0:
        argp = 0
        nu = 0
    else:
        argp = np.arccos(np.dot(n, e_vec) / (n_norm * e))
        if e_vec[2] < 0:
            argp = 2 * np.pi - argp
        
        nu = np.arccos(np.dot(e_vec, r) / (e * np.linalg.norm(r)))
        if np.dot(r, v) < 0:
            nu = 2 * np.pi - nu
    
    return a, e, np.degrees(inc), np.degrees(raan), np.degrees(argp), np.degrees(nu)

# 计算最终位置和速度的轨道元素
a, e, inc, raan, argp, nu = orbital_elements(r_final, v_final)

# 判断目标是导弹还是卫星
h_min = a - a*e
if h_min <= 6371000:
    print("该目标是导弹")
else:
    print("该目标是卫星")

#输出结果
print()
print("十分钟后卫星位置:", r_final)
print("十分钟后卫星速度:", v_final)
print()
print("轨道参数:")
print("轨道半长轴:", a)
print("轨道偏心率:", e)
print("轨道倾角:", inc)
print("升交点赤经:", raan)