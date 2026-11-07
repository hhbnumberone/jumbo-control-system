# -*- coding: utf-8 -*-
"""
trajectory.py —— 关节空间轨迹规划（V0.2，总体设计 §9 阶段0b-1）
================================================================
功能：
  1) 单关节梯形/三角形速度规划（限速度、限加速度）
  2) 多关节同步 PTP：所有关节同时出发、同时到达（按最慢关节定总时长，其余缩放）
     —— 大臂协调运动的"不甩尾"基础；PLC 端按等周期下发插补点即可。

设计文档对应：§7.3（定位闭环的轨迹给定来源）、§8 S3（多关节协调运动）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------- 单关节规划

@dataclass
class JointLimit:
    name: str
    lower: float          # 下限（rad 或 m）
    upper: float          # 上限
    v_max: float          # 最大速度（单位/s）
    a_max: float          # 最大加速度（单位/s²）


@dataclass
class Segment:
    """一段梯形/三角形速度规划的解析描述。"""
    q0: float
    q1: float
    v_peak: float
    a: float
    t_acc: float          # 加速段时长
    t_flat: float         # 匀速段时长（三角形=0）
    t_total: float

    def sample(self, t: float) -> tuple[float, float]:
        """返回 (位置 q, 速度 v)。t>t_total 时返回终点。"""
        T, a, vp = self.t_total, self.a, self.v_peak
        t = min(max(t, 0.0), T)
        s = 1.0 if self.q1 >= self.q0 else -1.0
        if t < self.t_acc:                                   # 加速段
            return self.q0 + s * 0.5 * a * t * t, s * a * t
        t2 = t - self.t_acc
        if t2 < self.t_flat:                                 # 匀速段
            return self.q0 + s * (0.5 * a * self.t_acc**2 + vp * t2), s * vp
        t3 = t2 - self.t_flat                                # 减速段
        t_dec = T - self.t_acc - self.t_flat
        q_acc = 0.5 * a * self.t_acc**2
        q_flat = vp * self.t_flat
        return (self.q0 + s * (q_acc + q_flat + vp * t3 - 0.5 * a * t3 * t3),
                s * (vp - a * t3))


def plan_joint(q0: float, q1: float, lim: JointLimit) -> Segment:
    """单关节梯形规划：距离远用梯形，距离近自动退化为三角形。"""
    dist = abs(q1 - q0)
    sgn = 1.0 if q1 >= q0 else -1.0
    d_acc_full = lim.v_max**2 / (2 * lim.a_max)              # 加到满速需要的距离
    if dist >= 2 * d_acc_full:                               # 梯形
        t_acc = lim.v_max / lim.a_max
        t_flat = (dist - 2 * d_acc_full) / lim.v_max
        return Segment(q0, q1, lim.v_max, lim.a_max, t_acc, t_flat,
                       2 * t_acc + t_flat)
    v_peak = math.sqrt(dist * lim.a_max)                     # 三角形
    t_acc = v_peak / lim.a_max
    return Segment(q0, q1, v_peak, lim.a_max, t_acc, 0.0, 2 * t_acc)


# ---------------------------------------------------------------- 多关节同步

@dataclass
class PTPMotion:
    """多关节同步 PTP：总时长=最慢关节，其余关节按比例拉伸（等比降速保平滑）。"""
    segs: list[Segment]
    t_total: float

    @classmethod
    def plan(cls, q0: np.ndarray, q1: np.ndarray, limits: list[JointLimit]) -> "PTPMotion":
        assert len(q0) == len(q1) == len(limits)
        segs = [plan_joint(a, b, lim) for a, b, lim in zip(q0, q1, limits)]
        T = max(s.t_total for s in segs)
        # 各段相似拉伸到同一总时长 T：相位时长 ×(T/τ)=/k，速度 ×k，加速度 ×k²
        scaled = []
        for s in segs:
            if s.t_total < 1e-9:                       # 几乎不动：保持起点
                scaled.append(Segment(s.q0, s.q1, 0.0, 0.0, 0.0, T, T))
                continue
            k = s.t_total / T                          # ≤1
            t_acc = s.t_acc / k
            a = s.v_peak * k / t_acc                   # = s.a·k²，保持几何相似
            scaled.append(Segment(s.q0, s.q1, s.v_peak * k, a,
                                  t_acc, s.t_flat / k, T))
        return cls(scaled, T)

    def sample(self, t: float) -> np.ndarray:
        return np.array([s.sample(t)[0] for s in self.segs])


# ---------------------------------------------------------------- 自检

if __name__ == "__main__":
    deg = math.radians
    # 与 jumbo_kinematics.JumboParams 量级一致的关节限位（速度/加速度为设计值）
    limits = [
        JointLimit("摆臂",   deg(-45), deg(45),  deg(15), deg(40)),
        JointLimit("变幅",   deg(-10), deg(55),  deg(12), deg(35)),
        JointLimit("伸缩",   2.0, 5.8,  0.25, 0.5),          # m, m/s
        JointLimit("梁摆动", deg(-45), deg(45),  deg(20), deg(50)),
        JointLimit("梁俯仰", deg(-100), deg(100), deg(20), deg(50)),
        JointLimit("推进",   1.2, 5.8,  0.30, 0.8),          # m, m/s
    ]
    q0 = np.array([deg(3.8), deg(21.6), 3.25, deg(-4.1), deg(-21.6), 2.18])
    q1 = np.array([deg(-41.6), deg(27.5), 4.76, deg(45.0), deg(-20.2), 2.04])

    m = PTPMotion.plan(q0, q1, limits)
    print(f"同步 PTP 总时长：{m.t_total:.2f} s（{len(q0)} 关节）")

    # 校验：峰值速度不超限、终点精确、中途无越界
    ok = True
    for i, (s, lim) in enumerate(zip(m.segs, limits)):
        vs = [abs(s.sample(t)[1]) for t in np.linspace(0, s.t_total, 200)]
        if vs and max(vs) > lim.v_max * 1.001:
            ok = False
            print(f"  ✗ {lim.name} 峰值速度 {max(vs)/ (deg(1)):.1f}°/s 超限 "
                  f"{lim.v_max/deg(1):.1f}°/s")
    end = m.sample(m.t_total)
    err = np.max(np.abs(end - q1))
    ok &= err < 1e-9
    mid = m.sample(m.t_total / 2)
    ok &= all(lim.lower <= qi <= lim.upper
              for qi, lim in zip(mid, limits))
    print(f"  峰值速度/终点精度/中途限位检查：{'PASS' if ok else 'FAIL'}")
    print(f"  终点最大偏差 {err:.2e}")
    for name, seg in zip([l.name for l in limits], m.segs):
        print(f"  {name:<4} v_peak={abs(seg.v_peak):.4f}  T={seg.t_total:.2f}s")
