# -*- coding: utf-8 -*-
"""
hydraulic_sim.py —— 阀-缸位置闭环半仿真（V0.2，总体设计 §6.2/§7.3，阶段0b-3）
==============================================================================
软件在环（SIL）：把 PLC 定位 FB（FB_AxisPos）的控制算法原样移植到 Python，
驱动简化的比例阀-油缸模型，验证全链路：
  轨迹(关节) → 斜坡限幅(关节空间) → 标定表转换 → 行程给定
  → 比例阀 → 油缸 → 行程反馈 → 标定表反算关节 → 到位判断

架构结论（V0.2 调试得出，PLC 版必须照搬）：
  * 斜坡/限幅必须在**关节空间**做——设计速度限值是关节量，而关节→行程当量
    非线性（局部可达平均值的 ±20%），在行程空间限速会饱和滞后 0.9°；
  * 速度前馈按"转换后行程给定的差分"计算，天然含局部当量，无需手动换算；
  * 死区补偿用**叠加式**（给定+死区偏置抵消阀死区，全程线性），不能用
    "小信号顶到死区值"的替换式——那会给执行器最低速度，减速段停不下来；
  * 终态靠阀死区自锁保位（带内不补偿），无极限环。

阀-缸模型（简化，控制结构验证用）：流量控制 q=Kv·u/10·sqrt(Δp/p_rated)，
油缸 v=q/A，恒负载力。压力动力学/油液压缩/摩擦非线性留待台架整定。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import cylinder_mapping as cm
import trajectory as tr


# ---------------------------------------------------------------- 执行器模型

@dataclass
class ValveCylinder:
    """比例阀+单出杆油缸（行程空间）。"""
    A: float              # 无杆腔有效面积 (m²)
    Kv: float             # 阀全开额定流量 (m³/s) @ p_rated
    p_rated: float        # 额定阀压降 (Pa)，默认 1e6（10 bar）
    ps: float             # 供油压力 (Pa)
    F_load: float         # 等效负载力 (N)
    x_max: float          # 行程上限 (m)
    dz: float = 0.5       # 阀死区 (V)
    x: float = 0.0

    def step(self, u: float, dt: float) -> float:
        u_eff = 0.0 if abs(u) < self.dz else u - math.copysign(self.dz, u)
        dp = max(self.ps - self.F_load / self.A, 5e5)        # 保底防 sqrt 失真
        q = self.Kv * (u_eff / 10.0) * math.sqrt(dp / self.p_rated)
        self.x = min(max(self.x + q / self.A * dt, 0.0), self.x_max)
        return self.x


# ---------------------------------------------------------------- 定位 FB（关节空间给定，与 SCL 版同构）

@dataclass
class FB_AxisPos:
    """单关节定位闭环 FB。给定/限幅/带宽均在关节空间，经标定表转换到行程。
    与 SCL 版 FB_AxisPos.scl 同构：q2s/s2q 在 PLC 里就是标定表插值。"""
    kp: float             # 比例增益 (V/m，行程误差)
    v_ramp: float         # 斜坡限速（关节单位/s：rad/s 或 m/s）
    tol: float            # 到位带宽（关节单位：rad 或 m）
    q2s: object = None    # 关节→行程 标定转换 callable（必填）
    s2q: object = None    # 行程→关节 反算 callable（必填）
    k_vff: float = 24.4   # 速度前馈 (V·s/m) = 10/满开比速
    dz_comp: float = 0.5  # 死区补偿偏置 (V)，与阀死区匹配
    u_lim: float = 10.0
    n_stable: int = 20    # 连续 N 周期在带内 → done
    ramp: float = 0.0     # 斜坡给定（关节单位）
    s_prev: float = 0.0   # 上一周期行程给定（前馈差分用）
    stable_cnt: int = 0
    done: bool = False
    alarm: bool = False
    t: float = 0.0

    def reset(self, target0: float, act_s: float):
        """换孔/上电：斜坡对齐目标起点，前馈差分基准对齐转换值。"""
        self.ramp = target0
        self.s_prev = self.q2s(target0)
        self.stable_cnt, self.t = 0, 0.0
        self.done, self.alarm = False, False

    def run(self, target: float, act_s: float, dt: float, enable: bool = True) -> float:
        if not enable:
            self.done, self.alarm = False, False
            self.stable_cnt, self.t = 0, 0.0
            return 0.0
        self.t += dt
        # 1) 关节空间斜坡限幅
        step = self.v_ramp * dt
        self.ramp += min(max(target - self.ramp, -step), step)
        # 2) 标定表转换 → 行程给定；前馈取转换后差分（自动含局部当量）
        s_t = self.q2s(self.ramp)
        v_inst = (s_t - self.s_prev) / dt
        self.s_prev = s_t
        # 3) 行程误差 + 前馈 + P
        err_s = s_t - act_s
        u = self.k_vff * v_inst + self.kp * err_s
        # 4) 叠加式死区补偿（带内不加，防爬行；带外全线性）
        act_q = self.s2q(act_s)
        if u != 0.0 and abs(target - act_q) > self.tol:
            u += math.copysign(self.dz_comp, u)
        u = min(max(u, -self.u_lim), self.u_lim)
        # 5) 到位判断（关节带宽）
        self.stable_cnt = self.stable_cnt + 1 if abs(target - act_q) < self.tol else 0
        self.done = self.stable_cnt >= self.n_stable
        self.alarm = self.t > 30.0 and not self.done
        return u


# ---------------------------------------------------------------- 试验台架

def run_case(title: str, cal, vc: ValveCylinder, fb: FB_AxisPos,
             q_traj: np.ndarray | None, q_target: float,
             q_start: float | None = None, dt: float = 0.01, verbose=False):
    """跑一个定位案例。阶跃：q_start→q_target 恒值；跟踪：q_traj 为关节给定序列。"""
    q_first = q_traj[0] if q_traj is not None else (q_start if q_start is not None
                                                    else q_target)
    n = len(q_traj) if q_traj is not None else int(30.0 / dt)
    qs = q_traj if q_traj is not None else np.full(n, q_target)

    vc.x = cm.q_to_stroke(cal, q_first)
    fb.reset(qs[0], vc.x)
    xs = np.zeros(n)
    for k in range(n):
        u = fb.run(qs[k], vc.x, dt, enable=True)
        xs[k] = vc.step(u, dt)

    q_act = np.array([cm.stroke_to_q(cal, x) for x in xs])
    final_err = abs(qs[-1] - q_act[-1])
    settle_idx = next((k for k in range(n)
                       if np.all(np.abs(q_act[k:] - qs[k:]) < math.radians(0.1))), n)
    overshoot = 0.0
    if q_traj is None:
        span = abs(qs[-1] - q_first)
        peak = np.max(np.abs(q_act - q_first))
        overshoot = max(0.0, (peak - span) / max(span, 1e-9) * 100)
    if verbose:
        k_peak = int(np.argmax(np.abs(q_act - qs)))
        print(f"    [调试] 最大跟随误差 t={k_peak*dt:.2f}s，"
              f"{math.degrees(abs(q_act[k_peak]-qs[k_peak])):.3f}°")

    deg = math.degrees
    print(f"  {title:<22} 到位误差 {deg(final_err):7.4f}°  "
          f"整定时间 {settle_idx*dt:5.1f}s  超调 {overshoot:5.1f}%  "
          f"done={fb.done}  alarm={fb.alarm}")
    return dict(final_err_deg=deg(final_err), settle_s=settle_idx * dt,
                overshoot_pct=overshoot, q_act=q_act, qs=qs)


# ---------------------------------------------------------------- 自检：变幅关节全链路

if __name__ == "__main__":
    deg, rad = math.degrees, math.radians

    # 变幅关节标定表（与 cylinder_mapping.__main__ 同一占位铰点）
    an_bianfu = cm.Anchors(0.55, rad(-90), 0.55, rad(10))
    cal = cm.calibrate("变幅", an_bianfu, rad(-10), rad(55))
    m_per_rad = cal["stroke_span"] / rad(65)
    print(f"行程当量均值 {m_per_rad*1000:.1f}mm/rad（局部非线性见 trajectory 教训）")

    # 油缸/阀参数（示意，对标 20MPa 定位回路、缸径90mm、NG10 阀 40L/min）
    vc = ValveCylinder(A=math.pi * 0.045**2, Kv=40 / 60000.0, p_rated=1e6,
                       ps=2.0e7, F_load=30e3, x_max=cal["stroke_span"] + 0.05)

    fb = FB_AxisPos(kp=120.0, v_ramp=rad(12), tol=rad(0.03),
                    q2s=lambda q: cm.q_to_stroke(cal, q),
                    s2q=lambda s: cm.stroke_to_q(cal, s),
                    k_vff=24.4)

    print("阀-缸位置闭环半仿真（变幅关节，10ms 控制周期，关节空间斜坡）")
    r1 = run_case("阶跃 20°", cal, vc, fb, None, rad(10.0), q_start=rad(-10.0))
    ok1 = r1["final_err_deg"] < 0.05 and r1["overshoot_pct"] < 5

    # 跟踪 PTP 轨迹（变幅 −10°→+55°），轨迹后追加 4s 整定观察段
    lim = tr.JointLimit("变幅", rad(-10), rad(55), rad(12), rad(35))
    seg = tr.plan_joint(rad(-10), rad(55), lim)
    ts = np.arange(0, seg.t_total + 4.0, 0.01)
    q_traj = np.array([seg.sample(t)[0] for t in ts])
    fb2 = FB_AxisPos(kp=120.0, v_ramp=rad(12), tol=rad(0.03),
                     q2s=lambda q: cm.q_to_stroke(cal, q),
                     s2q=lambda s: cm.stroke_to_q(cal, s),
                     k_vff=24.4)
    r2 = run_case("跟踪 PTP −10°→+55°", cal, vc, fb2, q_traj, rad(55), verbose=True)
    track_err = np.max(np.abs(r2["q_act"] - r2["qs"]))
    print(f"  跟踪最大跟随误差 {deg(track_err):.3f}°  "
          f"(关节 v_max={deg(lim.v_max):.0f}°/s)")
    # 判据：终态到位 <0.1°（孔位判据）；动态峰值 <0.8°（影响节拍不影响精度）
    ok2 = r2["final_err_deg"] < 0.1 and deg(track_err) < 0.8

    print(f"结果：阶跃精度 {'PASS' if ok1 else 'FAIL'} / "
          f"轨迹跟随 {'PASS' if ok2 else 'FAIL'}")
    print("说明：k_vff=24.4=1/满开比速；kp=120、带宽0.03° 为首轮整定值；"
          "终态靠阀死区自锁保位（无极限环），台架阶段按实测死区重标。")
