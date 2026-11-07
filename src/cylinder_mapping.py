# -*- coding: utf-8 -*-
"""
cylinder_mapping.py —— 关节角 ↔ 油缸行程 映射与标定（V0.2，总体设计 §6.3）
==========================================================================
仿真/PLC 工作在"关节空间"（角度），液压执行在"行程空间"（毫米）。
本模块：
  1) 按铰点几何计算每个转动关节油缸的行程-角度曲线（标定表）；
  2) 三次多项式拟合双向映射 s(θ) / θ(s)，给出拟合误差——发给 PLC 的标定公式雏形
     （V0.2 用几何解算；实机阶段按实测 ≥5 点重新拟合，吸收制造误差）。

铰点模型（平面机构，单缸）：
    转轴 O；缸筒铰点 B 固定在基座上：半径 r_b、绝对方位 alpha_b（相对关节零位方向）；
    活塞杆铰点 A 随连杆转动：半径 r_a、连杆系方位 beta（绝对方位 = θ + beta）。
    缸长 l(θ)=|A−B|；行程 s(θ)=l(θ)−l(θ_lo)（关节下限位时缸全缩，行程记 0）。

双缸说明：实机变幅/摆动等关节为双缸**并排**布置（垂直于机构平面偏置），
两缸平面几何相同 → 行程相同，本模块按单缸计算后复制。V0.1 调试教训：
不要把双缸建成平面内镜像（那会一伸一缩，均值无意义）。

⚠ 当前铰点尺寸为占位示意值（保证单调可用的合理布置），机构图纸到位后只改
  Anchors 参数重新标定。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Anchors:
    """单缸铰点参数（平面）。alpha_b/beta 单位 rad。
    lever_k/theta_offset：四连杆放大——油缸驱动曲柄角 φ=(θ−theta_offset)/lever_k，
    关节角 θ=lever_k·φ+theta_offset。大转角关节（如梁俯仰±100°，跨度>180° 窗口）
    直驱必然非单调，须放大机构；lever_k=1 且 theta_offset=0 退化为直驱。"""
    r_b: float          # 基座铰点半径 (m)
    alpha_b: float      # 基座铰点绝对方位（关节零位方向起量，rad）
    r_a: float          # 连杆铰点半径 (m)
    beta: float         # 连杆系铰点方位（rad）
    lever_k: float = 1.0
    theta_offset: float = 0.0


def cyl_length(an: Anchors, q: float) -> float:
    """缸长 l(θ)。"""
    phi = (q - an.theta_offset) / an.lever_k
    bx, by = an.r_b * math.cos(an.alpha_b), an.r_b * math.sin(an.alpha_b)
    ax = an.r_a * math.cos(phi + an.beta)
    ay = an.r_a * math.sin(phi + an.beta)
    return math.hypot(ax - bx, ay - by)


def calibrate(name: str, an: Anchors, q_lo: float, q_hi: float,
              n: int = 60, fit_deg: int = 5) -> dict:
    """在关节范围内采样 → 标定表 + 双向多项式拟合 + 误差评估。
    行程零点取在 q_lo（缸全缩）。"""
    qs = np.linspace(q_lo, q_hi, n)
    ls = np.array([cyl_length(an, q) for q in qs])
    s = ls - ls[0]                                           # 行程：相对下限位
    p_s_of_q = np.polyfit(qs, s, fit_deg)
    p_q_of_s = np.polyfit(s, qs, fit_deg)
    return dict(
        name=name,
        table=np.stack([qs, s]),
        p_s_of_q=p_s_of_q, p_q_of_s=p_q_of_s,
        stroke_span=float(s[-1] - s[0]),
        err_s=float(np.max(np.abs(np.polyval(p_s_of_q, qs) - s))),
        err_q_rad=float(np.max(np.abs(np.polyval(p_q_of_s, s) - qs))),
        monotonic=bool(np.all(np.diff(s) > 0)),
    )


def err_q_deg(cal: dict) -> float:
    return math.degrees(cal["err_q_rad"])


def q_to_stroke(cal: dict, q: float) -> float:
    """标定表线性插值：关节角 → 油缸行程 (m)。PLC 侧用同样的表插值实现。"""
    qs, ss = cal["table"]
    return float(np.interp(q, qs, ss))


def stroke_to_q(cal: dict, s: float) -> float:
    """标定表线性插值（反向）：行程 → 关节角。行程反馈编码器失效时的备份。"""
    qs, ss = cal["table"]
    return float(np.interp(s, ss, qs))       # s 随 q 单调 → np.interp 合法


def interp_err_q_deg(cal: dict) -> float:
    """表格插值的往返误差（θ→s→θ）。"""
    qs = cal["table"][0]
    errs = [abs(stroke_to_q(cal, q_to_stroke(cal, q)) - q) for q in qs]
    return math.degrees(max(errs))


def report(cal: dict, q_lo: float, q_hi: float) -> str:
    return (f"{cal['name']:<6} 行程需求 {cal['stroke_span']*1000:6.1f} mm  "
            f"行程拟合误差 {cal['err_s']*1e6:6.1f} μm  "
            f"角度反拟合误差 {err_q_deg(cal):6.4f}°  "
            f"单调 {'是' if cal['monotonic'] else '否'}")


# ---------------------------------------------------------------- 自检

if __name__ == "__main__":
    deg, rad = math.degrees, math.radians

    # 占位铰点（TODO：机构图纸到位后替换）。
    # 布置原则：alpha_b 选在关节范围外侧 → 缸长随 θ 单调增长。
    joints = [
        # name        Anchors(r_b, alpha_b,        r_a,  beta)      q_lo        q_hi
        ("摆臂",   Anchors(0.50, rad(-90),            0.50, rad(0)),  rad(-45), rad(45)),
        ("变幅",   Anchors(0.55, rad(-90),            0.55, rad(10)), rad(-10), rad(55)),
        ("梁摆动", Anchors(0.38, rad(-90),            0.38, rad(0)),  rad(-45), rad(45)),
        # ±100°（跨度200°>180°窗）→ 四连杆放大 k=2，曲柄 φ=θ/2 ∈ ±50°
        # 基座铰点 −100°：曲柄转角与基座方位夹角全程 50°~150°，缸长单调
        ("梁俯仰", Anchors(0.32, rad(-100),           0.36, rad(0), lever_k=2.0),
                   rad(-100), rad(100)),
    ]
    print("关节-油缸几何标定（示意铰点）：")
    cals = []
    for name, an, lo, hi in joints:
        c = calibrate(name, an, lo, hi)
        cals.append(c)
        print("  " + report(c, lo, hi)
              + f"  表插值往返误差 {interp_err_q_deg(c):6.4f}°")
        assert c["monotonic"], f"{name} 行程-角度曲线非单调，铰点布置不成立"
        assert c["stroke_span"] > 0.05, f"{name} 行程过小，铰点尺寸不合理"
        # 验收以表格插值为准（PLC 侧主用）；多项式仅备份，误差仅供评估
        assert interp_err_q_deg(c) < 0.01, f"{name} 表插值往返误差超 0.01°"

    print("验收基准：标定表 + 线性插值（PLC 侧主用）；多项式拟合仅作备份参考。")
