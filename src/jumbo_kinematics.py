# -*- coding: utf-8 -*-
"""
jumbo_kinematics.py —— 单臂凿岩台车钻臂运动学（V0.1）
=====================================================
对应《01_钻孔定位系统总体设计》§5。

机构模型（4R+2P 活动关节，翻转 J5 固定；对标带伸缩臂单臂台车）：
    基座 Trans(B) · Rz(θ1 摆臂) · R(-y,θ2 变幅) · Trans(d1 大臂伸缩)
                 · Rz(θ3 梁摆动) · R(-y,θ4 梁俯仰) · [Rx(θ5 翻转,固定0)] · Trans(d2 推进) → 钻头
零位构型：大臂水平指向 +X（掌子面方向），工具轴 = 连杆系 X 列。

求解：
    FK  —— 连杆变换连乘
    IK  —— 阻尼最小二乘(DLS)+数值雅可比+关节限位投影+随机重启
           残差 e = [Δp(3); a_cur × a_des(3)]
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------- 基础变换

def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1.0]])


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1.0]])


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])


def trans(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    return np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1.0]])


_AXIS_ROT = {"x": rot_x, "y": rot_y, "z": rot_z}
_AXIS_VEC = {"x": np.array([1.0, 0, 0]),
             "y": np.array([0, 1.0, 0]),
             "z": np.array([0, 0, 1.0])}


# ---------------------------------------------------------------- 关节定义

@dataclass
class Joint:
    """单个关节。axis 支持 'x'/'y'/'z' 及负号（如 '-y' 表示绕 -y 正向为正）。"""
    name: str
    jtype: str            # 'R' 转动 / 'P' 移动
    axis: str             # 本体系转轴/滑轴
    lower: float          # 下限（弧度 或 米）
    upper: float          # 上限
    value: float = 0.0    # 当前值

    def transform(self, q: float) -> np.ndarray:
        if self.jtype == "R":
            fn = _AXIS_ROT[self.axis.lstrip("-")]
            return fn(-q if self.axis.startswith("-") else q)
        v = _AXIS_VEC[self.axis.lstrip("-")]
        if self.axis.startswith("-"):
            v = -v
        return trans(*(v * q))


# ---------------------------------------------------------------- 整机参数

@dataclass
class JumboParams:
    """示意参数，对标耿力 G1Z80 级单臂台车（带伸缩臂，钻孔断面可达 ~10.7×8.45m）；
    实机数据以机构图纸标定为准。"""
    base_pos: tuple = (1.0, -0.2, 1.4)   # 大臂基座在车体系的位置 {V}
    boom_tel: tuple = (2.0, 5.8)         # 大臂伸缩范围 (m)：缩回/伸出（对标 G3ZE 伸缩结构）
    feed_min: float = 1.2                # 推进(梁轴枢→钻头)下限 (m)
    feed_max: float = 5.8                # 推进上限 (m)
    swing_lim: float = math.radians(45)  # J1 摆臂 ±45°（对标 G1Z70 实测参数）
    luff_lo: float = math.radians(-10)   # J2 变幅下限
    luff_hi: float = math.radians(55)    # J2 变幅上限
    beam_swing_lim: float = math.radians(45)   # J3 梁摆动 ±45°
    beam_pitch_lim: float = math.radians(100)  # J4 梁俯仰 ±100°


def build_chain(p: JumboParams) -> list:
    """按 §5.1 变换链构建关节表（顺序即链序）。"""
    return [
        Joint("J1_摆臂",    "R", "z",  -p.swing_lim,      p.swing_lim),
        Joint("J2_变幅",    "R", "-y", p.luff_lo,         p.luff_hi),
        Joint("J2b_大臂伸缩", "P", "x", p.boom_tel[0],     p.boom_tel[1]),
        Joint("J3_梁摆动",  "R", "z",  -p.beam_swing_lim, p.beam_swing_lim),
        Joint("J4_梁俯仰",  "R", "-y", -p.beam_pitch_lim, p.beam_pitch_lim),
        Joint("J5_翻转",    "R", "x",  0.0,               0.0),      # V0.1 固定
        Joint("J6_推进",    "P", "x",  p.feed_min,        p.feed_max),
    ]


class SingleArmJumbo:
    """单臂台车（带伸缩臂）：FK + IK。"""

    # 参与逆解的关节下标（J5 翻转固定不参与）
    ACTIVE = (0, 1, 2, 3, 4, 6)

    def __init__(self, params: JumboParams | None = None):
        self.p = params or JumboParams()
        self.joints = build_chain(self.p)
        self.base = trans(*self.p.base_pos)

    # ---------------- FK ----------------

    def fk(self, q: np.ndarray, frames: bool = False):
        """q: 完整关节向量(7,)。返回 4x4 工具位姿；frames=True 时附带关键构型点。
        变换序：base · J1摆臂 · J2变幅 · J2b伸缩 · J3 · J4 · J5 · J6推进。"""
        T = self.base.copy()
        key = {"base": T[:3, 3].copy()}
        for k, (j, qi) in enumerate(zip(self.joints, q)):
            T = T @ j.transform(qi)
            if k == 2:
                key["boom_head"] = T[:3, 3].copy()    # 大臂头部=推进梁枢点
            if k == 5:
                key["beam_pivot"] = T[:3, 3].copy()   # J5 后（翻转不影响枢点）
        key["tip"] = T[:3, 3].copy()
        return (T, key) if frames else T
    def tool_axis(self, T: np.ndarray) -> np.ndarray:
        """钻进方向（连杆系 X 列，指向掌子面为正）。"""
        return T[:3, 0]

    # ---------------- IK ----------------

    @staticmethod
    def _residual(T: np.ndarray, p_t: np.ndarray, a_t: np.ndarray) -> np.ndarray:
        dp = p_t - T[:3, 3]               # 期望−实际（沿雅可比下降的正确方向）
        da = a_t - T[:3, 0]               # 方向残差用差向量：带符号，180°反向时叉积会失明
        return np.concatenate([dp, da])

    def ik(self, target_pos, target_axis, q_init: np.ndarray | None = None,
           tol_pos: float = 5e-4, tol_axis: float = 1e-3,
           lam: float = 0.05, iters: int = 400, restarts: int = 5,
           seed: int = 7):
        """
        求 target_pos 处、钻轴沿 target_axis 的孔位逆解。
        返回 (q, info)；info 含收敛标志与误差。
        """
        p_t = np.asarray(target_pos, dtype=float)
        a_t = np.asarray(target_axis, dtype=float)
        a_t = a_t / np.linalg.norm(a_t)
        rng = random.Random(seed)
        idx = self.ACTIVE

        def clamp(qv):
            for i in idx:
                j = self.joints[i]
                qv[i] = min(max(qv[i], j.lower), j.upper)
            return qv

        def solve_from(q0):
            q = clamp(np.array(q0, dtype=float))
            for _ in range(iters):
                T = self.fk(q)
                e = self._residual(T, p_t, a_t)
                if np.linalg.norm(e[:3]) < tol_pos and np.linalg.norm(e[3:]) < tol_axis:
                    return q, True
                # 数值雅可比（残差 e(q) 的偏导，只对活动关节）
                J = np.zeros((6, len(idx)))
                for c, i in enumerate(idx):
                    dq = 1e-6
                    qp, qm = q.copy(), q.copy()
                    qp[i] += dq
                    qm[i] -= dq
                    J[:, c] = (self._residual(self.fk(qp), p_t, a_t)
                               - self._residual(self.fk(qm), p_t, a_t)) / (2 * dq)
                # 高斯-牛顿/DLS 下降步：J 是残差雅可比，故取负号
                dq = -(J.T @ np.linalg.solve(J @ J.T + lam * lam * np.eye(6), e))
                step = np.zeros_like(q)
                for c, i in enumerate(idx):
                    step[i] = dq[c]
                q = clamp(q + step)
            T = self.fk(q)
            e = self._residual(T, p_t, a_t)
            return q, bool(np.linalg.norm(e[:3]) < tol_pos and np.linalg.norm(e[3:]) < tol_axis)

        # 初始猜测：几何直指目标
        def guess():
            d = p_t - np.asarray(self.p.base_pos)
            q0 = np.zeros(7)
            q0[0] = math.atan2(d[1], d[0])                      # 摆臂朝向目标
            q0[1] = math.atan2(d[2], math.hypot(d[0], d[1]))    # 变幅仰角
            dist = float(np.linalg.norm(d))
            q0[2] = min(max(0.6 * dist, self.p.boom_tel[0]), self.p.boom_tel[1])  # 伸缩粗分
            q0[6] = min(max(0.4 * dist, self.p.feed_min), self.p.feed_max)        # 推进补余
            return q0

        cands = [guess() if q_init is None else np.array(q_init, float)]
        cands += [np.array([rng.uniform(j.lower, j.upper) for j in self.joints])
                  for _ in range(restarts)]

        best, best_err = None, np.inf
        for q0 in cands:
            q, ok = solve_from(q0)
            err = np.linalg.norm(self._residual(self.fk(q), p_t, a_t))
            if err < best_err:
                best, best_err = q, err
            if ok:
                break

        T = self.fk(best)
        info = {
            "converged": bool(np.linalg.norm(self._residual(T, p_t, a_t)[:3]) < tol_pos
                              and np.linalg.norm(self._residual(T, p_t, a_t)[3:]) < tol_axis),
            "pos_err": float(np.linalg.norm(T[:3, 3] - p_t)),
            "axis_err_deg": float(math.degrees(
                math.asin(min(1.0, np.linalg.norm(np.cross(self.tool_axis(T), a_t)))))),
        }
        return best, info

    def angles_deg(self, q: np.ndarray) -> dict:
        return {j.name: math.degrees(qi) for j, qi in zip(self.joints, q)}
