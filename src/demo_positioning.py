# -*- coding: utf-8 -*-
"""
demo_positioning.py —— 自动寻孔定位仿真 Demo（V0.1）
====================================================
场景：8m 宽 × 6m 高掌子面（平面 x=6.0），单臂台车自动定位 5 个代表炮孔。
运行：python demo_positioning.py
输出：孔位求解表（控制台） + 钻臂构型三维图 jumbo_ik_demo.png
"""

import math
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")  # 无显示环境也可出图
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

try:  # Windows 中文字体，失败则退回默认（图仍可用）
    matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
    matplotlib.rcParams["axes.unicode_minus"] = False
except Exception:
    pass

from jumbo_kinematics import JumboParams, SingleArmJumbo

FACE_X = 6.2                                    # 掌子面平面（基座至掌子面 5.2m）
FACE = dict(y=(-3.0, 3.0), z=(0.3, 4.9))        # 掌子面 6.0m × 4.6m（本臂配置的可达断面）

# 孔位清单：(孔号, 类型, 掌子面坐标, 钻轴方向)。方向垂直掌子面=+X，H5 为带外插角掏槽孔
HOLES = [
    ("H1", "辅助孔", (FACE_X, 0.0, 2.6),  (1, 0, 0)),
    ("H2", "周边孔", (FACE_X, 2.8, 4.2),  (1, 0, 0)),
    ("H3", "周边孔", (FACE_X, -2.8, 0.8), (1, 0, 0)),
    ("H4", "周边孔", (FACE_X, -3.0, 3.6), (1, 0, 0)),
    ("H5", "掏槽孔", (FACE_X, 1.8, 1.5),  (1.0, 0.36, 0.27)),
]


def solve_all():
    jumbo = SingleArmJumbo(JumboParams())
    results = []
    print("=" * 88)
    print("单臂凿岩台车 自动寻孔定位仿真（V0.1，示意参数对标 G1Z80 级）")
    print(f"大臂基座 {jumbo.p.base_pos}  大臂伸缩 {jumbo.p.boom_tel[0]}-{jumbo.p.boom_tel[1]}m  "
          f"推进 {jumbo.p.feed_min}-{jumbo.p.feed_max}m  掌子面 x={FACE_X}")
    print("=" * 88)
    print(f"{'孔号':<4}{'类型':<5}{'位置误差':>10}{'方向误差':>10}  "
          f"{'摆臂θ1':>8}{'变幅θ2':>8}{'伸缩m':>8}{'梁摆θ3':>8}{'梁俯θ4':>8}{'推进d':>8}  状态")
    print("-" * 96)
    for name, htype, pos, axis in HOLES:
        q, info = jumbo.ik(pos, axis)
        ang = jumbo.angles_deg(q)
        status = "PASS" if info["converged"] else "FAIL"
        print(f"{name:<4}{htype:<5}{info['pos_err']*1000:>9.2f}mm"
              f"{info['axis_err_deg']:>9.3f}° "
              f"{ang['J1_摆臂']:>8.2f}{ang['J2_变幅']:>8.2f}{q[2]:>8.2f}"
              f"{ang['J3_梁摆动']:>8.2f}{ang['J4_梁俯仰']:>8.2f}"
              f"{q[6]:>8.2f}m  {status}")
        _, key = jumbo.fk(q, frames=True)
        results.append((name, htype, np.array(pos, float),
                        np.array(axis, float), key, info))
    print("-" * 96)
    n_ok = sum(1 for r in results if r[5]["converged"])
    print(f"结果：{n_ok}/{len(HOLES)} 孔定位收敛（判据：位置<0.5mm 且 方向<0.057°）")
    return results


def plot(results):
    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(111, projection="3d")

    # ---- 掌子面（半透明板） ----
    y0, y1 = FACE["y"]; z0, z1 = FACE["z"]
    face = [[FACE_X, y0, z0], [FACE_X, y1, z0], [FACE_X, y1, z1], [FACE_X, y0, z1]]
    ax.add_collection3d(Poly3DCollection([face], alpha=0.15, facecolor="gray"))

    # ---- 车体示意（基座处小方块） ----
    base_pos = results[0][4]["base"]
    ax.scatter(*base_pos, color="k", s=80, marker="s", label="大臂基座")

    colors = plt.cm.tab10.colors
    for i, (name, htype, pos, axis, key, info) in enumerate(results):
        c = colors[i % len(colors)]
        b = key["base"]; h = key["boom_head"]; p = key["beam_pivot"]; t = key["tip"]
        # 大臂 → 梁枢 → 钻头 折线
        ax.plot([b[0], h[0]], [b[1], h[1]], [b[2], h[2]],
                "-o", color=c, lw=2, ms=4, label=f"{name} 构型")
        ax.plot([h[0], p[0]], [h[1], p[1]], [h[2], p[2]], "-", color=c, lw=2)
        ax.plot([p[0], t[0]], [p[1], t[1]], [p[2], t[2]], "-", color=c, lw=1.5)
        # 孔位 + 孔轴
        ax.scatter(*pos, color=c, marker="*", s=120, edgecolors="k", zorder=5)
        if info["converged"]:
            u = axis / np.linalg.norm(axis)
            ax.quiver(pos[0], pos[1], pos[2], -u[0], -u[1], -u[2],
                      length=0.9, color=c, arrow_length_ratio=0.12, lw=1.2)

    ax.set_xlabel("X 车→掌子面 (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_xlim(0, FACE_X + 1); ax.set_ylim(-4.5, 4.5); ax.set_zlim(0, 6.5)
    ax.set_box_aspect((FACE_X + 1, 9.0, 6.5))
    ax.set_title("单臂凿岩台车 自动寻孔定位仿真 V0.1\n"
                 "（每色一条构型：基座→大臂→推进梁→钻头；★孔位，箭头=孔轴）")
    ax.legend(loc="upper left", fontsize=8)
    ax.view_init(elev=22, azim=-58)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jumbo_ik_demo.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"三维构型图已保存：{out}")


if __name__ == "__main__":
    res = solve_all()
    plot(res)
