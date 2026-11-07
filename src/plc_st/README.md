# plc_st —— PLC 程序源码（阶段 0c）

本目录存放可导入 TIA Portal 的 SCL 源码，算法与 `src/hydraulic_sim.py` 中**已通过软件在环验证**的 Python 版一一对应。

## 文件清单

| 文件 | 内容 | 对应 Python |
| --- | --- | --- |
| `FB_AxisPos.scl` | 单关节定位闭环 FB（斜坡+前馈+P+死区补偿+到位判断） | `hydraulic_sim.FB_AxisPos` |
| `FC_CalInterp.scl` | 标定表线性插值 FC（关节↔行程，60 点） | `cylinder_mapping.q_to_stroke / stroke_to_q` |
| `FB_HoleSeq.scl` | 自动寻孔流程状态机（S0~S99） | 总体设计 §8 |
| `IO表.md` | I/O 点表（AI/AO/HSC/DI/DO/通讯） | 总体设计 §7.2 |

## 导入 TIA Portal 步骤（学习版，无需硬件）

1. 安装 TIA Portal + PLCSIM（需西门子账号，版本建议 V17+，S7-1500 系列授权）；
2. 新建项目 → 添加 PLC（如 S7-1511C，固件任意）；
3. 添加新块 → SCL 函数块/函数 → 分别粘贴三个文件内容（注意：块名保持 `FB_AxisPos`/`FC_CalInterp`/`FB_HoleSeq`）；
4. 新建全局 DB `DB_Cal`：`qTab`/`sTab` 各 60 个 REAL——数据从 `src/cylinder_mapping.py` 导出（`cal["table"]`，运行 `python -c "..."` 打印后粘贴），实机阶段换实测表；
5. Main(OB1) 或循环中断 OB30（10ms）中调用：先 4 个 FB_AxisPos 实例，再 FB_HoleSeq（把 4 个 done 接到 sDone1~4）；
6. 下载到 PLCSIM 仿真运行，在监视表观察 step/u/done。

## 待办（按优先级）

1. **PLCSIM 导入编译通过**（语法若报错多为版本差异，改局部变量声明风格即可）；
2. 循环中断 OB30（10ms）里调用并验证 FB 时序（现在 FB 内部用固定 tCycle=0.01，实机改读 OB 参数）；
3. 把 `IO表.md` 的地址写进 HMI 变量表；
4. 台架阶段：按实测比例阀死区更新 `dzComp`、按实测流量更新 `kVff`、重拟标定表。

## 已知限制

- FB_HoleSeq 中 S40 开孔步骤是骨架（真实开孔闭环在工控机侧，V0.2 只做联锁）；
- 标定 DB 初始值未填（需从 Python 导出或实测）；
- 安全功能（急停/限位）是硬线+本文状态记录，安全等级按机械安全标准另行设计，不能仅依赖本文程序。
