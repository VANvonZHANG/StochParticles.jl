# 凝并、混合态与活化示例：初始条件、模拟设置及科学问题

## 1. 文档范围

本文档依据当前版本的以下 Julia 模拟脚本、Python 绘图脚本和 `Project.toml` 整理：

- `simulate_single_component_coagulation.jl` / `plot_single_component_coagulation.py`
- `simulate_mixing_state_coagulation.jl` / `plot_mixing_state_coagulation.py`
- `simulate_activation_coagulation_comparison.jl` / `plot_activation_coagulation_comparison.py`
- `simulation_io.jl`
- `Project.toml`

这里的“科学问题”分为两类：脚本注释和输出量直接表明的问题，以及根据对照设计和绘图指标推断出的研究意图。本文只说明当前代码实际表达的实验设计，不把尚未运行或尚未查看的模拟结果写成科学结论。

## 2. 三组模拟的总体逻辑

| 场景 | 核心过程 | 对象与尺度 | 主要诊断 | 核心问题 |
|---|---|---|---|---|
| 单组分凝并 | Brownian 凝并，或 Brownian + 重力 + 湍流凝并 | 硫酸盐气溶胶与水云滴 | 粒子数衰减、粒径谱迁移、有效数浓度损失率、核贡献 | 不同尺度下凝并速度和主导机制如何变化？ |
| SO₄/BC 混合态凝并 | Brownian 凝并 | 初始完全外混的硫酸盐与黑碳 | 混合态指数、粒径—BC 质量分数、CCN 与光学偏差 | 凝并如何驱动外混向内混，并改变均一内混近似的误差？ |
| 活化—凝并对照 | H₂O 凝结；与 H₂O 凝结 + 复合凝并成对比较 | Aitken 模态和较大干粒径模态 | 模态数浓度、湿粒径、活化比例、凝并核归因 | 云滴活化过程中，凝并是否会改变数浓度、谱演化和尺寸分辨活化？ |

三组模拟均采用粒子分辨表示，二元凝并后两个粒子的质量相加、活跃粒子数减一。因此，无源汇情况下凝并应守恒总质量，但会降低数浓度并使粒径分布向较大粒径移动。

## 3. 通用数值与输出设置

### 3.1 重复试验与随机种子

- 默认每个案例运行 20 个重复，可用环境变量 `STOCHPARTICLES_EXAMPLE_REPLICATES` 修改。
- 每个案例的所有重复共用同一个 `initial_seed`，即重复之间的初始粒子样本完全相同。
- 第 `i` 个重复使用 `process_seed = seed_base + i`，使随机凝并历程彼此不同。
- 单组分案例默认每次使用 1000 个模拟粒子，可用 `STOCHPARTICLES_SINGLE_COMPONENT_N_SIM` 修改；其计算体积与粒子数同比例变化，所以初始数浓度不变。
- 混合态和活化案例的模拟粒子数在配置结构中固定，当前没有对应的环境变量入口。

这种设计主要隔离“过程随机性”：重复之间的差异来自随机事件序列，而不是重新抽样初始粒径分布。它适合比较过程噪声，但不能同时估计初始粒子抽样不确定性。

### 3.2 数据输出

三个 Julia 脚本分别写出：

- `data/single_component_coagulation.h5`
- `data/mixing_state_coagulation.h5`
- `data/activation_coagulation_comparison.h5`

HDF5 结构版本为 `examples-v1`，按场景、案例和重复组织。公共输出包括：

- 时间、计算体积、活跃粒子数和数浓度；
- 总质量浓度及分物种质量浓度；
- 平均、中位和第 90 百分位粒径；
- 每个保存时刻的粒径样本；
- `dN/dlog10D` 原始分箱粒径谱。

各场景还写出混合态、干粒径、活化标志或凝并核贡献等专用诊断量。变长的粒子样本补齐为固定宽度矩阵，无活跃粒子的位置以 `NaN` 表示。

### 3.3 软件环境

`Project.toml` 要求 Julia 1.10，并声明 `StochParticles`、`StaticArrays`、`HDF5`、`OrdinaryDiffEq`、`DiffEqCallbacks` 和 `JumpProcesses`。当前 Python 绘图与分析还实际依赖 NumPy、Matplotlib、h5py 和 PyMieScatt；这些 Python 包不由 Julia 的 `Project.toml` 管理。

## 4. 单组分凝并模拟

### 4.1 案例 A：硫酸盐气溶胶 Brownian 凝并

#### 初始条件

| 参数 | 当前设置 |
|---|---|
| 物种 | SO₄，单组分 |
| 粒子密度 | 1800 kg m⁻³ |
| 模拟粒子数 | 默认 1000 |
| 计算体积 | `2.5e-12 × n_sim` m³；默认 2.5 × 10⁻⁹ m³ |
| 总数浓度 | 4.0 × 10¹¹ m⁻³（4.0 × 10⁵ cm⁻³） |
| 小粒径模态 | 50% 粒子；几何平均直径 5 nm；几何标准差 1.5 |
| 大粒径模态 | 50% 粒子；几何平均直径 100 nm；几何标准差 1.5 |
| 两模态数浓度 | 各 2.0 × 10¹¹ m⁻³ |
| 初始随机种子 | 2026072100 |

两个模态分别从对数正态**直径**分布采样，再按球形粒子和给定密度换算为质量。

#### 模拟设置

| 参数 | 当前设置 |
|---|---|
| 温度 | 293.15 K |
| 压力 | 101325 Pa |
| 凝并核 | `BrownianKernel`，仅 Brownian 凝并 |
| 数值方法 | 基于全部粒子对核值的直接随机模拟算法（direct SSA，非 CNMC） |
| 模拟时段 | 0–3600 s |
| 保存频率 | 0–720 s 每 10 s；之后每 60 s，首个后段时刻为 780 s |
| 粒径谱边界 | 1 nm–10 μm，共 80 个对数等距区间 |

#### 绘图诊断与科学问题

绘图展示 `N/N0`、全时段及早期粒径谱热图、初末粒径分布与最大粒径，以及相对有效数浓度损失率。这里的有效损失率定义为

\[
k_{\mathrm{eff}}=-\frac{\ln(N_f/N_0)}{t_f-t_0}.
\]

该案例主要想回答：

1. 在纳米级双峰气溶胶中，Brownian 凝并在一小时内以多快的速度降低粒子数？
2. 凝并是否使双峰粒径谱展宽、并向较大粒径迁移？早期变化是否比后期更快？
3. 该气溶胶案例可否作为仅含 Brownian 机制的基准，用于对照云滴尺度的复合凝并？

### 4.2 案例 B：水云滴复合凝并

#### 初始条件

| 参数 | 当前设置 |
|---|---|
| 物种 | H₂O，单组分 |
| 粒子密度 | 1000 kg m⁻³ |
| 模拟粒子数 | 默认 1000 |
| 计算体积 | `5.0e-9 × n_sim` m³；默认 5.0 × 10⁻⁶ m³ |
| 总数浓度 | 2.0 × 10⁸ m⁻³（200 cm⁻³） |
| 小云滴模态 | 50% 粒子；几何平均直径 5 μm；几何标准差 1.3 |
| 大云滴模态 | 50% 粒子；几何平均直径 25 μm；几何标准差 1.3 |
| 两模态数浓度 | 各 1.0 × 10⁸ m⁻³（100 cm⁻³） |
| 初始随机种子 | 2026073100 |

#### 模拟设置

| 参数 | 当前设置 |
|---|---|
| 温度、压力 | 288.15 K，80000 Pa |
| 空气密度 | 1.06 kg m⁻³ |
| 动力/运动黏度 | 1.75 × 10⁻⁵ Pa s / 1.65 × 10⁻⁵ m² s⁻¹ |
| 重力加速度 | 9.81 m s⁻² |
| 湍流耗散率 `epsilon` | 0.01 m² s⁻³ |
| Taylor 微尺度 Reynolds 数 `R_lambda` | 50 |
| 凝并核 | Brownian + 重力差异沉降 + Ayala 湍流核之和 |
| 数值方法 | 直接 SSA，非 CNMC |
| 模拟时段与保存频率 | 0–200 s，每 4 s |
| 粒径谱边界 | 1 μm–1 mm，共 80 个对数等距区间 |

脚本在每个保存时刻对所有活跃粒子对分别求和 Brownian、重力和湍流核，再用三者之和归一化，得到瞬时核贡献比例。绘图还给出各分量的时间平均比例。

#### 科学问题

1. 在典型云滴浓度和双峰云滴谱下，复合凝并能否在数分钟内显著降低云滴数并产生更大液滴？
2. Brownian、重力差异沉降和湍流碰撞在不同演化阶段各占多大比例？
3. 从纳米气溶胶到微米云滴，粒径、浓度与核机制的共同变化会使有效数浓度损失时间尺度相差多少？

“Relative effective kernel strength”面板实际比较的是上述 `k_eff`，并以气溶胶案例的重复平均值归一化。它同时受初始粒径谱、数浓度、模拟时段和凝并核影响，不应解释为在相同粒径与环境下对单个粒子对核函数的直接比值。

## 5. SO₄/BC 混合态凝并模拟

![SO₄/BC 混合态凝并与均一内混近似示意图](schematics/mixing_state_coagulation_schematic.png)

*图 5-1｜左侧表示外混 SO₄/BC 粒子经 Brownian 凝并形成部分内混群体；右侧比较同一时刻的粒子分辨表示与均一内混后处理近似。均一内混粒子不是另一套动力学模拟粒子：其数量和外径与粒子分辨群体逐一对应，只将组成相关参数替换为群体总体值。*

### 5.1 初始条件

| 参数 | 当前设置 |
|---|---|
| 物种 | SO₄ 与 BC |
| 模拟粒子数 | 4000 |
| 计算体积 | 1.0 × 10⁻⁷ m³ |
| 总数浓度 | 4.0 × 10¹⁰ m⁻³（4.0 × 10⁴ cm⁻³） |
| SO₄ 模态 | 2000 个纯 SO₄ 粒子；几何平均直径 100 nm；几何标准差 1.5 |
| BC 模态 | 2000 个纯 BC 粒子；几何平均直径 50 nm；几何标准差 1.3 |
| 两模态数浓度 | 各 2.0 × 10¹⁰ m⁻³ |
| SO₄/BC 密度 | 1770/1800 kg m⁻³ |
| 初始混合状态 | 完全外混：每个粒子不是纯 SO₄，就是纯 BC |
| 初始随机种子 | 2026092100 |

### 5.2 模拟设置

| 参数 | 当前设置 |
|---|---|
| 温度、压力 | 298 K，101325 Pa |
| 凝并过程 | 仅 Brownian 凝并 |
| 数值方法 | 带全粒子对核缓存和行和缓存的直接 SSA，非 CNMC |
| 模拟时段 | 0–24 h |
| 保存频率 | 0–2 h 每 10 min；之后每 1 h，首个后段时刻为 3 h |
| 粒径谱边界 | 1 nm–1 μm，共 80 个对数等距区间 |

异类粒子凝并后，其 SO₄ 和 BC 质量向量逐分量相加，因而产生内部混合粒子；同类凝并则只改变尺寸和数浓度。模拟本身没有凝结、化学反应、排放、稀释或沉降。

### 5.3 混合态指标

脚本直接保存每个粒子的 BC 质量分数以及总体混合态指数 `chi`。当前代码中的定义为

\[
\chi=\frac{\overline{H(f_i)}}{H(\bar f)},
\]

其中 `H` 是基于物种质量分数的 Shannon 熵；分子是各粒子熵的**按粒子数平均**，分母使用总体**质量加权**平均组成的熵。初始纯粒子外混状态给出 `chi = 0`；若所有粒子具有相同的总体组成，则 `chi` 趋近 1。

粒径—组成图将等体积球径与 BC 质量分数组成二维 KDE，用初始和 24 h 时刻的分布直观展示从 `f_BC = 0/1` 两条外混分支向中间组成的迁移。

### 5.4 CCN 与光学后处理

这两类“误差”是**简化的均一内混假设所导致的参数化偏差**，不是模拟的数值误差。

#### 5.4.1 “均一内混计算粒子”究竟是什么

均一内混计算没有在 Julia 中重新初始化或推进第二套粒子系统。Julia 模拟只产生一套真实的粒子分辨轨迹；所谓 uniform mixing 是 `examples/analysis/mixing_state_analysis.py` 在绘图时针对**每个重复、每个保存时刻**即时构造的临时 NumPy 对照数组。它不使用新的随机数、不发生额外凝并、不写回 HDF5，也不会参与下一时刻的模拟。

对某个保存时刻 $t$，算法首先从 HDF5 读取：

- `diameter_samples[t, :]`：模拟粒子的等体积干直径；
- `bc_mass_fraction_samples[t, :]`：同一批粒子的 BC **质量分数**。

函数 `_active_samples` 保留满足 $D_i>0$、$0\le f_{\mathrm{BC},i}^{m}\le1$ 且数值有限的对应元素，去掉 inactive slot 使用的 `NaN` 填充值。清洗后得到 $N_t$ 个实际活跃粒子：

$$
\left\{D_i, f_{\mathrm{BC},i}^{m}\right\}_{i=1}^{N_t}.
$$

uniform-mixing 对照与这 $N_t$ 个实际粒子建立一一对应关系：

| 属性 | 粒子分辨基准 | 均一内混对照 |
|---|---|---|
| 粒子数 | $N_t$ | 同一个 $N_t$ |
| 第 $i$ 个粒子的总直径 | $D_i$ | 原样复制 $D_i$ |
| 第 $i$ 个粒子的实际组成 | 保留 | 不保留粒子间差异 |
| 统一替换的属性 | 无 | CCN 使用 $\kappa_{\mathrm{bulk}}$；光学使用 $f_{\mathrm{BC,bulk}}^V$ |
| 是否进入下一时刻 | 是，来自 Julia 动力学轨迹 | 否，仅用于当前时刻诊断 |

因此，“生成 uniform-mixing 粒子”在代码中的准确含义是：**复制当前实际粒子的粒径数组，在内存中将每颗粒子的组成诊断参数替换为总体平均值**，而不是重新抽样一套具有新尺寸分布的粒子。

#### 5.4.2 从质量分数反算每颗粒子的组分体积

HDF5 保存的是 BC 质量分数，但 $\kappa$ 混合规则和核壳几何都需要体积。代码使用 $\rho_{\mathrm{SO_4}}=1770\ \mathrm{kg\,m^{-3}}$、$\rho_{\mathrm{BC}}=1800\ \mathrm{kg\,m^{-3}}$，对每颗粒子执行体积加和反演。

令

$$
V_i=\frac{\pi D_i^3}{6},\qquad
f_{\mathrm{SO_4},i}^{m}=1-f_{\mathrm{BC},i}^{m},
$$

则总干质量为

$$
m_i=
\frac{V_i}
{f_{\mathrm{SO_4},i}^{m}/\rho_{\mathrm{SO_4}}
+f_{\mathrm{BC},i}^{m}/\rho_{\mathrm{BC}}},
$$

两种组分体积为

$$
V_{\mathrm{SO_4},i}=m_i\frac{f_{\mathrm{SO_4},i}^{m}}{\rho_{\mathrm{SO_4}}},
\qquad
V_{\mathrm{BC},i}=m_i\frac{f_{\mathrm{BC},i}^{m}}{\rho_{\mathrm{BC}}}.
$$

数值上有 $V_i=V_{\mathrm{SO_4},i}+V_{\mathrm{BC},i}$。这一步由 `particle_species_volumes` 同时服务于 CCN 和光学两条诊断路径。

#### 5.4.3 CCN 对照粒子的构造与计算

SO₄ 和 BC 的吸湿性参数分别取 $\kappa_{\mathrm{SO_4}}=0.61$ 和 $\kappa_{\mathrm{BC}}=0$；这些值只用于 Python 后处理。

1. **粒子分辨基准。** 每颗实际粒子的体积加权吸湿性为

   $$
   \kappa_i^{\mathrm{actual}}=
   \frac{\kappa_{\mathrm{SO_4}}V_{\mathrm{SO_4},i}
   +\kappa_{\mathrm{BC}}V_{\mathrm{BC},i}}{V_i}.
   $$

2. **计算群体总体吸湿性。** 代码先对组分体积求和，再计算

   $$
   \kappa_{\mathrm{bulk}}=
   \frac{\kappa_{\mathrm{SO_4}}\sum_i V_{\mathrm{SO_4},i}
   +\kappa_{\mathrm{BC}}\sum_i V_{\mathrm{BC},i}}
   {\sum_i V_i}.
   $$

3. **构造均一内混对照。** 创建长度为 $N_t$ 的常数数组，并令

   $$
   D_i^{\mathrm{uniform}}=D_i,
   \qquad
   \kappa_i^{\mathrm{uniform}}=\kappa_{\mathrm{bulk}}
   \quad(i=1,\ldots,N_t).
   $$

   因而只消除了粒子间的 $\kappa$ 差异，粒径和粒子数完全不变。CCN 计算只需要 $D_i$ 和 $\kappa_i$，所以这里不额外指定 uniform-mixing 粒子的内部几何结构。

4. **分别求临界过饱和度。** `critical_supersaturations` 对实际数组 $(D_i,\kappa_i^{\mathrm{actual}})$ 和对照数组 $(D_i,\kappa_{\mathrm{bulk}})$ 分别最大化精确 $\kappa$-Köhler 曲线。代码使用逐粒子向量化的黄金分割搜索，迭代 100 次。

5. **统计 CCN 并计算误差。** 在环境过饱和度 $S=0.1\%,0.3\%,1\%$ 下，满足 $S_{c,i}\le S$ 的粒子计为 CCN：

   $$
   \varepsilon_{\mathrm{CCN}}=
   \frac{N_{\mathrm{CCN}}^{\mathrm{uniform}}
   -N_{\mathrm{CCN}}^{\mathrm{actual}}}
   {N_{\mathrm{CCN}}^{\mathrm{actual}}}.
   $$

   当 $N_{\mathrm{CCN}}^{\mathrm{actual}}=0$ 时，代码返回 `NaN`。

#### 5.4.4 光学对照粒子的构造与计算

光学计算使用 $\lambda=550\ \mathrm{nm}$，SO₄ 和 BC 的复折射率分别为 `1.53 + 0i` 和 `1.95 + 0.79i`。

1. **粒子分辨基准。** 对每颗实际粒子：

   - $V_{\mathrm{BC},i}=0$ 时，使用纯 SO₄ 均质球 `MieQ`；
   - $V_{\mathrm{SO_4},i}=0$ 时，使用纯 BC 均质球 `MieQ`；
   - 两种体积均非零时，构造成 BC 核/SO₄ 壳，并使用 `MieQCoreShell`。

   实际混合粒子的核直径为

   $$
   D_{\mathrm{core},i}^{\mathrm{actual}}
   =D_i\left(\frac{V_{\mathrm{BC},i}}{V_i}\right)^{1/3}.
   $$

2. **计算总体 BC 体积分数。** 代码按整个粒子群的总体积求

   $$
   f_{\mathrm{BC,bulk}}^V=
   \frac{\sum_i V_{\mathrm{BC},i}}{\sum_i V_i}.
   $$

3. **构造均一内混光学粒子。** 第 $i$ 个对照粒子继续使用实际外径 $D_i$，但将其组分体积替换为

   $$
   V_{\mathrm{BC},i}^{\mathrm{uniform}}
   =f_{\mathrm{BC,bulk}}^V V_i,
   \qquad
   V_{\mathrm{SO_4},i}^{\mathrm{uniform}}
   =(1-f_{\mathrm{BC,bulk}}^V)V_i.
   $$

   所有对照粒子仍使用 BC 核/SO₄ 壳 `MieQCoreShell`，其核直径为

   $$
   D_{\mathrm{core},i}^{\mathrm{uniform}}
   =D_i\left(f_{\mathrm{BC,bulk}}^V\right)^{1/3}.
   $$

   因此所有对照粒子的**外径仍各不相同**，但 $D_{\mathrm{core},i}/D_i$ 完全相同。代码没有将两种折射率混合成一个有效折射率，也没有把对照粒子当作均质灰色球。

4. **计算总体光学量和误差。** 每颗粒子的效率因子乘以几何截面积 $\pi(D_i/2)^2$ 得到吸收与散射截面；随后分别对所有粒子求和。代码报告

   $$
   \varepsilon_{\mathrm{abs}}=
   \frac{\sum_i C_{\mathrm{abs},i}^{\mathrm{uniform}}
   -\sum_i C_{\mathrm{abs},i}^{\mathrm{actual}}}
   {\sum_i C_{\mathrm{abs},i}^{\mathrm{actual}}},
   \qquad
   \varepsilon_{\mathrm{sca}}=
   \frac{\sum_i C_{\mathrm{sca},i}^{\mathrm{uniform}}
   -\sum_i C_{\mathrm{sca},i}^{\mathrm{actual}}}
   {\sum_i C_{\mathrm{sca},i}^{\mathrm{actual}}}.
   $$

   若相应的实际总截面为 0，则误差记为 `NaN`。

#### 5.4.5 与代码变量名和当前数据的对应

上述算法可以压缩为以下伪代码：

```text
for each replicate:
    for each saved time t:
        D, f_BC_mass = active_samples(t)
        V_SO4, V_BC = particle_species_volumes(D, f_BC_mass)

        actual CCN   = Koehler(D, particle-specific kappa)
        uniform CCN  = Koehler(D, kappa_bulk repeated N_t times)

        actual optics  = Mie(D, particle-specific V_BC and V_SO4)
        uniform optics = core-shell Mie(D, bulk BC fraction repeated N_t times)

        error = (uniform - actual) / actual
```

代码中的 `sc_internal`、`n_internal`、`c_abs_i` 和 `c_sca_i` 表示上述 uniform-mixing/internal-mixture approximation；它们不是另一套动力学模拟的输出。

以当前 `data/mixing_state_coagulation.h5` 的 `replicate_001` 为数值核验：$t=0$ 和 $t=24\ \mathrm{h}$ 的总体 BC 体积分数均为

$$
f_{\mathrm{BC,bulk}}^V\approx0.06944,
$$

对应均一内混光学粒子的统一核径比

$$
\frac{D_{\mathrm{core}}}{D}\approx(0.06944)^{1/3}\approx0.411.
$$

相应的 $\kappa_{\mathrm{bulk}}\approx0.56764$。这些数值由当前数据在每个时刻重新计算，不是写死在算法中的常数；在本模拟中凝并守恒各组分质量，所以总体组成在数值精度内随时间保持不变。

### 5.5 科学问题

1. Brownian 凝并能以多快的速度把完全外混的 SO₄/BC 群体转化为部分内混群体？
2. 粒子数衰减、粒径增长与混合态指数增长之间是否存在不同的时间尺度？
3. 随着真实粒子群逐渐内混，把所有粒子直接视为均一内混所造成的 CCN 数浓度误差如何随时间和过饱和度变化？
4. 同一内混近似对 550 nm 吸收和散射的偏差是否不同，且是否随着凝并混合而减小？
5. 最终的尺寸—组成相关性是否仍保留初始纯 SO₄/纯 BC 模态的信息？

## 6. 活化与凝并成对对照模拟

### 6.1 初始干气溶胶

| 参数 | 当前设置 |
|---|---|
| 物种 | SO₄ 与 H₂O |
| 模拟粒子数 | 1000 |
| Aitken 模态 | 800 个粒子；干几何平均直径 20 nm；几何标准差 1.25；8.4 × 10¹¹ m⁻³ |
| 较大干粒径模态 | 200 个粒子；干几何平均直径 200 nm；几何标准差 1.4；2.1 × 10¹¹ m⁻³ |
| 总数浓度 | 1.05 × 10¹² m⁻³（1.05 × 10⁶ cm⁻³） |
| 计算体积 | `1000 / 1.05e12` = 9.5238 × 10⁻¹⁰ m³ |
| 初始干组成 | 所有粒子均为纯 SO₄，H₂O 质量初始为 0 |
| SO₄/H₂O 密度 | 1770/1000 kg m⁻³ |
| SO₄/H₂O `kappa` | 0.61/0（Petters & Kreidenweis, 2007） |
| 初始随机种子 | 2026072200 |

干粒子生成后，脚本先在目标温度和水汽压下调用 `pre_equilibrate!`，为每个粒子加入平衡水量。因此动态模拟的 `t = 0` 状态是预平衡后的湿粒子，而保存的 `dry_diameter_initial` 仍表示加水前的 SO₄ 干粒径。

### 6.2 环境与动力学设置

| 参数 | 当前设置 |
|---|---|
| 温度、压力 | 293.15 K，101325 Pa |
| 固定水汽过饱和度 | 0.005，即 0.5% |
| 气相边界 | 温度和水汽压随时间固定；垂直速度 `w = 0` |
| 模拟时段 | 0–600 s |
| 输出间隔 | 10 s |
| 过程分裂步长 | `dt_split = 10 s` |
| 凝结积分器 | `Tsit5` |
| 活化判据 | 湿半径不小于 1 μm，即湿直径不小于 2 μm |
| 动态模态阈值 | 当前干直径 60 nm；小于阈值为 Aitken，大于等于阈值为“大粒径/云滴”类 |
| 湍流参数 | `epsilon = 0.01 m² s⁻³`，`R_lambda = 50` |
| 粒径谱边界 | 约 5.01 nm–31.6 μm，共 95 个对数等距区间 |

水汽状态是外部给定且不因粒子凝结而耗尽，所以这是固定过饱和度下的理想化活化实验，而不是封闭气块中的水汽守恒模拟。

### 6.3 成对对照设计

每个重复由完全相同的预平衡粒子初态构造两个情景：

| 情景 | 过程 |
|---|---|
| `activation_only` | 仅 H₂O 凝结 |
| `activation_with_coagulation` | H₂O 凝结 + Brownian、重力和 Ayala 湍流复合凝并 |

含凝并情景使用 `NonCNMCCoagulationProcess(total_kernel, LocalMajorant())`，并与凝结过程通过 `solve_split` 耦合。两个情景记录相同的基础诊断，含凝并情景额外记录三类粒子对的核贡献：

- Aitken–Aitken；
- 大粒径–大粒径；
- Aitken–大粒径交叉对。

每一类内部再区分 Brownian、重力和湍流贡献。

需要注意：这里的模态 ID 按**当前 SO₄ 质量对应的干直径**和 60 nm 阈值重新计算，并非永久记录粒子的初始来源。凝并后越过阈值的粒子会改变类别。因此绘图中的 “Droplet (200 nm)” 更准确地说是“当前干直径不小于 60 nm 的大粒径类”，不等同于只追踪初始 200 nm 模态。

### 6.4 绘图诊断与科学问题

绘图包括：

- 两个动态粒径类各自的 `N/N0`；
- 平均湿粒径随时间的变化；
- 三类粒子对中 Brownian/重力/湍流核的贡献比例；
- 最终时刻按当前干粒径分箱的活化比例；
- 两情景湿粒径谱的时间演化热图。

该成对对照主要想回答：

1. 在固定 0.5% 过饱和度下，仅凝结与凝结+凝并对粒子数和平均湿粒径的演化有何差异？
2. 凝并是否通过移除小粒子、形成更大的含 SO₄ 粒子而改变最终的尺寸分辨活化曲线？
3. Aitken–Aitken、大粒径–大粒径和跨模态粒子对分别由哪种凝并机制主导，其主导关系是否随活化和湿增长而变化？
4. 凝并是否会改变湿粒径谱中从未活化气溶胶到云滴的数浓度分配和过渡结构？
5. 相同初态下两个情景的差值能否隔离凝并对活化动力学的增量影响？

## 7. 跨场景可以形成的科学叙事

这三组示例构成一个由简单到复杂的过程链：

1. **单组分、单过程基准**：先确认 Brownian 凝并如何改变粒子数和粒径谱。
2. **跨尺度和多凝并机制**：转到云滴尺度，比较 Brownian、重力与湍流的竞争。
3. **多组分混合态**：在保持凝并机制简单的情况下，研究粒子组成如何因凝并发生重排，以及均一内混假设对 CCN/光学量的影响。
4. **微物理过程耦合**：将凝并与水汽凝结、气溶胶活化同时计算，用成对情景分离凝并的附加作用。

因此，这些脚本不仅是数值示例，还围绕同一条主线展开：**凝并如何同时改变粒子数、尺寸、组成与云微物理响应，以及这种影响如何随粒径尺度和环境条件改变。**

## 8. 解释结果时的边界与注意事项

- 三组场景没有排放、稀释、沉降清除或化学反应；结果只代表所启用微物理过程造成的演化。
- 除活化案例的水凝结外，粒子直径增长来自凝并质量合并，不代表凝结生长。
- 所有重复共享同一初始粒径样本，重复间离散度不能代表初始分布抽样不确定性。
- 混合态案例中的 `chi` 使用粒子数平均的单粒子熵和质量加权总体组成；与采用质量加权粒子平均或指数化 diversity 的其他文献定义比较时，应先统一定义。
- 混合态 CCN 与光学诊断依赖 Python 后处理中指定的 `kappa`、折射率、核壳形态和均一内混对照假设；它们并非 Julia 动力学模拟中的额外物理过程。
- 活化案例使用固定水汽压，不考虑凝结导致的水汽耗竭；其高粒子数浓度下的结果应理解为理想化敏感性实验。
- 活化判据是固定湿半径阈值，不是直接用 Köhler 临界点判定。
- 核贡献比例是对当前所有可能粒子对的核值求和后归一化，表示潜在碰撞率构成；它不等同于已经发生的碰撞事件中各机制的事后标签。
- 绘图热图使用 KDE 平滑；它适合展示谱结构，但定量守恒检查应使用原始 `dN/dlog10D`、数浓度和质量浓度数据。

## 9. 文件与职责对应

| 文件 | 职责 |
|---|---|
| `simulate_single_component_coagulation.jl` | 构造气溶胶/云滴初态，执行直接 SSA 凝并并输出核贡献 |
| `plot_single_component_coagulation.py` | 比较粒子数衰减、谱迁移、有效损失率与复合核构成 |
| `simulate_mixing_state_coagulation.jl` | 执行 SO₄/BC Brownian 凝并并输出粒子组成和混合态指数 |
| `plot_mixing_state_coagulation.py` | 绘制混合态演化，并计算 CCN/光学内混近似偏差 |
| `simulate_activation_coagulation_comparison.jl` | 构造活化-only 与活化+凝并的成对过程模拟 |
| `plot_activation_coagulation_comparison.py` | 比较模态数浓度、湿粒径、核归因和尺寸分辨活化 |
| `simulation_io.jl` | 统一重复数、公共诊断量及 HDF5 写出格式 |
| `Project.toml` | 声明 Julia 1.10 和示例所需 Julia 依赖 |
