# Go2 stand: наблюдения и функции награды

Окружение обучает Unitree Go2 вставать на задние лапы и выполнять команды
продольной скорости $c_x$, боковой скорости $c_y$ и угловой скорости $c_\omega$.
Текущая конфигурация зафиксирована как минимальный **Baseline v1**.

## История наблюдений actor

Линейная скорость корпуса не входит в actor observation, поскольку она не
измеряется напрямую на реальном роботе. Она остаётся доступной симулятору для
вычисления reward и offline-метрик.

Один кадр содержит 47 значений:

$$
o_t = [\omega_b(3),\ g_b(3),\ c(3),\ q-q_0(12),\quad
       \dot q(12),\ a_{t-1}(12),\ \sin(2\pi\phi),\ \cos(2\pi\phi)].
$$

Actor получает стек от старого кадра к новому:

$$
O_t=[o_{t-K+1},\ldots,o_t],\qquad \dim O_t=47K.
$$

Длина задаётся в `go2_config.py`:

```python
class env(LeggedRobotCfg.env):
    num_single_observations = 47
    history_length = 5
    num_observations = num_single_observations * history_length
```

При policy step $\Delta t=0.02$ с история из $K=5$ кадров покрывает
$(K-1)\Delta t=0.08$ с. Значение `history_length=1` отключает историю. После
reset все позиции истории заполняются первым валидным кадром; observation noise
добавляется только новому кадру.

Изменение `history_length` меняет размер входа сети, поэтому требует нового
обучения и нового checkpoint.

В deployment-коде нужно поддерживать такой же rolling buffer, тот же порядок
признаков и порядок кадров «старый $\rightarrow$ новый». При старте реального
контроллера первый измеренный кадр также следует повторить $K$ раз.

## Суммирование наград

Isaac Gym использует

$$
R_t^{\mathrm{pre}}=\sum_i \Delta t\,w_i r_i(t),
\qquad
\Delta t=\texttt{sim.dt}\cdot\texttt{decimation}=0.005\cdot4=0.02.
$$

При `only_positive_rewards=True`:

$$
R_t=\max(0,R_t^{\mathrm{pre}})
+\Delta t\,w_{\mathrm{termination}}r_{\mathrm{termination}}.
$$

Следовательно, `scale` из config является приблизительным вкладом reward в
секунду, а фактический коэффициент одного policy step равен $0.02w_i$.

## Stand-up и locomotion stages

Первые $T_{\mathrm{stand}}=1.25$ с робот только встаёт. Затем locomotion-rewards
включаются за $T_{\mathrm{transition}}=0.30$ с с коэффициентом

$$
\beta(t)=\operatorname{clip}\left(
\frac{t-T_{\mathrm{stand}}}{T_{\mathrm{transition}}},0,1
\right).
$$

`tracking_lin_vel` и `tracking_ang_vel` умножаются на $\beta$. Контактное
чередование и `feet_clearance` умножаются также на признак locomotion-команды.
До этого обе задние стопы должны находиться в опоре, а контакт передних стоп и
остальных нежелательных звеньев штрафуется.

Линейная и угловая команды классифицируются раздельно:

$$
I_{\mathrm{loc}}=
\mathbb{1}[\lVert(c_x,c_y)\rVert_2>0.20]
\lor
\mathbb{1}[|c_\omega|>0.10].
$$

Здесь пороги имеют разные единицы: $0.20$ м/с и $0.10$ рад/с. Малые команды
обнуляются sampler-ом с теми же порогами. Период gait равен $0.40$ с, то есть
ровно $0.40/0.02=20$ policy steps. Фаза locomotion начинается после stand-up;
до него observation содержит $\sin\phi=0$, $\cos\phi=1$.

## Формулы custom rewards

### Ориентация `tracking_pitch`

Target pitch плавно изменяется от нуля до $\theta_f=-1.57$:

$$
\theta^*(t)=\operatorname{clip}
\left(\frac{t\theta_f}{T_{\mathrm{stand}}},\min(0,\theta_f),\max(0,\theta_f)\right).
$$

Для текущих Euler pitch $\theta$ и roll $\rho$:

$$
r_{\mathrm{pitch}}=\exp\left(
-\frac{(\theta^*-\theta)^2+(\rho^*-\rho)^2}{\sigma_{\mathrm{tracking}}}
\right).
$$

### Линейная скорость `tracking_lin_vel`

Скорость используется только в симуляционной награде. Она проецируется в
горизонтальную систему вертикального робота, продольная ось которой соответствует
$-Z$ корпуса:

$$
r_v'=\beta(t)\exp\left(-\frac{(c_x-v_x)^2+(c_y-v_y)^2}
{\sigma_{\mathrm{tracking}}}\right).
$$

### Yaw rate `tracking_ang_vel`

Используется угловая скорость вокруг мировой вертикали $\omega_z^W$:

$$
r_\omega'=\beta(t)\exp\left(-\frac{(c_\omega-\omega_z^W)^2}
{\sigma_{\mathrm{tracking}}}\right).
$$

### Положение hip joints `hip_pos`

$$
r_{\mathrm{hip}}=\exp\left(-\frac{
\sum_{j\in\mathcal H}(q_j-q_j^0)^2}{\sigma_{\mathrm{tracking}}}\right).
$$

### Высота корпуса `base_height`

$$
r_h=\exp\left[-\left(\frac{h-h^*}{\sigma_h}\right)^2\right],
\qquad h^*=0.50\ \text{м}.
$$

### CoM над опорой `com_over_support`

Для каждого звена используется его локальное смещение центра масс $c_i$, текущая
мировая поза $(p_i,R_i)$ и рандомизированная масса $m_i$:

$$
p_i^{CoM}=p_i+R_ic_i,
\qquad
p^{CoM}=\frac{\sum_i m_i p_i^{CoM}}{\sum_i m_i}.
$$

Если обе задние стопы контактируют, опорная область аппроксимируется отрезком
между $p_{RR}^{xy}$ и $p_{RL}^{xy}$, расширенным на `com_support_margin`. Если
контактирует одна стопа, используется её точка с тем же расширением. Обозначим
через $d$ расстояние проекции CoM до ближайшей точки этой опоры:

$$
e_{\mathrm{support}}=\max(d-m,0),
$$

$$
r_{\mathrm{CoM}}=\exp\left[-\left(
\frac{e_{\mathrm{support}}}{\sigma_{\mathrm{support}}}
\right)^2\right]\mathbb{1}[N_{\mathrm{contact}}>0].
$$

Текущие параметры:

```python
com_support_margin = 0.03  # м
com_support_sigma = 0.08   # м
```

Внутри опорной области reward равен 1. На расстоянии одной sigma за её границей
он равен $e^{-1}\approx0.368$, двух sigma — $e^{-4}\approx0.018$; без задних
контактов reward равен 0. Высота не входит в эту функцию, поскольку отдельно
контролируется `base_height`.

### Контактный шаблон `rear_feet_contact_and_air`

$$
C_j=\mathbb{1}[F_j^z>F_{\mathrm{rear}}],
$$

$$
r_{\mathrm{match}}=\sum_j
\left(C_jG_j+(1-C_j)(1-G_j)\right),
$$

$$
r_{\mathrm{contact}}=(1-\beta I_{\mathrm{loc}})
\sum_j C_j+\beta I_{\mathrm{loc}}r_{\mathrm{match}}
-15\sum_{k\in\mathcal U}\mathbb{1}[\lVert F_k\rVert_2>F_{\mathrm{bad}}].
$$

$G_j=1$ обозначает stance. При нулевой команде обе задние лапы имеют stance.
Reward не изменяет историю контактов: она обновляется один раз до вычисления всех
наград и очищается при reset.

### Клиренс `feet_clearance`

Только для swing-лап и ненулевой команды:

$$
r_{\mathrm{clear}}'=\beta I_{\mathrm{loc}}
\sum_j(1-G_j)\exp\left[-\left(
\frac{h_j-h_{\mathrm{foot}}^*}{\sigma_{\mathrm{clear}}}
\right)^2\right].
$$

### Проскальзывание `foot_slip`

Функция возвращает положительный raw penalty, поэтому её scale отрицательный:

$$
p_{\mathrm{slip}}=\sum_j C_j\lVert v_j^{xy}\rVert_2^2.
$$

### Плавность `smoothness`

$$
p_s=0.5\lVert a_t-a_{t-1}\rVert_2^2
+0.5\lVert a_t-2a_{t-1}+a_{t-2}\rVert_2^2
+0.1\lVert a_t\rVert_1.
$$

`low_speed` является небольшим дополнительным кусочно-постоянным сигналом для
продольной команды; основной контроль скорости выполняет `tracking_lin_vel`.

При $|c_x|>c_{\mathrm{dead}}$:

$$
r_{\mathrm{low}}=\begin{cases}
-2, & \operatorname{sign}(v_x)\ne\operatorname{sign}(c_x),\\
-1, & |v_x|<0.5|c_x|,\\
0, & |v_x|>1.2|c_x|,\\
1.2, & \text{иначе}.
\end{cases}
$$

## Формулы унаследованных penalties

Эти функции определены в `envs/base/legged_robot.py` и возвращают
неотрицательный raw penalty; отрицательный scale превращает его в штраф:

$$
p_\tau=\sum_j\tau_j^2,
\qquad
p_{\dot q}=\sum_j\dot q_j^2,
$$

$$
p_{\ddot q}=\sum_j\left(\frac{\dot q_j^- - \dot q_j}{\Delta t}\right)^2,
$$

$$
p_{\mathrm{collision}}=\sum_{k\in\mathcal P}
\mathbb{1}[\lVert F_k\rVert_2>0.1],
$$

$$
p_{\mathrm{vel\text{-}limit}}=\sum_j
\operatorname{clip}
\left(|\dot q_j|-\dot q_j^\mathrm{soft},0,1\right),
$$

$$
p_{\mathrm{contact\text{-}force}}=\sum_j
\max\left(\lVert F_j\rVert_2-F_{\mathrm{max}},0\right),
$$

$$
p_{\mathrm{termination}}=
\mathbb{1}[\text{reset не по timeout}].
$$

## Domain randomization контроллера

Для position control target и эффективные gains имеют вид

$$
q_{\mathrm{target}}=q_0+s_a a+\Delta q_{\mathrm{zero}},
$$

$$
K_p^{\mathrm{eff}}=K_p k_p,
\qquad
K_d^{\mathrm{eff}}=K_d k_d,
$$

$$
\tau=K_p^{\mathrm{eff}}(q_{\mathrm{target}}-q)
-K_d^{\mathrm{eff}}\dot q.
$$

$\Delta q_{\mathrm{zero}}$ измеряется в радианах и поэтому не умножается на
`action_scale`. При выключенной randomization множители равны единице, offset
равен нулю, и контроллер совпадает с исходным.

Перед длинным обучением randomization можно проверить командой:

```bash
python legged_gym/scripts/debug_go2_randomization.py \
    --task=go2_stand --headless
```

Она выводит значения нескольких env и проверяет torque на одинаковых
$q$, $\dot q$ и action.

## TensorBoard debug metrics

Стандартный runner уже пишет `Train/mean_reward`,
`Train/mean_episode_length` и отдельные активные reward terms: velocity,
pitch, CoM-support, rear gait и clearance. Дополнительно Baseline v1 пишет:

- `baseline_total_reward_per_step`;
- `front_feet_contact_rate`, `front_feet_contact_penalty_raw`;
- `mean_abs_torque`, `mean_abs_action`;
- `com_support_distance_m`;
- `forward_velocity_abs_error_m_s`, `lateral_velocity_abs_error_m_s`;
- `episode_length_s`, `terminated_count`, `termination_fraction`.

## Текущие scales

| Reward | Scale $w_i$ | Коэффициент за шаг $0.02w_i$ |
|---|---:|---:|
| `tracking_lin_vel` | 2.5 | 0.05 |
| `tracking_ang_vel` | 1.5 | 0.03 |
| `tracking_pitch` | 5.0 | 0.10 |
| `hip_pos` | 3.0 | 0.06 |
| `base_height` | 3.0 | 0.06 |
| `com_over_support` | 1.0 | 0.02 |
| `feet_clearance` | 1.0 | 0.02 |
| `foot_slip` | -2.0 | -0.04 |
| `low_speed` | 0.005 | 0.0001 |
| `rear_feet_contact_and_air` | 4.0 | 0.08 |
| `smoothness` | -0.01 | -0.0002 |
| `torques` | -0.0005 | -0.00001 |
| `dof_vel` | -0.00005 | -0.000001 |
| `dof_acc` | -0.0000025 | -0.00000005 |
| `collision` | -0.5 | -0.01 |
| `dof_vel_limits` | -10.0 | -0.20 |
| `feet_contact_forces` | -0.02 | -0.0004 |
| `termination` | -10.0 | -0.20 |

Нулевой `lin_vel_z` удаляется из списка reward-функций базовым классом.

Scales являются стартовой конфигурацией, а не доказанным оптимумом. Сравнивать
варианты следует по `evaluate_go2_stand.py`: command RMSE, failure rate,
pitch/height error, contact match, slip и episode duration.
