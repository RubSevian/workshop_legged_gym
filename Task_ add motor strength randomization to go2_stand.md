# Task: add motor strength randomization to `go2_stand`

Нужно немного усилить actuator domain randomization в текущем `go2_stand`, не меняя rewards, observations, PPO, gait logic и history.

Цель — сделать policy устойчивее к sim-to-sim gap Isaac Gym → MuJoCo.

## 1. Расширить существующий PD randomization

Сейчас используются multiplier'ы для `Kp` и `Kd`.

Изменить диапазоны на:

```python
stiffness_multiplier_range = [0.8, 1.2]
damping_multiplier_range = [0.7, 1.3]
```

То есть:

$$
K_p^{eff} = K_p \cdot k_p,
\qquad
k_p \sim U(0.8, 1.2)
$$

$$
K_d^{eff} = K_d \cdot k_d,
\qquad
k_d \sim U(0.7, 1.3)
$$

Логика существующего PD randomization должна остаться прежней.

---

# 2. Добавить отдельный motor strength randomization

Добавить в domain randomization новые параметры:

```python
randomize_motor_strength = True
motor_strength_range = [0.85, 1.15]
```

Смысл параметра:

после вычисления PD torque

$$
\tau^{PD}_j
=
K_{p,j}^{eff}(q_{target,j}-q_j)
-
K_{d,j}^{eff}\dot q_j
$$

нужно дополнительно масштабировать момент:

$$
\tau^{applied}_j
=
k_{\tau,j}\tau^{PD}_j
$$

где

$$
k_{\tau,j}\sim U(0.85,1.15).
$$

---

# 3. Randomization должна быть постоянной в течение episode

`motor_strength` НЕ должен генерироваться заново каждый simulation timestep.

Новый коэффициент генерируется:

- при создании environments;
- при reset соответствующего environment.

В течение episode:

$$
k_{\tau,j} = const.
$$

После reset генерируется новое значение.

---

# 4. Использовать отдельный coefficient для каждого мотора

Предпочтительно хранить:

```python
self.motor_strengths
```

shape:

```text
[num_envs, num_dof]
```

Для Go2:

$$
num\_dof=12.
$$

Таким образом:

$$
k_{\tau,1},k_{\tau,2},...,k_{\tau,12}
$$

могут немного различаться.

Это моделирует различия реальных приводов.

---

# 5. Применять multiplier перед torque clipping

Порядок должен быть:

```python
q_target = ...
kp_eff = ...
kd_eff = ...

torques = (
    kp_eff * (q_target - dof_pos)
    - kd_eff * dof_vel
)

torques = torques * motor_strengths

torques = torch.clip(
    torques,
    -torque_limits,
    torque_limits,
)
```

То есть:

$$
\tau^{PD}
\rightarrow
k_\tau\tau^{PD}
\rightarrow
clip(\tau).
$$

Не делать multiplier после saturation.

---

# 6. Поведение при отключённой randomization

Если:

```python
randomize_motor_strength = False
```

то:

$$
k_{\tau,j}=1
$$

для всех environments и всех motors.

Поведение controller должно полностью совпадать с текущей реализацией.

---

# 7. Добавить debug-проверку

Если в проекте уже есть `debug_randomization`, добавить туда вывод статистики:

```text
motor strength min
motor strength max
motor strength mean
```

И проверить, что два environment при одинаковых:

```text
q
dq
action
Kp
Kd
```

но разных `motor_strengths` дают разные torques.

---

# 8. Ничего больше не менять

Не менять сейчас:

- reward scales;
- reward functions;
- smoothness;
- `dof_acc`;
- history length;
- observation space;
- gait period;
- commands;
- PPO;
- action scale;
- estimator;
- latency;
- MuJoCo code.

Это отдельный эксперимент по actuator robustness.

Итоговая randomization должна быть:

$$
K_p^{eff}=K_p\cdot k_p
$$

$$
K_d^{eff}=K_d\cdot k_d
$$

$$
\tau=
k_\tau
\left[
K_p^{eff}(q_{target}-q)
-
K_d^{eff}\dot q
\right]
$$

с:

$$
k_p\sim U(0.8,1.2)
$$

$$
k_d\sim U(0.7,1.3)
$$

$$
k_\tau\sim U(0.85,1.15).
$$

После изменений показать diff и отдельно подтвердить, что `motor_strength` генерируется на reset и действительно влияет на torque.