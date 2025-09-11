import math

def check_and_calculate_parameters(dt, decimation, cycle_time, bias, num_steps_per_env, episode_length_s, real_latency_range=(0.02, 0.05)):
    """
    Проверяет и рассчитывает параметры для задачи хождения робота в Legged Gym.
    
    Аргументы:
        dt (float): Частота симуляции (с).
        decimation (int): Количество шагов симуляции между обновлениями политики.
        cycle_time (float): Длительность цикла походки (с).
        bias (float): Параметр для фазы двойной опоры (0–1).
        num_steps_per_env (int): Количество шагов симуляции на итерацию PPO.
        episode_length_s (float): Длительность эпизода (с).
        real_latency_range (tuple): Диапазон реальных задержек управления (с).

    Возвращает:
        dict: Результаты расчётов и рекомендации.
    """
    results = {}
    
    # 1. Расчёт основных метрик
    # Частота симуляции (Гц)
    sim_freq = 1 / dt
    results["Simulation frequency (Hz)"] = sim_freq
    
    # Задержка управления (с) и частота RL (Гц)
    control_latency = decimation * dt
    rl_freq = 1 / control_latency
    results["Control latency (s)"] = control_latency
    results["RL frequency (Hz)"] = rl_freq
    
    # Количество шагов в цикле походки
    cycle_steps = cycle_time / dt
    results["Cycle time (steps)"] = cycle_steps
    
    # Длительность фазы двойной опоры (с и шаги)
    phase_fraction = math.asin(bias) / (2 * math.pi)
    double_support_time = 2 * phase_fraction * cycle_time  # Учитываем обе стороны sin
    double_support_steps = double_support_time / dt
    results["Double support phase time (s)"] = double_support_time
    results["Double support phase steps"] = double_support_steps
    
    # Количество циклов в эпизоде
    episode_steps = episode_length_s / dt
    num_cycles = episode_steps / cycle_steps
    results["Episode steps"] = episode_steps
    results["Number of cycles per episode"] = num_cycles
    
    # Длительность итерации PPO (с)
    ppo_iteration_time = num_steps_per_env * dt
    results["PPO iteration time (s)"] = ppo_iteration_time
    results["PPO iteration cycles"] = ppo_iteration_time / cycle_time
    
    # 2. Проверка согласованности
    warnings = []
    
    # Проверка: Задержка управления vs. фаза двойной опоры
    if control_latency > double_support_time:
        warnings.append(
            f"Предупреждение: Задержка управления ({control_latency:.3f} с) больше фазы двойной опоры "
            f"({double_support_time:.3f} с). Рекомендуется увеличить bias до {min(0.4, bias * 1.5):.2f} "
            f"или уменьшить decimation до {int(double_support_time / dt):d}."
        )
    
    # Проверка: num_steps_per_env vs. cycle_time
    if ppo_iteration_time < cycle_time:
        warnings.append(
            f"Предупреждение: num_steps_per_env ({num_steps_per_env} шагов, {ppo_iteration_time:.3f} с) "
            f"меньше cycle_time ({cycle_time:.3f} с). Рекомендуется установить num_steps_per_env >= {int(cycle_steps):d}."
        )
    
    # Проверка: episode_length_s vs. num_cycles
    if num_cycles < 20:
        warnings.append(
            f"Предупреждение: Мало циклов в эпизоде ({num_cycles:.1f}). Рекомендуется увеличить episode_length_s "
            f"до {20 * cycle_time:.1f} с или больше."
        )
    
    # Проверка: Реальные задержки vs. decimation
    min_real_latency, max_real_latency = real_latency_range
    if control_latency < min_real_latency or control_latency > max_real_latency:
        warnings.append(
            f"Предупреждение: Задержка управления ({control_latency:.3f} с) вне диапазона реальных задержек "
            f"({min_real_latency:.3f}–{max_real_latency:.3f} с). Рекомендуется decimation в диапазоне "
            f"[{int(min_real_latency / dt):d}, {int(max_real_latency / dt):d}] или рандомизация задержек."
        )
    
    # Проверка: Частота RL vs. 100 Гц
    target_rl_freq = 100
    target_decimation = int(1 / (target_rl_freq * dt))
    if abs(rl_freq - target_rl_freq) > 20:
        warnings.append(
            f"Предупреждение: Частота RL ({rl_freq:.1f} Гц) сильно отличается от целевой 100 Гц. "
            f"Рекомендуется decimation = {target_decimation} для частоты ~100 Гц."
        )
    
    results["Warnings"] = warnings
    
    # 3. Рекомендации для устранения частого переключения лап
    recommendations = []
    
    # Рекомендация по bias
    if double_support_time < min_real_latency:
        recommended_bias = (math.asin(min_real_latency / cycle_time * 2 * math.pi) / (2 * math.pi)) * 1.2
        recommendations.append(
            f"Увеличьте bias до {min(0.4, recommended_bias):.2f}, чтобы фаза двойной опоры была >= {min_real_latency:.3f} с."
        )
    
    # Рекомендация по cycle_time
    if cycle_time < 0.4:
        recommendations.append(
            f"Увеличьте cycle_time до 0.5–0.6 с для более медленной походки, чтобы уменьшить 'суетливость'."
        )
    
    # Рекомендация по contact_change_penalty
    recommendations.append(
        "Увеличьте contact_change_penalty до -2.0 или -2.5 в _reward_rear_feet_contact_and_air для уменьшения частых переключений лап."
    )
    
    # Рекомендация по рандомизации задержек
    recommendations.append(
        "Добавьте рандомизацию задержек в domain_rand (latency_range = [0.02, 0.05]) для адаптации к реальным условиям."
    )
    
    # Рекомендация по base_height
    if control_latency > 0.03:
        recommendations.append(
            "Увеличьте вес base_height до 7.0–8.0 в _reward_base_height, чтобы компенсировать задержки управления."
        )
    
    results["Recommendations"] = recommendations
    
    return results

def print_results(results):
    """Выводит результаты расчётов и рекомендации."""
    print("=== Результаты расчётов параметров ===")
    for key, value in results.items():
        if key not in ["Warnings", "Recommendations"]:
            print(f"{key}: {value:.3f}")
        else:
            print(f"\n{key}:")
            for item in value:
                print(f"  - {item}")
    print("\n=== Конец отчёта ===")

# Пример использования
params = {
    "dt": 0.005,  # Частота симуляции (с)
    "decimation": 4,  # Количество шагов между обновлениями политики
    "cycle_time": 0.3,  # Длительность цикла походки (с)
    "bias": 0.25,  # Параметр для фазы двойной опоры
    "num_steps_per_env": 60,  # Шаги на итерацию PPO
    "episode_length_s": 15,  # Длительность эпизода (с)
    "real_latency_range": (0.02, 0.05)  # Диапазон реальных задержек (с)
}

results = check_and_calculate_parameters(**params)
print_results(results)