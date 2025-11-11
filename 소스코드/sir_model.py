import json
import random
import time
import csv
import argparse
import platform
import textwrap
import numpy as np
import os
import glob
import sys
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider, Button

# Matplotlib 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
else: # Mac
    plt.rc('font', family='AppleGothic')
plt.rc('axes', unicode_minus=False)

# --- 상태 정의 ---
SUSCEPTIBLE, INFECTED, RECOVERED, VACCINATED, QUARANTINED, DECEASED = 0, 1, 2, 3, 4, 5
cmap = mcolors.ListedColormap(['blue', 'red', 'green', 'gray', 'purple', 'black'])
bounds = [0, 1, 2, 3, 4, 5, 6]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# --- 함수 정의 ---
def get_settings_from_gui(script_dir):
    config_files = sorted([os.path.basename(f) for f in glob.glob(os.path.join(script_dir, '*.json'))])
    if not config_files:
        messagebox.showerror("오류", "설정(*.json) 파일을 찾을 수 없습니다.")
        return None

    settings = {"start": False, "config_data": None}
    root = tk.Tk()
    root.title("시뮬레이션 설정")

    frame = tk.Frame(root, padx=10, pady=10)
    frame.pack(fill="both", expand=True)

    is_updating_sliders = False

    def load_config_data(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def on_scenario_select(*args):
        nonlocal is_updating_sliders
        is_updating_sliders = True
        try:
            selected_file = os.path.join(script_dir, selected_scenario.get())
            config_data = load_config_data(selected_file)
            
            width_entry.delete(0, tk.END)
            width_entry.insert(0, str(config_data.get('GRID_WIDTH', 50)))
            height_entry.delete(0, tk.END)
            height_entry.insert(0, str(config_data.get('GRID_HEIGHT', 50)))
            
            total_potential = config_data.get('TOTAL_VIRULENCE_POTENTIAL', 80)
            total_potential_entry.delete(0, tk.END)
            total_potential_entry.insert(0, str(total_potential))

            infection_rate = config_data.get('INFECTION_RATE', 0.4) * 100
            infection_slider.set(infection_rate)
            death_slider.set(total_potential - infection_rate)

        except Exception as e:
            messagebox.showerror("파일 로드 오류", f"설정 파일을 읽는 중 오류가 발생했습니다:\n{e}")
        finally:
            is_updating_sliders = False

    def update_death_slider(inf_val_str):
        nonlocal is_updating_sliders
        if is_updating_sliders: return
        is_updating_sliders = True
        try:
            total_potential = float(total_potential_entry.get())
            new_death_val = total_potential - float(inf_val_str)
            death_slider.set(new_death_val if new_death_val > 0 else 0)
        finally:
            is_updating_sliders = False

    def update_infection_slider(death_val_str):
        nonlocal is_updating_sliders
        if is_updating_sliders: return
        is_updating_sliders = True
        try:
            total_potential = float(total_potential_entry.get())
            new_inf_val = total_potential - float(death_val_str)
            infection_slider.set(new_inf_val if new_inf_val > 0 else 0)
        finally:
            is_updating_sliders = False

    def on_start():
        try:
            selected_file_name = selected_scenario.get()
            selected_file_path = os.path.join(script_dir, selected_file_name)
            config_data = load_config_data(selected_file_path)
            
            config_data['GRID_WIDTH'] = int(width_entry.get())
            config_data['GRID_HEIGHT'] = int(height_entry.get())
            config_data['INFECTION_RATE'] = float(infection_slider.get()) / 100.0
            config_data['BASE_DEATH_RATE'] = float(death_slider.get()) / 100.0
            config_data['TOTAL_VIRULENCE_POTENTIAL'] = float(total_potential_entry.get())
            config_data['config_filename'] = selected_file_name

            if config_data['GRID_WIDTH'] <= 0 or config_data['GRID_HEIGHT'] <= 0:
                raise ValueError("격자 크기는 0보다 커야 합니다.")

            settings["config_data"] = config_data
            settings["start"] = True
            root.destroy()
        except ValueError as e:
            messagebox.showerror("입력 오류", f"유효한 숫자를 입력하세요. 오류: {e}")

    tk.Label(frame, text="시나리오 선택:").grid(row=0, column=0, sticky="w", pady=2)
    selected_scenario = tk.StringVar(frame)
    default_scenario = 'config.json' if 'config.json' in config_files else config_files[0]
    selected_scenario.set(default_scenario)
    scenario_menu = tk.OptionMenu(frame, selected_scenario, *config_files, command=on_scenario_select)
    scenario_menu.grid(row=0, column=1, columnspan=2, sticky="ew", pady=2)

    tk.Label(frame, text="격자 가로 크기:").grid(row=1, column=0, sticky="w", pady=2)
    width_entry = tk.Entry(frame)
    width_entry.grid(row=1, column=1, columnspan=2, sticky="ew", pady=2)

    tk.Label(frame, text="격자 세로 크기:").grid(row=2, column=0, sticky="w", pady=2)
    height_entry = tk.Entry(frame)
    height_entry.grid(row=2, column=1, columnspan=2, sticky="ew", pady=2)

    tk.Label(frame, text="바이러스 잠재력 (확산+치사):").grid(row=3, column=0, sticky="w", pady=2)
    total_potential_entry = tk.Entry(frame)
    total_potential_entry.grid(row=3, column=1, columnspan=2, sticky="ew", pady=2)

    tk.Label(frame, text="확산률 (%):").grid(row=4, column=0, sticky="w", pady=2)
    infection_slider = tk.Scale(frame, from_=0, to=100, orient="horizontal", command=update_death_slider)
    infection_slider.grid(row=4, column=1, columnspan=2, sticky="ew", pady=2)

    tk.Label(frame, text="치사율 (%):").grid(row=5, column=0, sticky="w", pady=2)
    death_slider = tk.Scale(frame, from_=0, to=100, orient="horizontal", command=update_infection_slider)
    death_slider.grid(row=5, column=1, columnspan=2, sticky="ew", pady=2)

    start_button = tk.Button(frame, text="시뮬레이션 시작", command=on_start, font=("Malgun Gothic", 10, "bold"))
    start_button.grid(row=6, columnspan=3, pady=15)

    frame.grid_columnconfigure(1, weight=1)
    on_scenario_select()
    root.mainloop()
    return settings if settings["start"] else None

def initialize_grid(width, height, initial_infected):
    grid = np.full((height, width), SUSCEPTIBLE, dtype=np.int8)
    if initial_infected > 0:
        # Concentrate initial outbreak in a central zone
        zone_size = 10
        r_start = max(0, (height - zone_size) // 2)
        c_start = max(0, (width - zone_size) // 2)
        
        zone_indices = [r * width + c for r in range(r_start, r_start + zone_size) for c in range(c_start, c_start + zone_size)]
        
        num_to_infect = min(initial_infected, len(zone_indices))
        if num_to_infect > 0:
            infected_flat_indices = np.random.choice(zone_indices, num_to_infect, replace=False)
            grid.flat[infected_flat_indices] = INFECTED
    return grid

def handle_internal_travel(grid, params, news_history, generation):
    is_china_scenario = 'china' in params.get('config_filename', '').lower()
    alert_level = params['current_alert_level']
    base_travel_rate = params.get('INTERNAL_TRAVEL_RATE', 0)
    current_travel_rate = 0

    # Special rule for China before generation 12
    if is_china_scenario and generation <= 12:
        current_travel_rate = base_travel_rate
        if not params.get('china_early_travel_news_shown', False):
            news_history[generation + 0.3] = f"{generation}세대: 초기 단계에서는 방역 단계와 무관하게 항공편 이동이 허용됩니다."
            params['china_early_travel_news_shown'] = True
    else:
        # Standard travel logic for all other cases
        travel_ban_level = params.get('TRAVEL_BAN_ALERT_LEVEL', 6)
        if alert_level >= travel_ban_level:
            if not params.get('travel_ban_news_shown', False):
                news_history[generation + 0.3] = f"{generation}세대: 방역 {travel_ban_level}단계 이상 격상에 따라 모든 항공편 이동이 중단됩니다."
                params['travel_ban_news_shown'] = True
            return grid
        
        # Reset the flag if level drops below ban level
        params['travel_ban_news_shown'] = False

        if base_travel_rate > 0:
            level_params = params['ALERT_LEVELS'][str(alert_level)]
            social_distancing = level_params['social_distancing']
            current_travel_rate = base_travel_rate * (1 - social_distancing)**2

    if current_travel_rate == 0:
        return grid

    # Find all people who are eligible to travel (not quarantined or deceased)
    can_travel_mask = (grid != QUARANTINED) & (grid != DECEASED)
    can_travel_indices = np.where(can_travel_mask.flat)[0]

    # Calculate number of travelers based on the mobile population
    num_to_travel = int(len(can_travel_indices) * current_travel_rate)

    if num_to_travel < 2:
        return grid

    # Randomly select travelers from the eligible population
    traveler_indices = np.random.choice(can_travel_indices, num_to_travel, replace=False)
    
    # And shuffle them among their own group
    shuffled_traveler_indices = traveler_indices.copy()
    np.random.shuffle(shuffled_traveler_indices)

    # Create a copy of the traveler states from their new locations
    traveler_states = grid.flat[shuffled_traveler_indices].copy()
    # Place them back in the original traveler locations, effectively swapping
    grid.flat[traveler_indices] = traveler_states
    
    return grid

def get_neighbors(x, y, width, height):
    neighbors = []
    for i in range(max(0, x - 1), min(height, x + 2)):
        for j in range(max(0, y - 1), min(width, y + 2)):
            if (i, j) != (x, y):
                neighbors.append((i, j))
    return neighbors

def find_and_execute_regional_lockdown(grid, params, news_history, generation):
    width, height = params['GRID_WIDTH'], params['GRID_HEIGHT']
    detection_radius = params.get('REGIONAL_LOCKDOWN_DETECTION_RADIUS', 5)
    lockdown_radius = params.get('REGIONAL_LOCKDOWN_RADIUS', 10)
    
    print(f"\n{generation}세대: 감염 밀집 지역 탐색 및 봉쇄 시도...")
    infected_coords = np.argwhere(grid == INFECTED)

    if infected_coords.shape[0] == 0:
        return grid, news_history

    density_map = np.zeros((height, width))
    for r_inf, c_inf in infected_coords:
        r_min, r_max = max(0, r_inf - detection_radius), min(height, r_inf + detection_radius + 1)
        c_min, c_max = max(0, c_inf - detection_radius), min(width, c_inf + detection_radius + 1)
        density_map[r_min:r_max, c_min:c_max] += 1
    
    new_grid = grid.copy()
    max_lockdowns = params.get('MAX_SIMULTANEOUS_LOCKDOWNS', 1)
    lockdowns_executed = 0

    for i in range(max_lockdowns):
        if np.max(density_map) < 1: # Stop if no more clusters
            break

        epicenter = np.unravel_index(np.argmax(density_map), density_map.shape)
        epicenter_r, epicenter_c = epicenter
        
        # Add news for this specific lockdown
        news_key = generation + 0.4 + (i * 0.01) # Unique key for each news
        news_history[news_key] = f"{generation}세대: 감염 밀집 지역 ({epicenter_c}, {epicenter_r}) 일대에 지역 봉쇄 선포."
        print(news_history[news_key])

        # Apply lockdown
        r_min_lock, r_max_lock = max(0, epicenter_r - lockdown_radius), min(height, epicenter_r + lockdown_radius + 1)
        c_min_lock, c_max_lock = max(0, epicenter_c - lockdown_radius), min(width, epicenter_c + lockdown_radius + 1)
        
        lockdown_zone = (new_grid[r_min_lock:r_max_lock, c_min_lock:c_max_lock] == SUSCEPTIBLE) | \
                        (new_grid[r_min_lock:r_max_lock, c_min_lock:c_max_lock] == INFECTED)
        new_grid[r_min_lock:r_max_lock, c_min_lock:c_max_lock][lockdown_zone] = QUARANTINED
        
        # Erase this region from the density map to find the next peak
        density_map[r_min_lock:r_max_lock, c_min_lock:c_max_lock] = 0
        lockdowns_executed += 1

    if lockdowns_executed > 0:
        print(f"{lockdowns_executed}개 지역에 대한 봉쇄를 실행했습니다.")

    return new_grid, news_history

def simulation_step(grid, params):
    width, height = params['GRID_WIDTH'], params['GRID_HEIGHT']
    alert_level = params['current_alert_level']
    level_params = params['ALERT_LEVELS'][str(alert_level)]
    
    new_grid = grid.copy()
    
    # --- Pre-calculate probabilities ---
    social_distancing = level_params['social_distancing']
    quarantine_rate = level_params['quarantine_rate']
    current_infection_rate = params['INFECTION_RATE'] * (1 - social_distancing)
    death_rate = params['OVERWHELMED_DEATH_RATE_MULTIPLIER'] * params['BASE_DEATH_RATE'] if params['is_overwhelmed'] else params['BASE_DEATH_RATE']
    
    # --- Vectorized state changes ---

    # 1. Waning immunity for RECOVERED and VACCINATED
    rand_matrix_waning = np.random.random((height, width))
    recovered_mask = (grid == RECOVERED)
    waning_from_recovered = recovered_mask & (rand_matrix_waning < params['WANING_IMMUNITY_RATE'])
    new_grid[waning_from_recovered] = SUSCEPTIBLE
    
    vaccinated_mask = (grid == VACCINATED)
    waning_from_vaccinated = vaccinated_mask & (rand_matrix_waning < params['WANING_IMMUNITY_RATE'] * params['VACCINE_WANING_MODIFIER'])
    new_grid[waning_from_vaccinated] = SUSCEPTIBLE

    # 2. State changes for QUARANTINED
    rand_matrix_q = np.random.random((height, width))
    quarantined_mask = (grid == QUARANTINED)
    dying_from_q = quarantined_mask & (rand_matrix_q < death_rate)
    recovering_from_q = quarantined_mask & (~dying_from_q) & (rand_matrix_q < death_rate + params['RECOVERY_RATE'])
    new_grid[dying_from_q] = DECEASED
    new_grid[recovering_from_q] = RECOVERED

    # 3. State changes for INFECTED
    rand_matrix_i = np.random.random((height, width))
    infected_mask = (grid == INFECTED)
    dying_from_i = infected_mask & (rand_matrix_i < death_rate)
    quarantining_from_i = infected_mask & (~dying_from_i) & (rand_matrix_i < death_rate + quarantine_rate)
    recovering_from_i = infected_mask & (~dying_from_i) & (~quarantining_from_i) & (rand_matrix_i < death_rate + quarantine_rate + params['RECOVERY_RATE'])
    new_grid[dying_from_i] = DECEASED
    new_grid[quarantining_from_i] = QUARANTINED
    new_grid[recovering_from_i] = RECOVERED

    # 4. New infections based on infected neighbors
    infected_neighbor_count = np.zeros_like(grid, dtype=np.int8)
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            rolled = np.roll(grid, (dx, dy), axis=(0, 1))
            infected_neighbor_count += (rolled == INFECTED)

    # Probability of NOT getting infected by a single neighbor
    prob_no_infection_single = 1 - current_infection_rate
    # Probability of NOT getting infected by any of the k neighbors
    prob_no_infection_total = prob_no_infection_single ** infected_neighbor_count
    # Probability of getting infected
    infection_prob_matrix = 1 - prob_no_infection_total
    
    rand_matrix_infection = np.random.random((height, width))
    
    susceptible_mask = (new_grid == SUSCEPTIBLE)
    newly_infected = susceptible_mask & (rand_matrix_infection < infection_prob_matrix)
    new_grid[newly_infected] = INFECTED

    # Breakthrough infections
    prob_no_breakthrough_single = 1 - (current_infection_rate * params['BREAKTHROUGH_RATE_MODIFIER'])
    prob_no_breakthrough_total = prob_no_breakthrough_single ** infected_neighbor_count
    breakthrough_prob_matrix = 1 - prob_no_breakthrough_total

    vaccinated_mask_new = (new_grid == VACCINATED)
    newly_infected_breakthrough = vaccinated_mask_new & (rand_matrix_infection < breakthrough_prob_matrix)
    new_grid[newly_infected_breakthrough] = INFECTED

    # 5. Vaccination campaign
    if params['is_vaccination_campaign_active']:
        # Vaccination for SUSCEPTIBLE
        rand_matrix_vaccine_s = np.random.random((height, width))
        susceptible_mask = (new_grid == SUSCEPTIBLE)
        getting_vaccinated_s = susceptible_mask & (rand_matrix_vaccine_s < params['DAILY_VACCINATION_RATE'])
        new_grid[getting_vaccinated_s] = VACCINATED

        # Vaccination for RECOVERED (at a higher rate)
        rand_matrix_vaccine_r = np.random.random((height, width))
        recovered_mask = (new_grid == RECOVERED)
        recovered_vaccination_rate = params['DAILY_VACCINATION_RATE'] * params.get('RECOVERED_VACCINATION_RATE_MODIFIER', 1.5)
        getting_vaccinated_r = recovered_mask & (rand_matrix_vaccine_r < recovered_vaccination_rate)
        new_grid[getting_vaccinated_r] = VACCINATED

    return new_grid

def count_states(grid):
    s = np.sum(grid == SUSCEPTIBLE)
    i = np.sum(grid == INFECTED)
    r = np.sum(grid == RECOVERED)
    v = np.sum(grid == VACCINATED)
    q = np.sum(grid == QUARANTINED)
    d = np.sum(grid == DECEASED)
    return int(s), int(i), int(r), int(v), int(q), int(d)

def save_results(fig, stats_history):
    if fig is not None:
        fig.savefig('simulation_stats.png')
    else:
        fig_save, ax_save = plt.subplots(figsize=(10, 6))
        ax_save.plot(stats_history['S'], label='감염 가능', color='blue')
        ax_save.plot(stats_history['I'], label='감염', color='red')
        ax_save.plot(stats_history['R'], label='회복', color='green')
        ax_save.plot(stats_history['V'], label='백신 접종', color='gray')
        ax_save.plot(stats_history['Q'], label='격리', color='purple')
        ax_save.plot(stats_history['D'], label='사망', color='black')
        ax_save.legend()
        ax_save.set_xlabel("세대")
        ax_save.set_ylabel("인구 수")
        ax_save.set_title("전염병 확산 시뮬레이션 통계")
        fig_save.savefig('simulation_stats.png')
        plt.close(fig_save)

    with open('simulation_stats.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = list(stats_history.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        num_entries = len(stats_history['S'])
        for i in range(num_entries):
            row_data = {key: stats_history[key][i] for key in fieldnames}
            writer.writerow(row_data)
    print(f"결과가 simulation_stats.png와 simulation_stats.csv 파일로 저장되었습니다.")

def main(batch_mode=False, no_history_mode=False):
    if getattr(sys, 'frozen', False):
        # PyInstaller executable
        script_dir = sys._MEIPASS
    else:
        # Regular Python script
        script_dir = os.path.dirname(os.path.abspath(__file__))
    if not batch_mode:
        settings = get_settings_from_gui(script_dir)
        if not settings:
            print("시뮬레이션이 사용자에 의해 취소되었습니다.")
            return
        params = settings["config_data"]
    else:
        config_path = os.path.join(script_dir, 'config_tester.json')
        if not os.path.exists(config_path):
            config_path = os.path.join(script_dir, 'config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            params = json.load(f)

    GRID_WIDTH, GRID_HEIGHT = params['GRID_WIDTH'], params['GRID_HEIGHT']
    total_pop = GRID_WIDTH * GRID_HEIGHT
    print("시뮬레이션 데이터를 생성하는 중입니다. 잠시만 기다려주세요...")
    
    grid = initialize_grid(GRID_WIDTH, GRID_HEIGHT, params['INITIAL_INFECTED'])
    grid_history = []
    if not no_history_mode:
        grid_history.append(grid)
    stats_history = {'S': [], 'I': [], 'R': [], 'V': [], 'Q': [], 'D': [], 'Alert Level': []}
    news_history = {}
    
    sim_params = params.copy()
    sim_params['current_alert_level'] = 1
    sim_params['pending_level_change'] = None
    sim_params['change_scheduled_at'] = -1
    sim_params['is_overwhelmed'] = False
    sim_params['is_vaccination_campaign_active'] = False
    sim_params['vaccination_campaign_triggered'] = False
    sim_params['relaxation_news_shown'] = False
    sim_params['travel_ban_news_shown'] = False
    sim_params['lockdown_delay_news_shown'] = False
    sim_params['china_early_travel_news_shown'] = False
    generation = 0

    try:
        while True:
            s, i, r, v, q, d = count_states(grid)
            stats_history['S'].append(s)
            stats_history['I'].append(i)
            stats_history['R'].append(r)
            stats_history['V'].append(v)
            stats_history['Q'].append(q)
            stats_history['D'].append(d)
            stats_history['Alert Level'].append(sim_params['current_alert_level'])

            infected_count = i + q
            progress_text = f"세대: {generation}, 감염/격리: {infected_count}/{total_pop}"
            print(f"\r{progress_text}  ", end="")

            if sim_params['change_scheduled_at'] == generation:
                new_level = sim_params['pending_level_change']
                news_history[generation] = f"{generation}세대: 방역 단계가 {new_level}단계로 조정되었습니다."
                sim_params['current_alert_level'] = new_level
                sim_params['pending_level_change'] = None
                sim_params['change_scheduled_at'] = -1

                trigger_level = sim_params.get('REGIONAL_LOCKDOWN_TRIGGER_LEVEL')
                if sim_params.get('LOCKDOWN_STRATEGY') == "REGIONAL" and trigger_level and new_level == trigger_level:
                    # China-specific delay for lockdown
                    is_china_scenario = 'china' in sim_params.get('config_filename', '').lower()
                    if is_china_scenario and generation <= 12:
                        if not sim_params.get('lockdown_delay_news_shown', False):
                            news_history[generation + 0.4] = f"{generation}세대: 초기 항공 이동 보장을 위해 지역 봉쇄를 12세대 이후로 연기합니다."
                            sim_params['lockdown_delay_news_shown'] = True
                    else:
                        grid, news_history = find_and_execute_regional_lockdown(grid, sim_params, news_history, generation)

            if generation > 0 and generation % 10 == 0:
                            grid = handle_internal_travel(grid, sim_params, news_history, generation)
            if sim_params['pending_level_change'] is None:
                infected_ratio = (i + q) / total_pop if total_pop > 0 else 0
                current_level_str = str(sim_params['current_alert_level'])
                lockdown_strategy = sim_params.get("LOCKDOWN_STRATEGY", "REGIONAL")

                if lockdown_strategy == "ZERO_COVID":
                    if infected_ratio > 0 and sim_params['current_alert_level'] != 5:
                        sim_params['pending_level_change'] = 5
                        sim_params['change_scheduled_at'] = generation + sim_params['RESPONSE_DELAY']
                    elif (i + q) == 0 and sim_params['current_alert_level'] == 5:
                        sim_params['pending_level_change'] = 1
                        sim_params['change_scheduled_at'] = generation + sim_params['RESPONSE_DELAY']
                else: # REGIONAL or NONE
                    target_level = None
                    for level in range(5, sim_params['current_alert_level'], -1):
                        level_str_check = str(level - 1)
                        if "threshold_up" in sim_params['ALERT_LEVELS'][level_str_check] and infected_ratio > sim_params['ALERT_LEVELS'][level_str_check]["threshold_up"]:
                            target_level = level
                            break
                    if target_level:
                        sim_params['pending_level_change'] = target_level
                        sim_params['change_scheduled_at'] = generation + sim_params['RESPONSE_DELAY']
                    else:
                        if "threshold_down" in sim_params['ALERT_LEVELS'][current_level_str] and infected_ratio < sim_params['ALERT_LEVELS'][current_level_str]["threshold_down"] and sim_params['current_alert_level'] > 1:
                            vaccinated_ratio = v / total_pop if total_pop > 0 else 0
                            vaccination_threshold = sim_params.get('VACCINATION_THRESHOLD_FOR_RELAXATION', 0.7)

                            if vaccinated_ratio >= vaccination_threshold:
                                sim_params['pending_level_change'] = sim_params['current_alert_level'] - 1
                                sim_params['change_scheduled_at'] = generation + sim_params['RESPONSE_DELAY']
                            else:
                                if not sim_params.get('relaxation_news_shown', False):
                                    news_key = generation + 0.1 # Avoid overwriting other news
                                    news_history[news_key] = f"{generation}세대: 감염자 수는 감소했으나, 백신 접종률({vaccinated_ratio:.1%})이 목표치({vaccination_threshold:.1%})에 도달하지 못해 방역 단계를 완화하지 않습니다."
                                    sim_params['relaxation_news_shown'] = True

            if not sim_params['is_overwhelmed'] and (i + q) / total_pop > sim_params['HOSPITAL_CAPACITY_THRESHOLD']:
                sim_params['is_overwhelmed'] = True
                news_history[generation] = f"{generation}세대: 의료 시스템이 과부하 상태입니다."

            if not sim_params['vaccination_campaign_triggered'] and (i + d) / total_pop > sim_params['VACCINATION_START_THRESHOLD']:
                sim_params['is_vaccination_campaign_active'] = True
                sim_params['vaccination_campaign_triggered'] = True
                news_history[generation] = f"{generation}세대: 백신 접종 캠페인이 시작되었습니다."

            grid = simulation_step(grid, sim_params)
            if not no_history_mode:
                grid_history.append(grid)
            generation += 1

            if i == 0 and q == 0:
                break
    except KeyboardInterrupt:
        print("\n\n시뮬레이션을 사용자가 중단했습니다. 현재까지의 결과로 그래프를 생성합니다.")
    
    print()
    print(f"총 {generation} 세대 시뮬레이션 완료.")

    if batch_mode or no_history_mode:
        print("\n시뮬레이션이 완료되었습니다. 결과를 파일로 저장합니다.")
        save_results(None, stats_history)
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    fig.canvas.manager.set_window_title('전염병 확산 시뮬레이션')
    plt.subplots_adjust(bottom=0.3, left=0.3)

    img = ax1.imshow(grid_history[0], cmap=cmap, norm=norm, interpolation='nearest')
    line_s, = ax2.plot(stats_history['S'], label='감염 가능', color='blue')
    line_i, = ax2.plot(stats_history['I'], label='감염', color='red')
    line_r, = ax2.plot(stats_history['R'], label='회복', color='green')
    line_v, = ax2.plot(stats_history['V'], label='백신 접종', color='gray')
    line_q, = ax2.plot(stats_history['Q'], label='격리', color='purple')
    line_d, = ax2.plot(stats_history['D'], label='사망', color='black')
    vline = ax2.axvline(0, color='black', linestyle='--')
    ax2.set_xlabel("세대")
    ax2.set_ylabel("인구 수")
    ax2.set_title("통계")
    ax2.set_ylim(0, total_pop)
    ax2.set_xlim(0, generation)
    ax2.legend(loc='upper right')

    kpi_text = (f"--- 핵심 지표 ---\n" f"총사망자: {stats_history['D'][-1]:,}명\n" f"최대 감염자: {max(stats_history['I']):,}명 (at Gen {stats_history['I'].index(max(stats_history['I']))})\n" f"설정 파일: {params.get('config_filename', 'config.json')}")
    fig.text(0.7, 0.15, kpi_text, ha='left', va='top', fontsize=10, bbox={"facecolor": 'white', "alpha": 0.8, "pad": 5})
    
    news_log_text = fig.text(0.05, 0.8, "뉴스 기록:", va='top', ha='left', fontsize=10)

    ax_slider = plt.axes([0.3, 0.2, 0.6, 0.03])
    frame_slider = Slider(ax_slider, '세대', 0, generation, valinit=0, valstep=1)

    for gen_key in news_history.keys():
        ax_slider.axvline(gen_key, color='red', linestyle='--', lw=1)

    ax_play = plt.axes([0.85, 0.1, 0.1, 0.04])
    play_button = Button(ax_play, '재생')
    ax_pause = plt.axes([0.75, 0.1, 0.1, 0.04])
    pause_button = Button(ax_pause, '일시정지')
    ax_save = plt.axes([0.6, 0.1, 0.15, 0.04])
    save_button = Button(ax_save, '결과 저장')

    class Player:
        def __init__(self):
            self.is_playing = False
            self.timer = fig.canvas.new_timer(interval=200)
            self.timer.add_callback(self.on_tick)
        def on_tick(self):
            if not self.is_playing: return
            current_frame = frame_slider.val
            if current_frame < generation:
                frame_slider.set_val(current_frame + 1)
            else:
                self.pause()
        def play(self, event):
            if not self.is_playing:
                self.is_playing = True
                self.timer.start()
        def pause(self, event=None):
            self.is_playing = False
            self.timer.stop()

    player = Player()
    play_button.on_clicked(player.play)
    pause_button.on_clicked(player.pause)

    def save_results_callback(event):
        save_results(fig, stats_history)

    save_button.on_clicked(save_results_callback)

    def update(val):
        frame = int(frame_slider.val)
        img.set_data(grid_history[frame])
        title = f"세대: {frame} (방역 {stats_history['Alert Level'][frame]}단계)"
        ax1.set_title(title)

        relevant_news = [news_history[g] for g in sorted(news_history.keys()) if g <= frame]
        
        wrapped_news = []
        for news_item in relevant_news[-5:]: # Process last 5 news items
            wrapped_lines = textwrap.wrap(news_item, width=45, initial_indent='- ', subsequent_indent='  ')
            wrapped_news.extend(wrapped_lines)

        # Limit total lines to prevent overflow
        display_news_lines = wrapped_news[-10:] 
        
        display_news = "뉴스 기록:\n" + "\n".join(display_news_lines)
        news_log_text.set_text(display_news)

        line_s.set_label(f'감염 가능 ({stats_history["S"][frame]})')
        line_i.set_label(f'감염 ({stats_history["I"][frame]})')
        line_r.set_label(f'회복 ({stats_history["R"][frame]})')
        line_v.set_label(f'백신 접종 ({stats_history["V"][frame]})')
        line_q.set_label(f'격리 ({stats_history["Q"][frame]})')
        line_d.set_label(f'사망 ({stats_history["D"][frame]})')
        ax2.legend(loc='upper right')
        vline.set_xdata([frame, frame])
        fig.canvas.draw_idle()

    frame_slider.on_changed(update)
    update(0)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='전염병 확산 시뮬레이션')
    parser.add_argument('--batch', action='store_true', help='GUI 없이 배치 모드로 실행')
    parser.add_argument('--no-history', action='store_true', help='메모리 사용을 최적화하기 위해 세대별 기록을 저장하지 않는 고속 모드로 실행')
    args = parser.parse_args()
    main(batch_mode=args.batch, no_history_mode=args.no_history)
