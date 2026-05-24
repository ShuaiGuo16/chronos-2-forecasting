from __future__ import annotations

import numpy as np
import pandas as pd


SEED = 42
START_DATE = pd.Timestamp("2025-03-01 00:00")
N_DAYS = 170
TIMESTAMPS = pd.date_range(START_DATE, periods=N_DAYS * 24, freq="h")


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def outdoor_temperature(index: pd.DatetimeIndex) -> np.ndarray:
    hour = index.hour.to_numpy()
    day = ((index - START_DATE).days).to_numpy()
    daily_cycle = 7.0 * np.sin(2 * np.pi * (hour - 15) / 24)
    seasonal_trend = 18.5 + 0.045 * day
    synoptic_variation = 2.2 * np.sin(2 * np.pi * day / 9 + 0.8) + 1.2 * np.sin(2 * np.pi * day / 17)
    return seasonal_trend + daily_cycle + synoptic_variation


def solar_profile(index: pd.DatetimeIndex) -> np.ndarray:
    hour = index.hour.to_numpy()
    day = ((index - START_DATE).days).to_numpy()
    daylight = np.clip(np.sin(np.pi * (hour - 6) / 13), 0, None)
    seasonal_strength = 0.8 + 0.2 * np.sin(2 * np.pi * (day - 40) / 365)
    return daylight * seasonal_strength


def occupancy_profile(index: pd.DatetimeIndex, scale: float) -> np.ndarray:
    hour = index.hour.to_numpy()
    weekday = (index.dayofweek.to_numpy() < 5).astype(float)
    workday_shape = _sigmoid((hour - 7.5) / 1.0) * _sigmoid((18.0 - hour) / 1.2)
    lunch_dip = 1.0 - 0.12 * np.exp(-((hour - 12.5) / 1.3) ** 2)
    return scale * weekday * workday_shape * lunch_dip


def simulate_building(
    building: str,
    floor_area_m2: float,
    occupancy_scale: float,
    insulation_factor: float,
    hvac_size_factor: float,
    cooling_setpoint_c: float,
    heating_setpoint_c: float,
    plug_scale: float,
) -> pd.DataFrame:
    n = len(TIMESTAMPS)
    building_seed = SEED + int(building.split()[-1]) * 97
    rng = np.random.default_rng(building_seed)

    outdoor_temp = outdoor_temperature(TIMESTAMPS)
    solar = solar_profile(TIMESTAMPS)
    occupancy = occupancy_profile(TIMESTAMPS, occupancy_scale)
    thermal_mass = floor_area_m2 * 0.058
    thermal_resistance = 0.022 * insulation_factor
    max_cooling_kw = floor_area_m2 * 0.034 * hvac_size_factor
    max_heating_kw = floor_area_m2 * 0.020 * hvac_size_factor
    controller_gain = floor_area_m2 * 0.014 * hvac_size_factor
    cooling_cop = 3.1
    heating_cop = 3.4

    indoor_temp = np.zeros(n)
    base_load = np.zeros(n)
    plug_load = np.zeros(n)
    lighting_load = np.zeros(n)
    hvac_load = np.zeros(n)
    total_load = np.zeros(n)
    cooling_thermal = np.zeros(n)
    heating_thermal = np.zeros(n)

    indoor_temp[0] = 23.0

    for t in range(n - 1):
        plug_load[t] = (
            floor_area_m2 * (0.0045 + 0.0100 * plug_scale * occupancy[t])
            + rng.normal(0, floor_area_m2 * 0.00018)
        )
        lighting_load[t] = (
            floor_area_m2 * (0.0020 + 0.0060 * occupancy[t] * (1.0 - 0.45 * solar[t]))
            + rng.normal(0, floor_area_m2 * 0.00018)
        )
        base_load[t] = floor_area_m2 * 0.0035 + 5.0 * np.sin(2 * np.pi * t / (24 * 14))

        internal_gain_kw = 0.65 * plug_load[t] + 0.85 * lighting_load[t] + floor_area_m2 * 0.0012 * occupancy[t]
        solar_gain_kw = floor_area_m2 * 0.012 * solar[t]

        cooling_need = max(0.0, indoor_temp[t] - cooling_setpoint_c)
        heating_need = max(0.0, heating_setpoint_c - indoor_temp[t])
        desired_cooling = min(max_cooling_kw, controller_gain * cooling_need)
        desired_heating = min(max_heating_kw, controller_gain * heating_need)
        cooling_thermal[t] = 0.55 * (cooling_thermal[t - 1] if t > 0 else 0.0) + 0.45 * desired_cooling
        heating_thermal[t] = 0.55 * (heating_thermal[t - 1] if t > 0 else 0.0) + 0.45 * desired_heating
        hvac_load[t] = cooling_thermal[t] / cooling_cop + heating_thermal[t] / heating_cop

        temperature_change = (
            (outdoor_temp[t] - indoor_temp[t]) / thermal_resistance
            + internal_gain_kw
            + solar_gain_kw
            - cooling_thermal[t]
            + heating_thermal[t]
        ) / thermal_mass
        indoor_temp[t + 1] = indoor_temp[t] + temperature_change

        total_load[t] = max(
            0.0,
            base_load[t] + plug_load[t] + lighting_load[t] + hvac_load[t] + rng.normal(0, floor_area_m2 * 0.00035),
        )

    base_load[-1] = base_load[-2]
    plug_load[-1] = plug_load[-2]
    lighting_load[-1] = lighting_load[-2]
    hvac_load[-1] = hvac_load[-2]
    total_load[-1] = base_load[-1] + plug_load[-1] + lighting_load[-1] + hvac_load[-1]

    return pd.DataFrame(
        {
            "building": building,
            "timestamp": TIMESTAMPS,
            "total_load_kw": total_load,
            "hvac_load_kw": hvac_load,
            "plug_load_kw": plug_load,
            "lighting_load_kw": lighting_load,
            "indoor_temp_c": indoor_temp,
            "outdoor_temp_c": outdoor_temp,
            "occupancy": occupancy,
            "solar_irradiance": solar,
            "is_weekend": (TIMESTAMPS.dayofweek >= 5).astype(int),
        }
    )


def make_building_demand_dataset() -> pd.DataFrame:
    building_profiles = [
        ("Building 01", 9_200, 1.00, 0.92, 1.05, 24.0, 19.0, 0.95),
        ("Building 02", 7_600, 0.88, 1.10, 0.92, 24.5, 19.0, 0.85),
        ("Building 03", 10_400, 1.18, 0.85, 1.15, 24.0, 19.5, 1.05),
        ("Building 04", 6_800, 0.75, 1.30, 0.75, 24.0, 19.0, 0.70),
        ("Building 05", 11_800, 1.30, 0.78, 1.25, 23.5, 19.0, 1.15),
        ("Building 06", 8_400, 0.95, 1.05, 0.90, 24.5, 19.5, 0.82),
        ("Building 07", 9_800, 1.08, 0.95, 1.00, 24.0, 19.0, 1.00),
        ("Building 08", 10_000, 1.12, 0.90, 1.12, 24.0, 19.5, 1.02),
    ]
    return pd.concat([simulate_building(*profile) for profile in building_profiles], ignore_index=True)
