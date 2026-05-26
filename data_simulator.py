import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from project_paths import DATASETS_DIR, ensure_project_dirs


MAX_TRAINING_RUL_HOURS = 195.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Common machine sensor data simulator.")
    parser.add_argument(
        "kind",
        choices=("few-days", "healthy", "long-life"),
        help="Dataset type to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed. Omit this to generate fresh data every run.",
    )
    parser.add_argument(
        "--profile",
        choices=("auto", "healthy", "watch", "warning", "critical"),
        default="auto",
        help="Timeline profile for few-days data. Auto picks one randomly.",
    )
    return parser.parse_args()


def next_numbered_path(stem: str) -> tuple[int, Path]:
    dataset_number = 1
    while True:
        path = DATASETS_DIR / f"{stem}_{dataset_number}.csv"
        if not path.exists():
            return dataset_number, path
        dataset_number += 1


def save_latest_and_numbered(
    df: pd.DataFrame,
    latest_name: str,
    numbered_stem: str,
) -> tuple[int, Path, Path]:
    latest_path = DATASETS_DIR / latest_name
    dataset_number, numbered_path = next_numbered_path(numbered_stem)
    df.to_csv(latest_path, index=False)
    df.to_csv(numbered_path, index=False)
    return dataset_number, latest_path, numbered_path


def failure_risk_score(
    vibration: np.ndarray,
    bearing_wear: np.ndarray,
    lubricant_quality: np.ndarray,
    temperature: np.ndarray,
) -> np.ndarray:
    risk = (
        0.35 * (vibration / 8.8)
        + 0.25 * (bearing_wear / 100.0)
        + 0.20 * (1.0 - lubricant_quality / 100.0)
        + 0.20 * (temperature / 90.0)
    )
    return np.clip(risk, 0.0, 1.0)


def profile_rul_range(rng: np.random.Generator, profile: str) -> tuple[str, float, float]:
    if profile == "auto":
        profile = str(rng.choice(("healthy", "watch", "warning", "critical")))

    if profile == "healthy":
        end_rul = float(rng.uniform(115.0, 165.0))
        start_rul = float(rng.uniform(end_rul + 18.0, min(190.0, end_rul + 45.0)))
    elif profile == "watch":
        end_rul = float(rng.uniform(45.0, 95.0))
        start_rul = float(rng.uniform(end_rul + 25.0, min(180.0, end_rul + 70.0)))
    elif profile == "warning":
        end_rul = float(rng.uniform(12.0, 28.0))
        start_rul = float(rng.uniform(end_rul + 35.0, min(140.0, end_rul + 95.0)))
    else:
        end_rul = float(rng.uniform(2.0, 9.0))
        start_rul = float(rng.uniform(end_rul + 18.0, min(90.0, end_rul + 65.0)))

    return profile, start_rul, end_rul


def generate_few_days_dataset(seed: int = 42, profile: str = "auto") -> tuple[pd.DataFrame, str]:
    rng = np.random.default_rng(seed)
    selected_profile, start_rul, end_rul = profile_rul_range(rng, profile)
    n_samples = int(rng.integers(180, 421))
    frequency_minutes = int(rng.choice((15, 30, 45, 60)))
    timestamps = pd.date_range(
        "2026-01-02 00:00:00",
        periods=n_samples,
        freq=f"{frequency_minutes}min",
    )

    progress = np.linspace(0.0, 1.0, n_samples)
    curve_strength = float(rng.uniform(0.75, 1.8))
    remaining_life = end_rul + (start_rul - end_rul) * (1.0 - progress**curve_strength)
    remaining_life += rng.normal(0, max(0.5, (start_rul - end_rul) * 0.025), n_samples)
    remaining_life = np.maximum.accumulate(remaining_life[::-1])[::-1]
    remaining_life = np.clip(remaining_life, 0.0, MAX_TRAINING_RUL_HOURS)

    degradation = 1.0 - (remaining_life / MAX_TRAINING_RUL_HOURS)
    vibration = np.clip(
        1.8 + degradation * 6.5 + rng.normal(0, 0.22, n_samples),
        1.5,
        8.8,
    )
    temperature = np.clip(
        42.0 + degradation * 45.0 + rng.normal(0, 1.8, n_samples),
        40.0,
        90.0,
    )
    rpm = np.clip(
        1500.0 - degradation * 165.0 + rng.normal(0, 10.0, n_samples),
        1330.0,
        1510.0,
    )
    noise = np.clip(
        55.0 + degradation * 30.0 + rng.normal(0, 1.2, n_samples),
        55.0,
        85.0,
    )
    bearing_wear = np.clip(
        degradation * 100.0 + rng.normal(0, 2.2, n_samples),
        0.0,
        100.0,
    )
    lubricant_quality = np.clip(
        100.0 - degradation * 75.0 + rng.normal(0, 2.0, n_samples),
        24.0,
        100.0,
    )

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "vibration_rms_mm_s": vibration.round(3),
            "temperature_c": temperature.round(2),
            "rpm": rpm.round(1),
            "acoustic_noise_db": noise.round(2),
            "bearing_wear_percent": bearing_wear.round(2),
            "lubricant_quality_percent": lubricant_quality.round(2),
            "failure_risk_score": failure_risk_score(
                vibration, bearing_wear, lubricant_quality, temperature
            ).round(4),
            "remaining_useful_life_hours": remaining_life.round(2),
            "breakdown_label": 0,
        }
    )
    df.attrs["profile"] = selected_profile
    return df, selected_profile


def generate_healthy_dataset() -> pd.DataFrame:
    source_path = DATASETS_DIR / "simulated_testing_machine_dataset.csv"
    df = pd.read_csv(source_path).copy()
    df["vibration_sensor"] = df["vibration_sensor"] * 0.3
    df["actual_remaining_life"] = 1400
    return df


def generate_long_life_datasets(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    np.random.seed(seed)
    n_samples = 3000
    timestamps = np.arange(n_samples)

    vibration = 0.1 + (timestamps / n_samples) * 5.0 + np.random.normal(
        0, 0.2, n_samples
    )
    vibration = np.clip(vibration, 0.05, 8.0)
    bearing_wear = 1 + (timestamps / n_samples) * 8.0 + np.random.normal(
        0, 0.3, n_samples
    )
    bearing_wear = np.clip(bearing_wear, 0.5, 10.0)
    actual_rul = 150000 - (timestamps / n_samples) * 150000
    actual_rul += np.random.normal(0, 500, n_samples)
    actual_rul = np.clip(actual_rul, 0, 150000)

    train_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "vibration_sensor": vibration,
            "temperature_sensor": 40 + np.random.normal(0, 2, n_samples),
            "pressure_sensor": 100 + np.random.normal(0, 3, n_samples),
            "rpm_sensor": 1500 + np.random.normal(0, 50, n_samples),
            "bearing_wear_sensor": bearing_wear,
            "health_status": ["Healthy"] * n_samples,
            "actual_remaining_life": actual_rul,
        }
    )

    n_test = 500
    vibration_test = np.clip(0.5 + np.random.normal(0, 0.1, n_test), 0.3, 1.0)
    bearing_wear_test = np.clip(1.5 + np.random.normal(0, 0.15, n_test), 1.0, 2.5)
    rul_test = np.clip(135000 + np.random.normal(0, 2000, n_test), 130000, 140000)

    test_df = pd.DataFrame(
        {
            "timestamp": np.arange(n_test),
            "vibration_sensor": vibration_test,
            "temperature_sensor": 45 + np.random.normal(0, 1.5, n_test),
            "pressure_sensor": 102 + np.random.normal(0, 2, n_test),
            "rpm_sensor": 1500 + np.random.normal(0, 40, n_test),
            "bearing_wear_sensor": bearing_wear_test,
            "health_status": ["Healthy"] * n_test,
            "actual_remaining_life": rul_test,
        }
    )
    return train_df, test_df


def save_few_days(seed: int, profile: str) -> None:
    df, selected_profile = generate_few_days_dataset(seed, profile)
    dataset_number, latest_path, numbered_path = save_latest_and_numbered(
        df,
        "few_days_machine_dataset.csv",
        "few_days_machine_dataset",
    )
    print(f"[OK] Created dataset_{dataset_number}: {numbered_path}")
    print(f"[OK] Updated latest file: {latest_path}")
    print(f"Profile: {selected_profile}")
    print(f"Samples: {len(df)}")
    print(
        "Actual RUL range: "
        f"{df['remaining_useful_life_hours'].min():.1f} - "
        f"{df['remaining_useful_life_hours'].max():.1f} hours"
    )
    print(
        "Actual RUL range: "
        f"{df['remaining_useful_life_hours'].min() / 24:.1f} - "
        f"{df['remaining_useful_life_hours'].max() / 24:.1f} days"
    )


def save_healthy() -> None:
    df = generate_healthy_dataset()
    dataset_number, latest_path, numbered_path = save_latest_and_numbered(
        df,
        "healthy_machine_dataset.csv",
        "healthy_machine_dataset",
    )
    print(f"[OK] Created dataset_{dataset_number}: {numbered_path}")
    print(f"[OK] Updated latest file: {latest_path}")
    print(f"Vibration range: {df['vibration_sensor'].min():.3f} - {df['vibration_sensor'].max():.3f}")
    print(f"Expected RUL: {df['actual_remaining_life'].iloc[0]}")


def save_long_life(seed: int) -> None:
    train_df, test_df = generate_long_life_datasets(seed)
    train_number, train_latest_path, train_numbered_path = save_latest_and_numbered(
        train_df,
        "extended_training_dataset.csv",
        "extended_training_dataset",
    )
    test_number, test_latest_path, test_numbered_path = save_latest_and_numbered(
        test_df,
        "long_life_machine_dataset.csv",
        "long_life_machine_dataset",
    )
    print(f"[OK] Created training dataset_{train_number}: {train_numbered_path}")
    print(f"[OK] Updated latest training file: {train_latest_path}")
    print(f"   Samples: {len(train_df)}")
    print(f"   RUL range: {train_df['actual_remaining_life'].min():.0f} - {train_df['actual_remaining_life'].max():.0f}")
    print(f"\n[OK] Created test dataset_{test_number}: {test_numbered_path}")
    print(f"[OK] Updated latest test file: {test_latest_path}")
    print(f"   Samples: {len(test_df)}")
    print(f"   Expected RUL range: {test_df['actual_remaining_life'].min():.0f} - {test_df['actual_remaining_life'].max():.0f}")


def main() -> None:
    args = parse_args()
    ensure_project_dirs()
    seed = args.seed if args.seed is not None else int(np.random.default_rng().integers(1, 1_000_000))
    print(f"Using seed: {seed}")

    if args.kind == "few-days":
        save_few_days(seed, args.profile)
    elif args.kind == "healthy":
        save_healthy()
    else:
        save_long_life(seed)


if __name__ == "__main__":
    main()
