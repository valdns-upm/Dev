import pandas as pd
import numpy as np


def build_glacier_velocity_stats(displacements):
    glacier_stats = (
        displacements.groupby("glacier")[["dx", "dy", "dt_days"]]
        .sum()
        .reset_index()
    )

    glacier_stats["vx"] = glacier_stats["dx"] / glacier_stats["dt_days"]
    glacier_stats["vy"] = glacier_stats["dy"] / glacier_stats["dt_days"]

    return glacier_stats[["glacier", "vx", "vy"]]


def estimate_velocity_components(
    stake_segments,
    glacier,
    glacier_stats,
):
    vx_est, vy_est = None, None
    method = "NONE"
    quality = "NONE"

    total_dt = stake_segments["dt_days"].sum() if not stake_segments.empty else 0
    n_segments = len(stake_segments)

    # Priority 1: GLOBAL
    if n_segments >= 2 and total_dt > 0:
        vx_est = stake_segments["dx"].sum() / total_dt
        vy_est = stake_segments["dy"].sum() / total_dt
        method = "GLOBAL"

        span_days = (stake_segments["date_end"].max() - stake_segments["date_start"].min()).days
        if n_segments >= 10 and span_days > 365:
            quality = "HIGH"
        elif n_segments >= 5:
            quality = "MEDIUM"
        else:
            quality = "LOW"
    # Priority 2: LAST
    elif n_segments == 1:
        only_seg = stake_segments.iloc[0]
        vx_est = only_seg["dx"] / only_seg["dt_days"]
        vy_est = only_seg["dy"] / only_seg["dt_days"]
        method = "LAST"
        quality = "LOW"
    # Priority 3: GLACIER
    else:
        gl = glacier_stats[glacier_stats["glacier"] == glacier]
        if len(gl) > 0:
            vx_est = gl["vx"].values[0]
            vy_est = gl["vy"].values[0]
            method = "GLACIER"
            quality = "LOW"

    return vx_est, vy_est, method, quality


# Main function to compute summary statistics for each stake
def compute_stake_summary(df, displacements, issues):

    # METADATA
    meta = df.groupby("stake_id").agg(
        first_date=("date", "min"),
        last_date=("date", "max"),
        n_points=("date", "count"),
        glacier=("glacier", "first")
    ).reset_index()


    # MISSING YEARS
    def has_gap(group):
        years = group["date"].dt.year.unique()
        return set(range(years.min(), years.max() + 1)) != set(years)

    gaps = df.groupby("stake_id").apply(has_gap).reset_index(name="has_missing_year")

    # DISPLACEMENTS SUMMARY
    rows = []

    for stake_id, meta_group in df.groupby("stake_id"):

        group = displacements[displacements["stake_id"] == stake_id].sort_values("date_end")

        # invalid case: no valid segments after cleaning
        if len(group) == 0:
            rows.append({
                "stake_id": stake_id,
                "valid_segments": 0,
                "total_dx_m": None,
                "total_dy_m": None,
                "total_dz_m": None,
                "total_dt_days": None,
                "mean_speed_m_per_year": None
            })
            continue

        total_dx = group["dx"].sum()
        total_dy = group["dy"].sum()
        total_dz = group["dz"].sum()
        total_dt = group["dt_days"].sum()

        total_distance = np.sqrt(total_dx**2 + total_dy**2)
        mean_speed = total_distance / total_dt if total_dt > 0 else None

        rows.append({
            "stake_id": stake_id,
            "valid_segments": len(group),
            "total_dx_m": total_dx,
            "total_dy_m": total_dy,
            "total_dz_m": total_dz,
            "total_dt_days": total_dt,
            "mean_speed_m_per_year": mean_speed * 365 if mean_speed else None
        })

    summary_disp = pd.DataFrame(rows)

    summary = meta.merge(summary_disp, on="stake_id", how="left")
    summary = summary.merge(gaps, on="stake_id", how="left")

    # VELOCITY MODEL
    velocity_model = compute_stake_velocity_model(displacements, df, issues)

    summary = summary.merge(velocity_model, on="stake_id", how="left")

    return summary

# Summarize data availability per year for each stake
def compute_year_summary(df):

    rows = []

    for stake_id, group in df.groupby("stake_id"):

        counts = group["date"].dt.year.value_counts().to_dict()

        min_year = group["date"].dt.year.min()
        max_year = group["date"].dt.year.max()

        for year in range(min_year, max_year + 1):

            n = counts.get(year, 0)

            if n == 0:
                status = "NO_DATA"
            elif n == 1:
                status = "ONE_MEASUREMENT"
            else:
                status = "MULTIPLE_MEASUREMENTS"

            rows.append({
                "stake_id": stake_id,
                "year": year,
                "n_measurements": n,
                "status": status
            })

    return pd.DataFrame(rows).sort_values(["stake_id", "year"])


# Choose method for velocity calculation depending on data quality/availability
def compute_stake_velocity_model(displacements, df, issues):
    glacier_stats = build_glacier_velocity_stats(displacements)

    # per stake
    rows = []

    for stake_id, stake_meta in df.groupby("stake_id"):

        group = displacements[displacements["stake_id"] == stake_id].sort_values("date_end")

        glacier = stake_meta["glacier"].iloc[0]
        vx, vy, method, quality = estimate_velocity_components(
            stake_segments=group,
            glacier=glacier,
            glacier_stats=glacier_stats,
        )

        rows.append({
            "stake_id": stake_id,
            "historic_vx_m_per_day": vx,
            "historic_vy_m_per_day": vy,
            "velocity_method": method,
            "velocity_quality": quality
        })

    return pd.DataFrame(rows)

# Predict future position using the same shared velocity logic as the summary
def compute_prediction(df, displacements, target_date):

    last_positions = (
        df.sort_values("date")
        .groupby("stake_id")
        .last()
        .reset_index()
    )

    # Calculate time difference between last measurement and target date
    last_positions["delta_t_days"] = (
        pd.to_datetime(target_date) - last_positions["date"]
    ).dt.days

    glacier_stats = build_glacier_velocity_stats(displacements)

    rows = []
    for row in last_positions.itertuples(index=False):
        stake_segments = displacements[
            (displacements["stake_id"] == row.stake_id) & (displacements["date_end"] <= row.date)
        ].sort_values("date_end")

        vx_est, vy_est, method, quality = estimate_velocity_components(
            stake_segments=stake_segments,
            glacier=row.glacier,
            glacier_stats=glacier_stats,
        )

        x_pred = row.x + vx_est * row.delta_t_days if pd.notna(vx_est) else np.nan
        y_pred = row.y + vy_est * row.delta_t_days if pd.notna(vy_est) else np.nan

        rows.append({
            "stake_id": row.stake_id,
            "date": row.date,
            "delta_t_days": row.delta_t_days,
            "x": row.x,
            "y": row.y,
            "x_pred": x_pred,
            "y_pred": y_pred,
            "vx_est": vx_est,
            "vy_est": vy_est,
            "method": method,
            "quality": quality,
        })

    return pd.DataFrame(rows)


# Count number of stakes with data from recent campaigns
def summarize_recent_campaigns(df, n_campaigns=2):
    campaign_years = sorted(int(year) for year in df["date"].dt.year.dropna().unique())
    recent_campaigns = campaign_years[-n_campaigns:]

    stakes_with_recent_campaigns = (
        df[df["date"].dt.year.isin(recent_campaigns)]
        .assign(year=lambda frame: frame["date"].dt.year)
        .groupby("stake_id")["year"]
        .nunique()
        .eq(n_campaigns)
        .sum()
    )

    return {
        "campaign_years": campaign_years,
        "recent_campaigns": recent_campaigns,
        "stakes_with_recent_campaigns": stakes_with_recent_campaigns,
    }
