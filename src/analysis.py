import pandas as pd
import numpy as np

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
    outlier_stakes = set(issues[issues["issue_type"] == "OUTLIER"]["stake_id"])

    rows = []

    for stake_id, meta_group in df.groupby("stake_id"):

        group = displacements[displacements["stake_id"] == stake_id].sort_values("date_end")

        # invalid case: outlier or no valid segments
        if stake_id in outlier_stakes or len(group) == 0:
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

    # Glacier-level average velocity
    glacier_stats = (
        displacements.groupby("glacier")[["dx", "dy", "dt_days"]]
        .sum()
        .reset_index()
    )

    glacier_stats["vx"] = glacier_stats["dx"] / glacier_stats["dt_days"]
    glacier_stats["vy"] = glacier_stats["dy"] / glacier_stats["dt_days"]

    glacier_stats = glacier_stats[["glacier", "vx", "vy"]]

    # per stake
    rows = []

    outlier_stakes = set(issues[issues["issue_type"] == "OUTLIER"]["stake_id"])

    for stake_id, stake_meta in df.groupby("stake_id"):

        group = displacements[displacements["stake_id"] == stake_id].sort_values("date_end")

        glacier = stake_meta["glacier"].iloc[0]

        total_dx = group["dx"].sum()
        total_dy = group["dy"].sum()
        total_dt = group["dt_days"].sum()
        n_segments = len(group)

        vx, vy = None, None
        method = "NONE"
        quality = "NONE"

        # GLOBAL CASE: only if not an outlier and has at least 2 valid segments
        if stake_id not in outlier_stakes and total_dt > 0 and n_segments >= 2:

            vx = total_dx / total_dt
            vy = total_dy / total_dt
            method = "GLOBAL"

            if n_segments >= 10 and total_dt > 365:
                quality = "HIGH"
            elif n_segments >= 5:
                quality = "MEDIUM"
            else:
                quality = "LOW"

        # FALLBACK: if global velocity cannot be computed, use glacier average
        else:
            gl = glacier_stats[glacier_stats["glacier"] == glacier]

            if len(gl) > 0:
                vx = gl["vx"].values[0]
                vy = gl["vy"].values[0]
                method = "GLACIER_AVERAGE"
                quality = "LOW"

        rows.append({
            "stake_id": stake_id,
            "historic_vx_m_per_day": vx,
            "historic_vy_m_per_day": vy,
            "velocity_method": method,
            "velocity_quality": quality
        })

    return pd.DataFrame(rows)

# Predict future position based on historic velocity model
def compute_prediction(df, stakes_summary, target_date):

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

    # Merge with velocity model to get historic velocities
    pred = last_positions.merge(
        stakes_summary[[
            "stake_id",
            "historic_vx_m_per_day",
            "historic_vy_m_per_day",
            "velocity_method",
            "velocity_quality"
        ]],
        on="stake_id",
        how="left"
    )

    # Predict position at target date with simple linear extrapolation 
    pred["x_pred"] = pred["x"] + pred["historic_vx_m_per_day"] * pred["delta_t_days"]
    pred["y_pred"] = pred["y"] + pred["historic_vy_m_per_day"] * pred["delta_t_days"]

    return pred[[
        "stake_id",
        "date",
        "delta_t_days",
        "x", "y",
        "x_pred", "y_pred",
        "historic_vx_m_per_day",
        "historic_vy_m_per_day",
        "velocity_method",
        "velocity_quality"
    ]]


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
