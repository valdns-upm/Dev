import pandas as pd
import numpy as np


def estimate_velocity_components(
    stake_segments,
):
    vx_est, vy_est = None, None

    total_dt = stake_segments["dt_days"].sum() if not stake_segments.empty else 0
    n_segments = len(stake_segments)

    if n_segments >= 2 and total_dt > 0:
        vx_est = stake_segments["dx"].sum() / total_dt
        vy_est = stake_segments["dy"].sum() / total_dt
    return vx_est, vy_est


# Main function to compute summary statistics for each stake
def compute_stake_summary(df, displacements):

    # METADATA
    meta = df.groupby("stake_id").agg(
        first_date=("date", "min"),
        last_date=("date", "max"),
        n_points=("date", "count"),
        glacier=("glacier", "first")
    ).reset_index()
    meta.loc[meta["n_points"] == 1, "last_date"] = pd.NaT

    outlier_stakes = set()
    if len(df) >= 2:
        for stake_id, group in df.groupby("stake_id"):
            group = group.sort_values("date").reset_index(drop=True)
            if len(group) < 2:
                continue

            for i in range(1, len(group)):
                p1 = group.iloc[i - 1]
                p2 = group.iloc[i]
                dt_days = (p2["date"] - p1["date"]).total_seconds() / 86400
                if dt_days <= 0:
                    continue

                dx = p2["x"] - p1["x"]
                dy = p2["y"] - p1["y"]
                distance = np.sqrt(dx**2 + dy**2)
                segment_speed = distance / dt_days

                if segment_speed > 5:
                    outlier_stakes.add(stake_id)
                    break

    meta["has_outlier_value"] = meta["stake_id"].isin(outlier_stakes)

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
                "dt_days": None,
                "mean_speed_m_per_year": None
            })
            continue

        total_dx = group["dx"].sum()
        total_dy = group["dy"].sum()
        total_dz = group["dz"].sum()
        total_dt = group["dt_days"].sum()

        total_distance = np.sqrt(total_dx**2 + total_dy**2)
        mean_speed = total_distance / total_dt if total_dt > 0 else None
        vx, vy = estimate_velocity_components(
            stake_segments=group,
        )

        rows.append({
            "stake_id": stake_id,
            "valid_segments": len(group),
            "total_dx_m": total_dx,
            "total_dy_m": total_dy,
            "total_dz_m": total_dz,
            "dt_days": round(total_dt),
            "mean_speed_m_per_year": mean_speed * 365 if mean_speed else None,
            "historic_vx_m_per_day": vx,
            "historic_vy_m_per_day": vy,
        })

    summary_disp = pd.DataFrame(rows)

    summary = meta.merge(summary_disp, on="stake_id", how="left")
    summary["mean_speed_m_per_year"] = summary["mean_speed_m_per_year"].round(2)

    ordered_columns = [
        "stake_id",
        "glacier",
        "n_points",
        "valid_segments",
        "first_date",
        "last_date",
        "dt_days",
        "total_dx_m",
        "total_dy_m",
        "total_dz_m",
        "mean_speed_m_per_year",
        "historic_vx_m_per_day",
        "historic_vy_m_per_day",
        "has_outlier_value",
    ]
    summary = summary[ordered_columns]

    return summary

# Summarize data availability per campaign for each stake
def compute_campaign_summary(df):

    if "campaign" not in df.columns:
        raise ValueError("Missing required column: campaign")

    campaign_meta = (
        df[["campaign"]]
        .dropna(subset=["campaign"])
        .drop_duplicates()
        .sort_values("campaign")
    )

    rows = []

    for stake_id, group in df.groupby("stake_id"):
        counts = group["campaign"].value_counts().to_dict()

        for campaign_row in campaign_meta.itertuples(index=False):
            n = counts.get(campaign_row.campaign, 0)

            if n == 0:
                status = "NO_DATA"
            elif n == 1:
                status = "ONE_MEASUREMENT"
            else:
                status = "MULTIPLE_MEASUREMENTS"

            rows.append({
                "stake_id": stake_id,
                "campaign": campaign_row.campaign,
                "n_measurements": n,
                "status": status
            })

    return pd.DataFrame(rows).sort_values(["stake_id", "campaign"])

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

    rows = []
    for row in last_positions.itertuples(index=False):
        stake_segments = displacements[
            (displacements["stake_id"] == row.stake_id) & (displacements["date_end"] <= row.date)
        ].sort_values("date_end")

        vx_est, vy_est = estimate_velocity_components(
            stake_segments=stake_segments,
        )

        x_pred = row.x + vx_est * row.delta_t_days if pd.notna(vx_est) else np.nan
        y_pred = row.y + vy_est * row.delta_t_days if pd.notna(vy_est) else np.nan
        delta_x = x_pred - row.x if pd.notna(x_pred) else np.nan
        delta_y = y_pred - row.y if pd.notna(y_pred) else np.nan

        rows.append({
            "stake_id": row.stake_id,
            "date": row.date,
            "target_date": pd.to_datetime(target_date),
            "delta_t_days": row.delta_t_days,
            "x": row.x,
            "y": row.y,
            "x_pred": x_pred,
            "y_pred": y_pred,
            "delta_x": delta_x,
            "delta_y": delta_y,
            "vx_est": vx_est,
            "vy_est": vy_est,
        })

    return pd.DataFrame(rows)


# Count number of stakes with data from recent campaigns
def summarize_recent_campaigns(df, n_campaigns=2):
    if "campaign" not in df.columns:
        raise ValueError("Missing required column: campaign")

    campaign_meta = (
        df[["campaign"]]
        .dropna(subset=["campaign"])
        .drop_duplicates()
        .sort_values("campaign")
    )
    recent_campaigns = campaign_meta.tail(n_campaigns)
    recent_campaign_labels = recent_campaigns["campaign"].tolist()

    stakes_with_recent_campaigns = (
        df[df["campaign"].isin(recent_campaign_labels)]
        .groupby("stake_id")["campaign"]
        .nunique()
        .eq(n_campaigns)
        .sum()
    )

    return {
        "campaigns": campaign_meta["campaign"].tolist(),
        "recent_campaigns": recent_campaign_labels,
        "stakes_with_recent_campaigns": stakes_with_recent_campaigns,
    }
