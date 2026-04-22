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

        # GLOBAL CASE: use remaining valid segments after cleaning
        if total_dt > 0 and n_segments >= 2:

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


def evaluate_half_lives_with_validation(
    train_df,
    validation_df,
    displacements,
    stakes_summary,
    half_lives_days,
):
    if validation_df is None or validation_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    last_train = (
        train_df.sort_values("date")
        .groupby("stake_id")
        .last()
        .reset_index()[["stake_id", "date", "x", "y"]]
        .rename(columns={"date": "train_date", "x": "x_train", "y": "y_train"})
    )

    first_val = (
        validation_df.sort_values("date")
        .groupby("stake_id")
        .first()
        .reset_index()[["stake_id", "date", "x", "y"]]
        .rename(columns={"date": "val_date", "x": "x_val", "y": "y_val"})
    )

    eval_base = last_train.merge(first_val, on="stake_id", how="inner")
    eval_base = eval_base[eval_base["val_date"] > eval_base["train_date"]].copy()
    if eval_base.empty:
        return pd.DataFrame(), pd.DataFrame()

    seg = displacements.copy()
    if seg.empty:
        return pd.DataFrame(), pd.DataFrame()

    seg["vx"] = seg["dx"] / seg["dt_days"]
    seg["vy"] = seg["dy"] / seg["dt_days"]

    fallback = stakes_summary[
        [
            "stake_id",
            "historic_vx_m_per_day",
            "historic_vy_m_per_day",
            "velocity_method",
            "velocity_quality",
        ]
    ].copy()

    fallback["velocity_method"] = fallback["velocity_method"].replace(
        {"GLACIER_AVERAGE": "GLACIER"}
    )

    fallback_by_stake = fallback.set_index("stake_id").to_dict("index")

    detail_rows = []

    for half_life_days in half_lives_days:
        tau = half_life_days / np.log(2)

        for row in eval_base.itertuples(index=False):
            stake_segments = seg[
                (seg["stake_id"] == row.stake_id) & (seg["date_end"] <= row.train_date)
            ].copy()

            delta_t_days = (row.val_date - row.train_date).days
            method = "NONE"
            quality = "NONE"
            vx_est = np.nan
            vy_est = np.nan

            if len(stake_segments) >= 2 and stake_segments["dt_days"].sum() > 0:
                ages = (row.train_date - stake_segments["date_end"]).dt.days.clip(lower=0)
                weights = stake_segments["dt_days"] * np.exp(-ages / tau)

                vx_est = np.average(stake_segments["vx"], weights=weights)
                vy_est = np.average(stake_segments["vy"], weights=weights)
                method = "WEIGHTED"

                span_days = (stake_segments["date_end"].max() - stake_segments["date_start"].min()).days
                if len(stake_segments) >= 10 and span_days > 365:
                    quality = "HIGH"
                elif len(stake_segments) >= 5:
                    quality = "MEDIUM"
                else:
                    quality = "LOW"
            elif len(stake_segments) == 1:
                vx_est = stake_segments["vx"].iloc[0]
                vy_est = stake_segments["vy"].iloc[0]
                method = "LAST"
                quality = "LOW"
            else:
                fb = fallback_by_stake.get(row.stake_id)
                if fb is not None:
                    vx_est = fb["historic_vx_m_per_day"]
                    vy_est = fb["historic_vy_m_per_day"]
                    method = fb["velocity_method"]
                    quality = fb["velocity_quality"]

            if pd.isna(vx_est) or pd.isna(vy_est):
                continue

            x_pred = row.x_train + vx_est * delta_t_days
            y_pred = row.y_train + vy_est * delta_t_days

            err_x = x_pred - row.x_val
            err_y = y_pred - row.y_val
            err_dist = float(np.sqrt(err_x**2 + err_y**2))

            detail_rows.append({
                "half_life_days": half_life_days,
                "stake_id": row.stake_id,
                "train_date": row.train_date,
                "val_date": row.val_date,
                "delta_t_days": delta_t_days,
                "x_train": row.x_train,
                "y_train": row.y_train,
                "x_val": row.x_val,
                "y_val": row.y_val,
                "vx_est": vx_est,
                "vy_est": vy_est,
                "method": method,
                "quality": quality,
                "x_pred": x_pred,
                "y_pred": y_pred,
                "err_x_m": err_x,
                "err_y_m": err_y,
                "err_dist_m": err_dist,
            })

    details = pd.DataFrame(detail_rows)
    if details.empty:
        return pd.DataFrame(), pd.DataFrame()

    summary = (
        details.groupby("half_life_days")
        .agg(
            n_stakes=("stake_id", "nunique"),
            mean_abs_err_x_m=("err_x_m", lambda s: s.abs().mean()),
            mean_abs_err_y_m=("err_y_m", lambda s: s.abs().mean()),
            mean_err_dist_m=("err_dist_m", "mean"),
            rmse_dist_m=("err_dist_m", lambda s: float(np.sqrt(np.mean(np.square(s))))),
            median_err_dist_m=("err_dist_m", "median"),
        )
        .reset_index()
        .sort_values("half_life_days")
    )

    return summary, details
