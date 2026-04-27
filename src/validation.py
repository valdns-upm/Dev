import pandas as pd
import numpy as np

from src.analysis import estimate_velocity_components


def evaluate_prediction_with_validation(
    train_df,
    validation_df,
    displacements,
):
    if validation_df is None or validation_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    last_start = (
        train_df.sort_values("date")
        .groupby("stake_id")
        .last()
        .reset_index()[["stake_id", "date", "x", "y", "glacier"]]
        .rename(
            columns={
                "date": "start_date",
                "x": "x_start",
                "y": "y_start",
            }
        )
    )

    first_obs = (
        validation_df.sort_values("date")
        .groupby("stake_id")
        .first()
        .reset_index()[["stake_id", "date", "x", "y"]]
        .rename(columns={"date": "obs_date", "x": "x_obs", "y": "y_obs"})
    )

    eval_base = last_start.merge(first_obs, on="stake_id", how="inner")
    eval_base = eval_base[eval_base["obs_date"] > eval_base["start_date"]].copy()
    if eval_base.empty:
        return pd.DataFrame(), pd.DataFrame()

    if displacements.empty:
        return pd.DataFrame(), pd.DataFrame()

    detail_rows = []

    for row in eval_base.itertuples(index=False):
        stake_segments = displacements[
            (displacements["stake_id"] == row.stake_id)
            & (displacements["date_end"] <= row.start_date)
        ].sort_values("date_end")

        delta_t_days = (row.obs_date - row.start_date).days
        vx_est, vy_est = estimate_velocity_components(
            stake_segments=stake_segments,
        )

        if pd.isna(vx_est) or pd.isna(vy_est):
            continue

        x_pred = row.x_start + vx_est * delta_t_days
        y_pred = row.y_start + vy_est * delta_t_days

        err_x = x_pred - row.x_obs
        err_y = y_pred - row.y_obs
        err_dist = float(np.sqrt(err_x**2 + err_y**2))

        detail_rows.append({
            "stake_id": row.stake_id,
            "start_date": row.start_date,
            "obs_date": row.obs_date,
            "delta_t_days": delta_t_days,
            "x_start": row.x_start,
            "y_start": row.y_start,
            "x_obs": row.x_obs,
            "y_obs": row.y_obs,
            "vx_est": vx_est,
            "vy_est": vy_est,
            "x_pred": x_pred,
            "y_pred": y_pred,
            "err_x_m": err_x,
            "err_y_m": err_y,
            "err_dist_m": err_dist,
        })

    details = pd.DataFrame(detail_rows)
    if details.empty:
        return pd.DataFrame(), pd.DataFrame()

    summary = pd.DataFrame([{
        "n_stakes": details["stake_id"].nunique(),
        "mean_abs_err_x_m": details["err_x_m"].abs().mean(),
        "mean_abs_err_y_m": details["err_y_m"].abs().mean(),
        "mean_err_dist_m": details["err_dist_m"].mean(),
        "rmse_dist_m": float(np.sqrt(np.mean(np.square(details["err_dist_m"])))),
        "median_err_dist_m": details["err_dist_m"].median(),
    }])

    return summary, details
