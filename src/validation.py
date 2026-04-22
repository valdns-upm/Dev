import pandas as pd
import numpy as np

from src.analysis import build_glacier_velocity_stats, estimate_velocity_components


def evaluate_half_lives_with_validation(
    train_df,
    validation_df,
    displacements,
    half_lives_days,
):
    if validation_df is None or validation_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    last_train = (
        train_df.sort_values("date")
        .groupby("stake_id")
        .last()
        .reset_index()[["stake_id", "date", "x", "y", "glacier"]]
        .rename(
            columns={
                "date": "train_date",
                "x": "x_train",
                "y": "y_train",
            }
        )
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

    if displacements.empty:
        return pd.DataFrame(), pd.DataFrame()

    glacier_stats = build_glacier_velocity_stats(displacements)
    detail_rows = []

    for half_life_days in half_lives_days:
        for row in eval_base.itertuples(index=False):
            stake_segments = displacements[
                (displacements["stake_id"] == row.stake_id)
                & (displacements["date_end"] <= row.train_date)
            ].sort_values("date_end")

            delta_t_days = (row.val_date - row.train_date).days
            vx_est, vy_est, method, quality = estimate_velocity_components(
                stake_segments=stake_segments,
                glacier=row.glacier,
                glacier_stats=glacier_stats,
                reference_date=row.train_date,
                half_life_days=half_life_days,
            )

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
