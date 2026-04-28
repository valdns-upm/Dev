from pathlib import Path


def _round_existing_columns(df, columns, decimals):
    existing_columns = [col for col in columns if col in df.columns]
    if existing_columns:
        df[existing_columns] = df[existing_columns].round(decimals)


def export_results(
    displacements,
    issues,
    stakes_summary,
    campaign_summary,
    prediction,
    validation_summary=None,
    validation_details=None,
):
    displacements_export = displacements.drop(
        columns=["annualized_speed"],
        errors="ignore"
    ).copy()

    float_columns = displacements_export.select_dtypes(include="float").columns
    displacements_export[float_columns] = displacements_export[float_columns].round(5)

    if "dt_days" in displacements_export.columns:
        displacements_export["dt_days"] = displacements_export["dt_days"].round().astype("Int64")

    displacements_export.to_csv(
        "outputs/displacements_list.csv",
        index=False,
        float_format="%.5f"
    )

    issues.to_csv(
        "outputs/trajectory_issues.csv",
        index=False
    )

    stakes_export = stakes_summary.copy()

    if "dt_days" in stakes_export.columns:
        stakes_export["dt_days"] = stakes_export["dt_days"].round().astype("Int64")

    if "mean_speed_m_per_year" in stakes_export.columns:
        stakes_export["mean_speed_m_per_year"] = stakes_export["mean_speed_m_per_year"].round(2)

    _round_existing_columns(
        stakes_export,
        ["total_dx_m", "total_dy_m"],
        decimals=3,
    )
    _round_existing_columns(
        stakes_export,
        ["total_dz_m", "historic_vx_m_per_day", "historic_vy_m_per_day"],
        decimals=5,
    )

    stakes_export.to_csv(
        "outputs/stakes_summary.csv",
        index=False
    )

    campaign_summary.to_csv(
        "outputs/campaign_summary.csv",
        index=False
    )

    prediction_export = prediction.copy()
    _round_existing_columns(
        prediction_export,
        ["x", "y", "x_pred", "y_pred", "delta_x", "delta_y"],
        decimals=3,
    )
    prediction_export = prediction_export.drop(
        columns=["vx_est", "vy_est"],
        errors="ignore",
    )

    prediction_export.to_csv(
        "outputs/predictions.csv",
        index=False
    )

    validation_summary_path = Path("outputs/validation_summary.csv")
    validation_details_path = Path("outputs/validation_details.csv")

    if validation_summary is not None and not validation_summary.empty:
        validation_summary_export = validation_summary.copy()
        _round_existing_columns(
            validation_summary_export,
            [
                "mean_abs_err_x_m",
                "mean_abs_err_y_m",
                "mean_err_dist_m",
                "rmse_dist_m",
                "median_err_dist_m",
            ],
            decimals=3,
        )

        validation_summary_export.to_csv(
            validation_summary_path,
            index=False
        )
    elif validation_summary_path.exists():
        validation_summary_path.unlink()

    if validation_details is not None and not validation_details.empty:
        validation_details_export = validation_details.copy()
        _round_existing_columns(
            validation_details_export,
            ["x_start", "y_start", "x_obs", "y_obs", "x_pred", "y_pred"],
            decimals=3,
        )
        _round_existing_columns(
            validation_details_export,
            ["err_x_m", "err_y_m", "err_dist_m"],
            decimals=3,
        )
        validation_details_export = validation_details_export.drop(
            columns=["vx_est", "vy_est"],
            errors="ignore",
        )

        validation_details_export.to_csv(
            validation_details_path,
            index=False
        )
    elif validation_details_path.exists():
        validation_details_path.unlink()
