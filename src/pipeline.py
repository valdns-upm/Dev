def export_results(
    displacements,
    issues,
    stakes_summary,
    year_summary,
    prediction,
    validation_summary=None,
    validation_details=None,
):

    displacements.to_csv(
        "outputs/displacements_valid.csv",
        index=False,
        float_format="%.5f"
    )

    issues.to_csv(
        "outputs/trajectory_issues.csv",
        index=False
    )

    stakes_summary.to_csv(
        "outputs/stakes_summary.csv",
        index=False,
        float_format="%.5f"
    )

    year_summary.to_csv(
        "outputs/year_summary.csv",
        index=False
    )

    prediction.to_csv(
        "outputs/predictions.csv",
        index=False
    )

    if validation_summary is not None and not validation_summary.empty:
        validation_summary.to_csv(
            "outputs/validation_half_life_summary.csv",
            index=False
        )

    if validation_details is not None and not validation_details.empty:
        validation_details.to_csv(
            "outputs/validation_half_life_details.csv",
            index=False
        )
