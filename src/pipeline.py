def export_results(displacements, issues, stakes_summary, year_summary):

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