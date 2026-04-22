from src.io import load_multiple_files
from src.trajectory import build_trajectories, compute_displacements
from src.analysis import (
    compute_prediction,
    compute_stake_summary,
    compute_year_summary,
    summarize_recent_campaigns,
    evaluate_half_lives_with_validation,
)
from src.pipeline import export_results
from pathlib import Path

data_path = "data/raw/"
validation_path = "data/validation/"

df = load_multiple_files(data_path)
campaign_summary = summarize_recent_campaigns(df, n_campaigns=2)

trajectories = build_trajectories(df)

# Compute displacements and identify issues
displacements, issues = compute_displacements(trajectories)

# Compute summaries
stakes_summary = compute_stake_summary(df, displacements, issues)
year_summary = compute_year_summary(df)

# Change target date for prediction if needed
predicted_positions = compute_prediction(df, stakes_summary, target_date="2025-12-20")

validation_summary = None
validation_details = None
if any(Path(validation_path).glob("*.xlsx")):
    validation_df = load_multiple_files(validation_path)
    validation_half_lives = [180, 365, 540, 730]
    validation_summary, validation_details = evaluate_half_lives_with_validation(
        train_df=df,
        validation_df=validation_df,
        displacements=displacements,
        stakes_summary=stakes_summary,
        half_lives_days=validation_half_lives,
    )

export_results(
    displacements,
    issues,
    stakes_summary,
    year_summary,
    predicted_positions,
    validation_summary=validation_summary,
    validation_details=validation_details,
)

# Summary statistics
print("Number of stakes monitored:", len(stakes_summary))
print(
    f"Number of stakes with data in the last two campaigns ({campaign_summary['recent_campaigns']}):",
    campaign_summary["stakes_with_recent_campaigns"],
)
print("Number of stakes with one measurement:", (stakes_summary["n_points"] == 1).sum())
print("Number of stakes with outliers:", issues.loc[issues["issue_type"] == "OUTLIER", "stake_id"].nunique())
if validation_summary is not None and not validation_summary.empty:
    best = validation_summary.sort_values("rmse_dist_m").iloc[0]
    print("Validation half-lives tested:", len(validation_summary))
    print("Best half-life (by RMSE distance):", int(best["half_life_days"]), "days")
