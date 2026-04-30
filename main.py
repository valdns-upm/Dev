from src.io import load_monitoring_metadata, load_multiple_files
from src.trajectory import build_trajectories, compute_displacements
from src.analysis import (
    compute_prediction,
    compute_stake_summary,
    compute_campaign_summary,
    summarize_recent_campaigns,
)
from src.validation import evaluate_prediction_with_validation
from src.pipeline import export_results
from pathlib import Path

data_path = "data/raw/"
validation_path = "data/validation/"

# Setting to change:
run_validation = True

df = load_multiple_files(data_path)
monitoring_df = load_monitoring_metadata(data_path)
recent_campaigns_summary = summarize_recent_campaigns(df, n_campaigns=2)

trajectories = build_trajectories(df)
cleaned_trajectories, displacements, issues = compute_displacements(trajectories)

# Compute summaries
stakes_summary = compute_stake_summary(df, displacements)
campaign_summary = compute_campaign_summary(df)

# Change target date for prediction if needed
predicted_positions = compute_prediction(
    df,
    displacements,
    target_date="2025-12-20",
    monitoring_df=monitoring_df,
)

# Optional validation
validation_summary = None
validation_details = None
if run_validation and any(Path(validation_path).glob("*.xlsx")):
    validation_df = load_multiple_files(validation_path)
    validation_summary, validation_details = evaluate_prediction_with_validation(
        train_df=df,
        validation_df=validation_df,
        displacements=displacements,
    )
elif run_validation:
    print(f"Validation requested, but no .xlsx files were found in {validation_path}")

export_results(
    cleaned_trajectories,
    displacements,
    issues,
    stakes_summary,
    campaign_summary,
    predicted_positions,
    validation_summary=validation_summary,
    validation_details=validation_details,
)

# Summary statistics
print("Number of stakes monitored:", len(stakes_summary))
print(
    f"Number of stakes with data in the last two campaigns ({recent_campaigns_summary['recent_campaigns']}):",
    recent_campaigns_summary["stakes_with_recent_campaigns"],
)
print("Number of stakes with one measurement:", (stakes_summary["n_points"] == 1).sum())
print("Number of stakes with outliers:", issues.loc[issues["issue_type"] == "OUTLIER", "stake_id"].nunique())

if validation_summary is not None and not validation_summary.empty:
    metrics = validation_summary.iloc[0]
    print("Validation stakes compared:", int(metrics["n_stakes"]))
    print("Validation RMSE distance:", round(float(metrics["rmse_dist_m"]), 4), "m")
