from src.io import load_multiple_files
from src.trajectory import build_trajectories, compute_displacements
from src.analysis import compute_prediction, compute_stake_summary, compute_year_summary, summarize_recent_campaigns
from src.pipeline import export_results

data_path = "data/raw/"

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

export_results(displacements, issues, stakes_summary, year_summary, predicted_positions)

# Summary statistics
print("Number of stakes monitored:", len(stakes_summary))
print(
    f"Number of stakes with data in the last two campaigns ({campaign_summary['recent_campaigns']}):",
    campaign_summary["stakes_with_recent_campaigns"],
)
print("Number of stakes with one measurement:", (stakes_summary["n_points"] == 1).sum())
print("Number of stakes with outliers:", issues.loc[issues["issue_type"] == "OUTLIER", "stake_id"].nunique())
