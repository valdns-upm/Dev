from src.io import load_multiple_files
from src.trajectory import build_trajectories, compute_displacements
from src.analysis import compute_stake_summary, compute_year_summary
from src.pipeline import export_results

data_path = "data/raw/"

df = load_multiple_files(data_path)

trajectories = build_trajectories(df)

displacements, issues = compute_displacements(trajectories)

stakes_summary = compute_stake_summary(df, displacements, issues)
year_summary = compute_year_summary(df)

export_results(displacements, issues, stakes_summary, year_summary)

print("Number of stakes monitored:", len(stakes_summary))
print("Number of stakes with full trajectories:", stakes_summary["has_missing_year"].sum())
print("Number of issues:", len(issues))