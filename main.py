from src.io import load_multiple_files
from src.trajectory import build_trajectories, compute_displacements
from src.analysis import build_stakes_summary, build_year_summary, compute_recent_velocity, summarize_stakes
from src.pipeline import export_results

# To change if needed, but for now we assume all files are in the same folder and have the same format.
data_path = "data/raw/"
df = load_multiple_files(data_path)

trajectories = build_trajectories(df)
displacements, issues = compute_displacements(trajectories)

summary = summarize_stakes(df)
stakes_summary = build_stakes_summary(displacements, df)
year_summary = build_year_summary(df)


export_results(displacements, issues, stakes_summary, year_summary)

print("Number of stakes monitored:", len(summary))
print("Number of stakes with full trajectories:", summary["has_missing_year"].sum())
print("Number of issues:", len(issues))