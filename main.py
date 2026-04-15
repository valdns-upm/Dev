from src.io import load_multiple_files
from src.trajectory import build_trajectories, compute_displacements
from src.analysis import compute_stake_summary, compute_stake_velocity_model, compute_year_summary, compute_stake_velocity
from src.pipeline import export_results

data_path = "data/raw/"

df = load_multiple_files(data_path)

trajectories = build_trajectories(df)

# Compute displacements and identify issues
displacements, issues = compute_displacements(trajectories)

# Compute summaries
stakes_summary = compute_stake_summary(df, displacements, issues)
year_summary = compute_year_summary(df)

# Compute velocities
stake_velocity = compute_stake_velocity(displacements)

velocity_model = compute_stake_velocity_model(displacements, df, issues)
stakes_summary = stakes_summary.merge(
    velocity_model,
    on="stake_id",
    how="left"
)

export_results(displacements, issues, stakes_summary, year_summary)

print("Number of stakes monitored:", len(stakes_summary))
print("Number of stakes with full trajectories:", stakes_summary["has_missing_year"].sum())
print("Number of issues:", len(issues))