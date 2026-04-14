import numpy as np
import pandas as pd


def build_trajectories(df):

    trajectories = {}

    for stake_id, group in df.groupby("stake_id"):
        group = group.sort_values("date")
        trajectories[stake_id] = group

    return trajectories


def compute_displacements(trajectories):

    rows = []
    issues = []

    max_speed = 5  # m/day

    for stake_id, traj in trajectories.items():

        if len(traj) < 2:
            issues.append({
                "stake_id": stake_id,
                "issue_type": "ONLY_ONE_POINT",
                "date": traj.iloc[0]["date"],
                "details": "single measurement"
            })
            continue

        for i in range(1, len(traj)):

            p1 = traj.iloc[i-1]
            p2 = traj.iloc[i]

            dt_days = (p2["date"] - p1["date"]).total_seconds() / 86400

            if dt_days == 0:
                issues.append({
                    "stake_id": stake_id,
                    "issue_type": "DUPLICATE_DATE",
                    "date": p1["date"],
                    "details": "zero time difference"
                })
                continue

            dx = p2["x"] - p1["x"]
            dy = p2["y"] - p1["y"]
            dz = p2["z"] - p1["z"]

            distance = np.sqrt(dx**2 + dy**2)
            segment_speed = distance / dt_days

            if segment_speed > max_speed:
                issues.append({
                    "stake_id": stake_id,
                    "issue_type": "OUTLIER",
                    "date": p2["date"],
                    "details": f"{segment_speed:.2f} m/day"
                })
                continue

            rows.append({
                "stake_id": stake_id,
                "glacier": p2["glacier"],
                "date_start": p1["date"],
                "date_end": p2["date"],
                "dt_days": dt_days,
                "dx": dx,
                "dy": dy,
                "dz": dz,
                "daily_segment_speed": segment_speed,
                "annualized_speed": segment_speed * 365,
            })

    return pd.DataFrame(rows), pd.DataFrame(issues)
