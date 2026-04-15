import pandas as pd

def compute_stake_summary(df, displacements, issues):

    meta = df.groupby("stake_id").agg(
        first_date=("date", "min"),
        last_date=("date", "max"),
        n_points=("date", "count"),
        glacier=("glacier", "first")
    ).reset_index()

    # detect missing years
    def has_gap(group):
        years = group["date"].dt.year.unique()
        return set(range(years.min(), years.max() + 1)) != set(years)

    gaps = df.groupby("stake_id").apply(has_gap).reset_index(name="has_missing_year")

    # detect stakes w/ outliers
    outlier_stakes = set(
        issues[issues["issue_type"] == "OUTLIER"]["stake_id"]
    )

    rows = []

    for stake_id, group in displacements.groupby("stake_id"):

        group = group.sort_values("date_end")

        if stake_id in outlier_stakes or len(group) == 0:
            rows.append({
                "stake_id": stake_id,
                "n_segments": 0,
                "total_dx": None,
                "total_dy": None,
                "total_dz": None,
                "total_distance": None,
                "total_dt_days": None,
                "mean_speed_m_per_day": None,
                "mean_speed_m_per_year": None
            })
            continue

        total_dx = group["dx"].sum()
        total_dy = group["dy"].sum()
        total_dz = group["dz"].sum()

        total_dt = group["dt_days"].sum()

        total_distance = (total_dx**2 + total_dy**2)**0.5

        mean_speed = total_distance / total_dt if total_dt > 0 else None

        rows.append({
            "stake_id": stake_id,
            "n_segments": len(group),
            "total_dx": total_dx,
            "total_dy": total_dy,
            "total_dz": total_dz,
            "total_distance": total_distance,
            "total_dt_days": total_dt,
            "mean_speed_m_per_day": mean_speed,
            "mean_speed_m_per_year": mean_speed * 365 if mean_speed else None
        })

    summary_disp = pd.DataFrame(rows)

    summary = meta.merge(summary_disp, on="stake_id", how="left")
    summary = summary.merge(gaps, on="stake_id", how="left")

    return summary


def compute_year_summary(df):

    rows = []

    for stake_id, group in df.groupby("stake_id"):

        counts = group["date"].dt.year.value_counts().to_dict()

        min_year = group["date"].dt.year.min()
        max_year = group["date"].dt.year.max()

        for year in range(min_year, max_year + 1):

            n = counts.get(year, 0)

            if n == 0:
                status = "NO_DATA"
            elif n == 1:
                status = "ONE_MEASUREMENT"
            else:
                status = "MULTIPLE_MEASUREMENTS"

            rows.append({
                "stake_id": stake_id,
                "year": year,
                "n_measurements": n,
                "status": status
            })

    return pd.DataFrame(rows).sort_values(["stake_id", "year"])
