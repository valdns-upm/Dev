import pandas as pd

def compute_stake_summary(df, displacements):

    # base metadata (from raw data)
    meta = df.groupby("stake_id").agg(
        glacier=("glacier", "first"),
        n_points=("date", "count"),
        first_date=("date", "min"),
        last_date=("date", "max"),
    ).reset_index()

    # compute missing years
    def has_gap(group):
        years = group["date"].dt.year.unique()
        return set(range(years.min(), years.max() + 1)) != set(years)

    gaps = df.groupby("stake_id").apply(has_gap).reset_index(name="has_missing_year")

    # displacement aggregation
    disp = displacements.groupby("stake_id").agg(
        total_dx=("dx", "sum"),
        total_dy=("dy", "sum"),
        total_dz=("dz", "sum"),
        total_dt_days=("dt_days", "sum")
    ).reset_index()

    disp["total_distance"] = (disp["total_dx"]**2 + disp["total_dy"]**2)**0.5

    disp["mean_speed_m_per_day"] = disp["total_distance"] / disp["total_dt_days"]
    disp["mean_speed_m_per_year"] = disp["mean_speed_m_per_day"] * 365

    # merge all
    summary = meta.merge(disp, on="stake_id", how="left")
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
