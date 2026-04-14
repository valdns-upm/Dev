import pandas as pd


def summarize_stakes(df):

    rows = []

    for stake_id, group in df.groupby("stake_id"):

        group = group.sort_values("date")

        n_points = len(group)
        first_date = group.iloc[0]["date"]
        last_date = group.iloc[-1]["date"]

        years = group["date"].dt.year.unique()
        min_year = years.min()
        max_year = years.max()

        expected_years = set(range(min_year, max_year + 1))
        observed_years = set(years)

        has_missing_year = expected_years != observed_years

        rows.append({
            "stake_id": stake_id,
            "n_points": n_points,
            "first_date": first_date,
            "last_date": last_date,
            "has_missing_year": has_missing_year
        })

    return pd.DataFrame(rows)

def build_stakes_summary(displacements, df):

    rows = []

    # infos globales issues du df brut
    meta = df.groupby("stake_id").agg({
        "date": ["min", "max", "count"],
        "glacier": "first"
    })

    meta.columns = ["first_date", "last_date", "n_points", "glacier"]
    meta = meta.reset_index()

    for stake_id, group in displacements.groupby("stake_id"):

        group = group.sort_values("date_end")

        total_dx = group["dx"].sum()
        total_dy = group["dy"].sum()
        total_dz = group["dz"].sum()

        total_dt = group["dt_days"].sum()

        total_distance = (total_dx**2 + total_dy**2)**0.5

        if total_dt > 0:
            mean_speed = total_distance / total_dt
        else:
            mean_speed = None

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

    # merge avec meta
    summary = meta.merge(summary_disp, on="stake_id", how="left")

    # ajouter has_missing_year (reprendre fonction existante)
    year_check = summarize_stakes(df)[["stake_id", "has_missing_year"]]

    summary = summary.merge(year_check, on="stake_id", how="left")

    return summary


def build_year_summary(df):

    rows = []

    for stake_id, group in df.groupby("stake_id"):

        group = group.sort_values("date")

        years_present = group["date"].dt.year.value_counts().to_dict()

        min_year = group["date"].dt.year.min()
        max_year = group["date"].dt.year.max()

        for year in range(min_year, max_year + 1):

            count = years_present.get(year, 0)

            if count == 0:
                status = "NO_DATA"
            elif count == 1:
                status = "ONE_MEASUREMENT"
            else:
                status = "MULTIPLE_MEASUREMENTS"

            rows.append({
                "stake_id": stake_id,
                "year": year,
                "n_measurements": count,
                "status": status
            })

    return pd.DataFrame(rows).sort_values(["stake_id", "year"])


def compute_recent_velocity(displacements, stake_summary, n_segments=2):

    rows = []

    # créer un mapping rapide stake_id → has_missing_year
    missing_map = dict(
        zip(stake_summary["stake_id"], stake_summary["has_missing_year"])
    )

    for stake_id, group in displacements.groupby("stake_id"):

        group = group.sort_values("date_end")

        if "status" in group.columns:
            group = group[group["status"] == "VALID"]

        has_missing = missing_map.get(stake_id, True)
        is_full = not has_missing

        if len(group) == 0:
            rows.append({
                "stake_id": stake_id,
                "vx": None,
                "vy": None,
                "vz": None,
                "speed_m_per_day": None,
                "speed_m_per_year": None,
                "n_segments_used": 0,
                "is_full_trajectory": is_full
            })
            continue

        recent = group.tail(n_segments)

        dt_sum = recent["dt_days"].sum()

        if dt_sum == 0:
            rows.append({
                "stake_id": stake_id,
                "vx": None,
                "vy": None,
                "vz": None,
                "speed_m_per_day": None,
                "speed_m_per_year": None,
                "n_segments_used": 0,
                "is_full_trajectory": is_full
            })
            continue

        dx_sum = recent["dx"].sum()
        dy_sum = recent["dy"].sum()
        dz_sum = recent["dz"].sum()

        vx = dx_sum / dt_sum
        vy = dy_sum / dt_sum
        vz = dz_sum / dt_sum

        speed = (vx**2 + vy**2)**0.5

        rows.append({
            "stake_id": stake_id,
            "vx": vx,
            "vy": vy,
            "vz": vz,
            "speed_m_per_day": speed,
            "speed_m_per_year": speed * 365,
            "n_segments_used": len(recent),
            "is_full_trajectory": is_full
        })

    return pd.DataFrame(rows)