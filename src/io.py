import re
from pathlib import Path

import pandas as pd


def normalize_date_value(value):
    value = str(value).strip().replace("/", "-")
    match = re.match(r"(\d{2}-\d{2}-)(\d{2})$", value)
    if match:
        year = int(match.group(2))
        year_full = 2000 + year if year < 50 else 1900 + year
        return f"{match.group(1)}{year_full}"
    return value


def parse_date_series(series):
    normalized = series.apply(normalize_date_value)
    return pd.to_datetime(
        normalized,
        format="%d-%m-%Y",
        errors="coerce"
    )


def infer_file_metadata(file_path):
    file_name = Path(file_path).name
    match = re.search(r"estacas(\d{2})(\d{2})([a-z]?)", file_name, flags=re.IGNORECASE)

    if not match:
        return {
            "campaign": None,
            "phase": None,
            "source_order": None,
        }

    start_year = 2000 + int(match.group(1))
    end_year = 2000 + int(match.group(2))
    phase = (match.group(3) or "").lower() or None
    phase_order = 1 if phase == "a" else 2 if phase == "b" else 0

    return {
        "campaign": f"{start_year}-{end_year}",
        "phase": phase,
        "source_order": start_year * 10 + phase_order,
    }


def read_stakes_sheet(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    df.columns = df.iloc[1]
    df = df.iloc[4:].reset_index(drop=True)
    return df


def clean_stakes_df(df):

    required_cols = ["X (E-UTM)", "Y (N-UTM)", "Z (WGS84)"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    df = df.rename(columns={
        "Id. estaca": "stake_id",
        "Fecha": "date_raw",
        "X (E-UTM)": "x",
        "Y (N-UTM)": "y",
        "Z (WGS84)": "z"
    })

    df["stake_id"] = df["stake_id"].astype(str).str.strip()

    df = df[["stake_id", "date_raw", "x", "y", "z"]]

    df = df.dropna(subset=["stake_id", "date_raw", "x", "y"])

    df["date"] = parse_date_series(df["date_raw"])

    if df["date"].isna().sum() > 0:
        print("Warning: unparsed dates detected")

    df = df.dropna(subset=["date"])

    def infer_glacier(s):
        if s.startswith("EH"):
            return "hurd"
        if s.startswith("EJ"):
            return "johnson"
        return None

    df["glacier"] = df["stake_id"].apply(infer_glacier)

    return df


def infer_campaign_from_file(file_path):
    return infer_file_metadata(file_path)["campaign"]


def extract_monitoring_metadata(file_path):
    sheets = ["Estacas Hurd", "Estacas Johnsons"]
    file_meta = infer_file_metadata(file_path)
    rows = []

    for sheet in sheets:
        df_raw = read_stakes_sheet(file_path, sheet)
        if "Id. estaca" not in df_raw.columns:
            continue

        gps_comment_col = None
        for col in df_raw.columns:
            if isinstance(col, str) and col.strip() == "Comentarios sobre medida GPS":
                gps_comment_col = col
                break

        monitoring_df = pd.DataFrame({
            "stake_id": df_raw["Id. estaca"].astype(str).str.strip(),
            "date_raw": df_raw["Fecha"],
            "gps_comment": df_raw[gps_comment_col] if gps_comment_col else None,
            "x": df_raw["X (E-UTM)"] if "X (E-UTM)" in df_raw.columns else None,
            "y": df_raw["Y (N-UTM)"] if "Y (N-UTM)" in df_raw.columns else None,
        })
        monitoring_df = monitoring_df.dropna(subset=["stake_id"])
        monitoring_df = monitoring_df[monitoring_df["stake_id"] != ""].copy()
        monitoring_df["date"] = parse_date_series(monitoring_df["date_raw"])
        monitoring_df["source_file"] = Path(file_path).name
        monitoring_df["campaign"] = file_meta["campaign"]
        monitoring_df["phase"] = file_meta["phase"]
        monitoring_df["source_order"] = file_meta["source_order"]
        monitoring_df["gps_comment"] = monitoring_df["gps_comment"].fillna("").astype(str).str.strip()
        monitoring_df["has_measurement"] = (
            monitoring_df["date"].notna()
            & monitoring_df["x"].notna()
            & monitoring_df["y"].notna()
        )
        monitoring_df["is_lost"] = monitoring_df["gps_comment"].str.contains(
            "ESTACA PERDIDA",
            case=False,
            na=False,
        )
        monitoring_df["is_new"] = monitoring_df["gps_comment"].str.contains(
            "ESTACA NUEVA",
            case=False,
            na=False,
        )
        rows.append(
            monitoring_df[
                [
                    "stake_id",
                    "date",
                    "source_file",
                    "campaign",
                    "phase",
                    "source_order",
                    "gps_comment",
                    "has_measurement",
                    "is_lost",
                    "is_new",
                ]
            ]
        )

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def load_single_file(file_path):
    sheets = ["Estacas Hurd", "Estacas Johnsons"]
    file_meta = infer_file_metadata(file_path)

    dfs = []

    for sheet in sheets:
        df_raw = read_stakes_sheet(file_path, sheet)
        df_clean = clean_stakes_df(df_raw)
        df_clean["source_file"] = Path(file_path).name
        df_clean["campaign"] = file_meta["campaign"]
        df_clean["phase"] = file_meta["phase"]
        df_clean["source_order"] = file_meta["source_order"]
        dfs.append(df_clean)

    return pd.concat(dfs, ignore_index=True)


def load_multiple_files(folder_path):
    dfs = []

    for file in Path(folder_path).glob("*.xlsx"):
        df = load_single_file(file)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def load_monitoring_metadata(folder_path):
    dfs = []

    for file in Path(folder_path).glob("*.xlsx"):
        df = extract_monitoring_metadata(file)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)
