import pandas as pd
from pathlib import Path
import re


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

    # Normalize dates: convert DD-MM-YY → DD-MM-YYYY
    def normalize_date(d):
        d = str(d).strip().replace("/", "-")
        match = re.match(r"(\d{2}-\d{2}-)(\d{2})$", d)
        if match:
            year = int(match.group(2))
            year_full = 2000 + year if year < 50 else 1900 + year
            return f"{match.group(1)}{year_full}"
        return d

    df["date_raw"] = df["date_raw"].apply(normalize_date)

    # Strict parsing after normalization
    df["date"] = pd.to_datetime(
        df["date_raw"],
        format="%d-%m-%Y",
        errors="coerce"
    )

    if df["date"].isna().sum() > 0:
        print("Warning: unparsed dates detected")

    df = df.dropna(subset=["date"])

    def infer_glacier(s):
        if s.startswith("EH"):
            return "hurd"
        elif s.startswith("EJ"):
            return "johnson"
        return None

    df["glacier"] = df["stake_id"].apply(infer_glacier)

    return df


def load_single_file(file_path):
    sheets = ["Estacas Hurd", "Estacas Johnsons"]

    dfs = []

    for sheet in sheets:
        df_raw = read_stakes_sheet(file_path, sheet)
        df_clean = clean_stakes_df(df_raw)
        dfs.append(df_clean)

    return pd.concat(dfs, ignore_index=True)


def load_multiple_files(folder_path):
    dfs = []

    for file in Path(folder_path).glob("*.xlsx"):
        df = load_single_file(file)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)