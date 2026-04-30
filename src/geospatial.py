from pathlib import Path

DEFAULT_CRS = "EPSG:32720"

def _load_geospatial_dependencies():
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError(
            "GeoPackage export requires geopandas. Install geopandas, shapely and "
            "a GeoPackage IO backend such as pyogrio or fiona."
        ) from exc

    try:
        from shapely.geometry import LineString, Point
    except ImportError as exc:
        raise ImportError(
            "GeoPackage export requires shapely."
        ) from exc

    return gpd, LineString, Point


def _empty_geodataframe(columns, crs):
    gpd, _, _ = _load_geospatial_dependencies()
    return gpd.GeoDataFrame(columns=columns, geometry="geometry", crs=crs)


def build_historic_layer(cleaned_trajectories, stakes_summary, prediction, crs=DEFAULT_CRS):
    gpd, LineString, _ = _load_geospatial_dependencies()
    predicted_stake_ids = set(
        prediction.loc[prediction["prediction_status"] == "predicted", "stake_id"]
    )

    records = []
    for stake_id, trajectory in cleaned_trajectories.items():
        if stake_id not in predicted_stake_ids:
            continue
        if trajectory is None or trajectory.empty:
            continue

        ordered = trajectory.sort_values("date").reset_index(drop=True)
        if len(ordered) < 2:
            continue

        records.append({
            "stake_id": stake_id,
            "cleaned_points": len(ordered),
            "cleaned_first_date": ordered.iloc[0]["date"],
            "cleaned_last_date": ordered.iloc[-1]["date"],
            "geometry": LineString(list(zip(ordered["x"], ordered["y"]))),
        })

    if not records:
        return _empty_geodataframe(
            [
                "stake_id",
                "cleaned_points",
                "cleaned_first_date",
                "cleaned_last_date",
                "geometry",
            ],
            crs,
        )

    historic = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)
    return historic.merge(stakes_summary.copy(), on="stake_id", how="left")


def build_predictions_layer(prediction, crs=DEFAULT_CRS):
    gpd, LineString, _ = _load_geospatial_dependencies()

    prediction_layer = prediction.dropna(
        subset=["x", "y", "x_pred", "y_pred"]
    ).copy()

    if prediction_layer.empty:
        return _empty_geodataframe(list(prediction.columns) + ["pred_dist_m", "geometry"], crs)

    prediction_layer["pred_dist_m"] = (
        (prediction_layer["x_pred"] - prediction_layer["x"]) ** 2
        + (prediction_layer["y_pred"] - prediction_layer["y"]) ** 2
    ) ** 0.5
    prediction_layer["geometry"] = prediction_layer.apply(
        lambda row: LineString([(row["x"], row["y"]), (row["x_pred"], row["y_pred"])]),
        axis=1,
    )

    return gpd.GeoDataFrame(prediction_layer, geometry="geometry", crs=crs)


def build_unpredicted_points_layer(prediction, crs=DEFAULT_CRS):
    gpd, _, Point = _load_geospatial_dependencies()

    points_layer = prediction[
        prediction["prediction_status"] == "unpredicted"
    ].dropna(subset=["x", "y"]).copy()

    if points_layer.empty:
        return _empty_geodataframe(list(prediction.columns) + ["geometry"], crs)

    points_layer["geometry"] = points_layer.apply(
        lambda row: Point(row["x"], row["y"]),
        axis=1,
    )

    return gpd.GeoDataFrame(points_layer, geometry="geometry", crs=crs)


def build_validation_layer(validation_details, crs=DEFAULT_CRS):
    gpd, LineString, _ = _load_geospatial_dependencies()

    if validation_details is None or validation_details.empty:
        return _empty_geodataframe(["geometry"], crs)

    validation_layer = validation_details.dropna(
        subset=["x_pred", "y_pred", "x_obs", "y_obs"]
    ).copy()

    if validation_layer.empty:
        return _empty_geodataframe(list(validation_details.columns) + ["geometry"], crs)

    validation_layer["geometry"] = validation_layer.apply(
        lambda row: LineString(
            [(row["x_pred"], row["y_pred"]), (row["x_obs"], row["y_obs"])]
        ),
        axis=1,
    )

    return gpd.GeoDataFrame(validation_layer, geometry="geometry", crs=crs)


def export_geopackage(
    cleaned_trajectories,
    stakes_summary,
    prediction,
    validation_details=None,
    output_path="outputs/results.gpkg",
    crs=DEFAULT_CRS,
):
    layers = [
        ("historic", build_historic_layer(cleaned_trajectories, stakes_summary, prediction, crs=crs)),
        ("predictions", build_predictions_layer(prediction, crs=crs)),
        ("unpredicted_points", build_unpredicted_points_layer(prediction, crs=crs)),
        ("validation", build_validation_layer(validation_details, crs=crs)),
    ]
    layers = [(name, layer) for name, layer in layers if not layer.empty]

    if not layers:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        try:
            output_path.unlink()
        except PermissionError:
            print(
                f"Warning: could not overwrite {output_path} because it is locked by another process. "
                "CSV exports were still generated."
            )
            return

    for index, (layer_name, layer) in enumerate(layers):
        mode = "w" if index == 0 else "a"
        layer.to_file(
            output_path,
            layer=layer_name,
            driver="GPKG",
            mode=mode,
            index=False,
        )
