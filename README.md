README - Estacas

Objective
    Read Excel files of stake measurements (Hurd and Johnson), compute cleaned trajectories,
    estimate historical stake velocity, and project stake positions to a target date.

Input data
    .xlsx files with tabs:
        Estacas Hurd
        Estacas Johnsons
    Columns used: Id. estaca, Fecha, X (E-UTM), Y (N-UTM), Z (WGS84)

Workflow
    Automatic loading of all Excel files (in data/raw/)
    Date & missing-value cleaning
    Trajectory reconstruction by stake
    Segment displacement/speed calculation
    Point-level outlier handling:
        Abnormal measurement points are removed
        Remaining valid points are kept for calculations
    Velocity model by stake:
        GLOBAL when enough valid segments exist
        GLACIER_AVERAGE fallback otherwise
    Linear position prediction to a target date

Outputs (outputs/)
    displacements_valid.csv
    -> valid displacements and segment speeds
    trajectory_issues.csv
    -> detected issues (ONLY_ONE_POINT, DUPLICATE_DATE, OUTLIER)
    year_summary.csv
    -> data availability by stake and year
    stakes_summary.csv
    -> summary by stake + historical velocity method/quality
    predictions.csv
    -> projected position (x_pred, y_pred) from historic velocity

Notes
    Dates are normalized before parsing (dd-mm-yy -> dd-mm-yyyy).
    Stakes with outliers can still use GLOBAL if enough valid data remains after cleaning.
