README — Estacas

Objective
    Reads Excel files of stake measurements (cf. UPM Drive) for Johnsons & Hurd, and calculates displacements and velocities between successive measurements. 
    It also flags any problematic data.

Input data
    .xlsx files with the following tabs:
        Hurd stakes
        Johnson stakes
    Columns used: ID, Fecha, X, Y, Z

Function
    Automatic reading of all files
    Data cleaning (dates, missing values...)
    Reconstruction of trajectories for each stake
    Calculation of displacements/velocities
    Detection of problems (missing data, gaps...)

Outputs
    displacements_full.csv
    → displacements + velocities between two measurements
    trajectory_issues.csv
    → detected issues (duplicate dates, long gaps, outliers, etc.)
    stakes_summary.csv
    → summary by stake (number of points, period, gaps)

Notes
    Dates are standardised (YY → YYYY)
    Incomplete trajectories are retained