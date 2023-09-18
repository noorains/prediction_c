from pathlib import Path
import yaml
from CompetitionEvaluation import load_data, structure_data, calculate_metrics
from dataclasses import dataclass
import logging
import os

def evaluate_forecast(forecast_file: str | os.PathLike) -> None:

    @dataclass(frozen=True)
    class Actuals:
        target: str
        test_window: str
        parquet_file: str

    actuals_folder = Path("./actuals")
    actuals = [Actuals("cm", "test_window_2018", "cm_actuals_2018.parquet"),
    Actuals("cm", "test_window_2019", "cm_actuals_2019.parquet"),
    Actuals("cm", "test_window_2020", "cm_actuals_2020.parquet"),
    Actuals("cm", "test_window_2021", "cm_actuals_2021.parquet"),
    Actuals("pgm", "test_window_2018", "pgm_actuals_2018.parquet"),
    Actuals("pgm", "test_window_2019", "pgm_actuals_2019.parquet"),
    Actuals("pgm", "test_window_2020", "pgm_actuals_2020.parquet"),
    Actuals("pgm", "test_window_2021", "pgm_actuals_2021.parquet")]

    _, name, target, window = forecast_file.parent.parts
    actual = [actual for actual in actuals if actual.target == target and actual.test_window == window]

    if target == "pgm":
        target_column = "priogrid_gid"
    elif target == "cm":
        target_column = "country_id"
    else:
        raise ValueError(f'Target {target} must be either "pgm" or "cm".')

    if len(actual) != 1:
        raise ValueError("Only one hit allowed.")
    actual = actual[0]
    observed, predictions = load_data(observed_path = actuals_folder/actual.parquet_file, forecasts_path=f)

    if predictions.index.names != ['month_id', target_column, 'draw'] and len(predictions.index.names) == 3:
        logging.warning(f'Predictions file {f} does not have correct index. Currently: {predictions.index.names}')
        logging.warning(f'Attempts to rename index.')
        if len(predictions.index.names) == 3:
            predictions.index.names = ['month_id', target_column, 'draw']

    if len(observed.columns) == 1 and "outcome" not in observed.columns:
        logging.warning(f'Actuals file {actuals_folder/actual.parquet_file} does not have the "outcome" folder.')
        logging.warning(f'Renaming column.')
        observed.columns = ["outcome"]

    if len(predictions.columns) == 1 and "outcome" not in observed.columns:
        logging.warning(f'Predictions file {forecast_file} does not have the "outcome" folder.')
        logging.warning(f'Renaming column.')
        predictions.columns = ["outcome"]

    if len(predictions.columns) != 1:
        raise ValueError("Predictions file can only have 1 column.")

    if len(observed.columns) != 1:
        raise ValueError("Actuals file can only have 1 column.")

    observed, predictions = structure_data(observed, predictions, draw_column_name="draw", data_column_name = "outcome")

    predictions[target_column] in observed[target_column]
    predictions["month_id"] in observed["month_id"]

    crps_per_unit = calculate_metrics(observed, predictions, metric = "crps", aggregate_over="month_id")
    ign_per_unit = calculate_metrics(observed, predictions, metric = "ign", bins = [0, 0.5, 2.5, 5.5, 10.5, 25.5, 50.5, 100.5, 250.5, 500.5, 1000.5], aggregate_over="month_id")
    mis_per_unit = calculate_metrics(observed, predictions, metric = "mis", prediction_interval_level = 0.9, aggregate_over="month_id")

    crps_per_month = calculate_metrics(observed, predictions, metric = "crps", aggregate_over=target_column)
    ign_per_month = calculate_metrics(observed, predictions, metric = "ign", bins = [0, 0.5, 2.5, 5.5, 10.5, 25.5, 50.5, 100.5, 250.5, 500.5, 1000.5], aggregate_over=target_column)
    mis_per_month = calculate_metrics(observed, predictions, metric = "mis", prediction_interval_level = 0.9, aggregate_over=target_column)

    eval_path = forecast_file.parent/"eval"
    eval_path.mkdir(exist_ok=True)
    crps_per_unit.to_parquet(eval_path/"crps_per_unit.parquet")
    crps_per_month.to_parquet(eval_path/"crps_per_month.parquet")
    ign_per_unit.to_parquet(eval_path/"ign_per_unit.parquet")
    ign_per_month.to_parquet(eval_path/"ign_per_month.parquet")
    mis_per_unit.to_parquet(eval_path/"mis_per_unit.parquet")
    mis_per_month.to_parquet(eval_path/"mis_per_month.parquet")

submission_path = Path("./submissions/")
submissions = [submission for submission in submission_path.iterdir() if submission.is_dir()]

for submission in submissions:
    with open(submission/"submission_details.yml") as f:
        submission_details = yaml.safe_load(f)

    prediction_files = list(submission.glob("**/*.parquet"))
    [evaluate_forecast(f) for f in prediction_files]




