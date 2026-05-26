# Machine Breakdown Prediction - Working Documentation

## Project Purpose

This project predicts machine breakdown timing from sensor readings. It trains a Remaining Useful Life (RUL) regression model using historical machine data, then applies that trained model to a new machine CSV to estimate how many cycles are left before failure.

In simple terms, the project answers:

- How much useful life is still left in the machine?
- When should an early warning be raised?
- Which sensor behavior is contributing most to the prediction?
- How close are predicted RUL values to actual RUL values when actual values are available?

## Complete Workflow

1. The user runs `python app.py`.
2. The program asks for a historical training CSV if a saved model is not already available, or if retraining is forced.
3. The program reads the training data and finds the target RUL column.
4. Sensor readings are converted into model features such as current value, difference, exponential moving average, rolling mean, rolling standard deviation, rolling minimum, and rolling maximum.
5. Three regression models are trained and validated.
6. The model with the lowest validation MAE is saved to `machine_breakdown_model.pkl`.
7. The user provides a new machine CSV.
8. The saved model predicts RUL for each row of the new machine data.
9. The program creates alert levels, early-warning information, predicted breakdown timing, plots, metrics, CSV output, JSON output, and a Streamlit dashboard.

## Expected CSV Format

The training CSV must contain numeric sensor columns. Recommended names are:

- `vibration_sensor`
- `temperature_sensor`
- `pressure_sensor`
- `rpm_sensor`
- `bearing_wear_sensor`

The training CSV must also contain one actual RUL target column. Supported target names are:

- `actual_remaining_life`
- `remaining_useful_life`
- `remaining_life`
- `rul`

The prediction CSV should contain the same sensor columns. If it also contains actual RUL, the program calculates prediction accuracy and draws an actual vs predicted graph. If it does not contain actual RUL, the program still predicts the breakdown timeline and draws the predicted RUL curve.

## Model Design

This is a supervised machine learning regression project. The model learns the relationship between machine sensor patterns and Remaining Useful Life.

The model intentionally does not use timestamp or row number as a shortcut for learning. Timestamp or cycle number is used only for plotting the graph on a cycle-based x-axis. The prediction itself learns from the machine condition shown by sensor behavior:

- Current sensor value shows the present condition.
- Difference shows sudden increase or decrease from the previous row.
- Exponential moving average captures recent trend smoothly.
- Rolling mean captures local average behavior.
- Rolling standard deviation captures instability or fluctuation.
- Rolling minimum and maximum capture local operating range.

The target RUL is normalized during training by dividing it by the maximum RUL in the training set. This helps the model learn a general degradation pattern and then scale predictions back to cycles.

## Breakdown-Affecting Factors

The model studies sensor behavior to understand whether the machine is healthy, degrading, or close to breakdown. These are the main factors that can affect predicted breakdown:

- `vibration_sensor`: High or increasing vibration is usually one of the strongest signs of mechanical looseness, imbalance, shaft misalignment, bearing damage, or internal wear. If vibration rises while RUL falls, it means the model is treating vibration as a breakdown indicator.
- `bearing_wear_sensor`: Bearing wear directly represents mechanical degradation. Higher bearing wear usually means more friction, more heat, more vibration, and lower remaining life.
- `temperature_sensor`: Increasing temperature can show overheating, high friction, poor lubrication, overload, or cooling problems. Temperature alone may not always mean failure, but temperature combined with vibration or bearing wear is important.
- `pressure_sensor`: Unstable or abnormal pressure can show flow restriction, leakage, pump/compressor stress, or process imbalance. Pressure variation is useful because breakdown is often linked to instability, not only high values.
- `rpm_sensor`: RPM changes can show load variation, motor stress, slipping, unstable operation, or speed-control problems. If RPM becomes irregular while other sensors worsen, the model may reduce predicted RUL.
- Sensor difference features such as `vibration_sensor_diff`: These show sudden jumps. A sudden increase in vibration, temperature, pressure, or wear can indicate a developing fault.
- Exponential moving average features such as `vibration_sensor_ema_10`: These show the recent trend while reducing random noise. They help answer whether the sensor has been steadily increasing or decreasing.
- Rolling mean features such as `vibration_sensor_mean_30`: These show average behavior over a short recent window. A high rolling mean means the sensor has stayed high, not just spiked once.
- Rolling standard deviation features such as `pressure_sensor_std_30`: These show instability. A machine that fluctuates heavily can be less healthy than a machine with stable readings.
- Rolling minimum and maximum features: These show the recent operating range. A wide range can indicate unstable machine behavior.

For presentation, explain it like this: the model is not only checking one sensor value at one row. It checks current condition, recent trend, sudden change, and stability. Breakdown prediction becomes stronger when multiple signs point in the same direction, for example increasing vibration plus increasing bearing wear plus unstable pressure.

## Models Used

The program compares three regression models:

- `HistGradientBoostingRegressor`: a boosted tree model that handles nonlinear sensor degradation patterns well.
- `ExtraTreesRegressor`: an ensemble of randomized decision trees that is strong for tabular sensor data.
- `Ridge`: a regularized linear regression model used as a simpler baseline.

The data is split in time order. Earlier rows are used for training and later rows are used for validation. This is better for machine-life data than random splitting because future rows should not be mixed into the past.

The selected model is the one with the lowest validation MAE. MAE means Mean Absolute Error, or the average difference between actual RUL and predicted RUL.

## Prediction And Alert Logic

For each row in the new machine CSV, the model predicts `Predicted_RUL`.

The program then adds:

- `Predicted_Breakdown_In_Cycles`: predicted cycles remaining.
- `Predicted_Breakdown_Time`: cycles converted into time using seconds per cycle.
- `Alert_Level`: current condition label.
- `Prediction_Error`: predicted RUL minus actual RUL, when actual RUL exists.

Alert levels are decided from thresholds:

- `RUNNING`: predicted RUL is above the warning threshold.
- `EARLY WARNING`: predicted RUL is below the warning threshold.
- `BREAKDOWN IMMINENT`: predicted RUL is very close to zero or below the failure threshold.

By default, the early warning threshold is 12 percent of the prediction scale, with a minimum of 20 cycles. The breakdown threshold is 2 percent of the prediction scale, with a minimum of 5 cycles.

## Actual Vs Predicted Graph

The graph saved at `results/actual_vs_predicted.png` compares actual RUL and predicted RUL against machine cycles. The x-axis represents cycle progression. If the CSV has a numeric `cycle`, `cycles`, `machine_cycle`, `cycle_number`, `timestamp`, or `time` column, that column is used as the cycle axis. If no such column exists, the program uses `0, 1, 2, 3...` as machine cycle positions.

The y-axis represents Remaining Useful Life in cycles. Higher values mean the machine is expected to run longer. Lower values mean the machine is closer to failure.

The plotted values are lightly rolling-smoothed and then drawn as curved spline lines so the chart looks continuous and easier to explain during presentation.

The graph also marks:

- Early warning row
- Predicted breakdown row

If actual RUL is not available in the prediction CSV, only the predicted RUL curve is shown.

How to analyze this graph:

- If the predicted curve is close to the actual curve, the model is predicting well.
- If the predicted curve is above the actual curve, the model is overestimating remaining life. This can be risky because it may predict failure later than reality.
- If the predicted curve is below the actual curve, the model is underestimating remaining life. This is safer for maintenance, but it may create earlier warnings than needed.
- If both curves move downward as cycles increase, the machine degradation pattern is being captured correctly.
- If the predicted curve is flat while actual RUL is falling, the model is missing the degradation pattern in the new data.
- If the predicted curve suddenly drops, check the prediction table and feature importance to see which sensor changed strongly around that cycle.
- The early warning vertical line shows where maintenance planning should start.
- The predicted breakdown vertical line shows where the model thinks the machine is very close to failure.

For explanation, say: the best graph is not necessarily perfectly overlapping, but the predicted line should follow the same downward trend as the actual line and should warn before the actual RUL reaches a dangerous low value.

## How To Run

Install requirements:

```powershell
pip install -r requirements.txt
```

Interactive run:

```powershell
python app.py
```

Example non-interactive run:

```powershell
python app.py --train-data datasets/simulated_training_machine_dataset.csv --predict-data datasets/simulated_testing_machine_dataset.csv --seconds-per-cycle 5
```

Force retraining even if `machine_breakdown_model.pkl` already exists:

```powershell
python app.py --train-data datasets/simulated_training_machine_dataset.csv --predict-data datasets/simulated_testing_machine_dataset.csv --force-retrain
```

For a prediction file that has no actual RUL column, provide the expected maximum RUL or cycle horizon:

```powershell
python app.py --train-data datasets/simulated_training_machine_dataset.csv --predict-data datasets/new_machine_data.csv --rul-scale 900
```

## Generated Files

- `machine_breakdown_model.pkl`: saved trained model bundle.
- `results/machine_failure_predictions.csv`: row-by-row predictions with alert levels.
- `results.json`: final machine status, risk, model name, metrics, and timeline values.
- `results/actual_vs_predicted.png`: curved-line graph comparing actual and predicted RUL.
- `results/feature_importance.png`: bar chart of the most important sensor-derived features.
- `results/feature_importance.csv`: feature importance values in table form.
- `results/shap_summary.png`: SHAP explainability plot when SHAP succeeds.
- `dashboard.py`: generated Streamlit dashboard.

## Dashboard

Run:

```powershell
streamlit run dashboard.py
```

Then open the local URL shown by Streamlit. The dashboard displays machine status, remaining life, failure risk, model name, actual vs predicted graph, feature importance graph, SHAP plot if available, and a prediction table.

## How To Analyze Generated Graphs

The project generates three important visual outputs.

### 1. Actual Vs Predicted Graph

File: `results/actual_vs_predicted.png`

Purpose: This graph shows whether the model prediction follows the real remaining life of the machine.

How to read it:

- X-axis: machine cycles.
- Y-axis: Remaining Useful Life in cycles.
- Actual RUL line: the true remaining life from the dataset.
- Predicted RUL line: the model's estimated remaining life.
- Early warning line: the cycle where the machine enters warning condition.
- Predicted breakdown line: the cycle where the machine is close to failure.

Importance: This graph proves whether the model is practically useful. If the predicted line follows the actual line, the model can support predictive maintenance decisions.

### 2. Feature Importance Graph

File: `results/feature_importance.png`

Purpose: This graph shows which sensor-derived factors affected the prediction most.

How to read it:

- Longer bars mean stronger influence on the model.
- Feature names ending in `_mean_30` describe a 30-row rolling average.
- Feature names ending in `_std_30` describe recent instability.
- Feature names ending in `_diff` describe sudden change.
- Feature names ending in `_ema_10` describe recent smooth trend.

Importance: This graph helps explain why the model predicted breakdown. For example, if `vibration_sensor_mean_30` is the top factor, then sustained vibration behavior was very important for the prediction.

### 3. SHAP Summary Graph

File: `results/shap_summary.png`

Purpose: SHAP explains how features push predictions higher or lower across sample rows.

How to read it:

- Features at the top are usually more influential.
- Points spread widely on the x-axis mean that feature strongly changes predictions.
- The color shows whether the feature value is high or low.
- Points on one side of the center increase the model output, while points on the other side decrease it.

Importance: SHAP gives deeper explainability than normal feature importance. It helps answer not just which feature matters, but how high or low values of that feature influence the prediction.

## Code Summary

`app.py` is the main project file. It handles argument parsing, interactive input, data loading, feature engineering, model training, model selection, prediction, plotting, output file generation, and dashboard generation.

Important functions in `app.py`:

- `parse_args()`: reads command-line options.
- `fill_interactive_args()`: asks the user for missing file paths and cycle time.
- `find_target_column()`: finds the actual RUL column in the CSV.
- `detect_sensor_columns()`: chooses numeric sensor columns and ignores output/leakage columns.
- `build_features()`: creates current, difference, EMA, rolling mean, rolling standard deviation, rolling minimum, and rolling maximum features.
- `candidate_models()`: defines the three regression models.
- `train_model()`: trains all candidate models, compares validation MAE, selects the best model, and saves the model bundle.
- `add_breakdown_timeline()`: creates predicted RUL, alert level, early-warning row, and breakdown row.
- `smooth_line()`: converts actual and predicted series into smooth curved lines for plotting.
- `save_actual_vs_predicted_plot()`: saves the curved actual vs predicted graph.
- `feature_importance()`: calculates the most important model features.
- `save_feature_importance_plot()`: saves the feature importance chart.
- `save_shap_plot()`: creates an explainability plot if SHAP is available.
- `write_dashboard()`: writes the Streamlit dashboard file.
- `write_documentation()`: creates the documentation file if it is missing.
- `run_prediction()`: controls the complete end-to-end workflow.

`create_long_life_dataset.py` creates a larger synthetic training dataset and a long-life test dataset. It simulates sensor degradation by gradually increasing vibration and bearing wear while RUL decreases from a high value toward zero.

`create_test_data.py` creates a healthier test dataset from the simulated testing data by reducing vibration and setting a higher actual RUL.

`debug_predictions.py` quickly checks prediction output by printing minimum, maximum, mean, non-zero count, and first rows of predicted vs actual RUL.

`dashboard.py` is generated by `app.py`. It reads `results.json`, `results/machine_failure_predictions.csv`, and result images, then displays them in Streamlit.

## Explanation For Presentation

This project uses historical sensor data to learn how machine health changes before failure. During training, the model sees sensor values and the actual remaining life. It learns patterns such as increasing vibration, increasing bearing wear, unstable pressure, or changing RPM. During prediction, the model receives new machine sensor readings and estimates the remaining useful life for each row.

The final output is useful for predictive maintenance. Instead of waiting for a sudden breakdown, the system gives an early warning when predicted RUL becomes low. The feature importance chart helps explain which sensor patterns influenced the prediction the most.

## Notes

Accuracy should be close, not artificially perfect. The model avoids leaked output columns such as `Predicted_RUL`, `Actual_RUL`, `RMS`, `EMA`, and old generated columns from earlier runs.
