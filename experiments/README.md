# Experiment Run Metrics Format

All training runs log structured evaluation and training telemetry to this directory.

## File Naming Convention

Each training run produces exactly one CSV file named:
```
experiments/<date>_<config>_<tag>.csv
```
* Example: `2026-11-15_small_baseline.csv`
* Example: `2026-11-15_medium_rope-swiglu.csv`

## CSV Columns Specification

The CSV files contain a header row followed by one row per logging step:

```csv
step,train_loss,val_loss,perplexity,lr,tokens_per_sec,elapsed_s,peak_mem_mb
```

### Column Definitions

- **`step`**: The current training iteration / optimizer step counter (integer).
- **`train_loss`**: The cross-entropy training loss calculated over the current micro-batch or gradient accumulation window.
- **`val_loss`**: The cross-entropy validation loss evaluated across held-out evaluation batches (blank or repeated on non-eval steps).
- **`perplexity`**: The exponential of the validation loss ($e^{\text{val\_loss}}$), measuring the model's prediction uncertainty (lower is better).
- **`lr`**: The active learning rate for the current step output by the warmup and cosine decay scheduler.
- **`tokens_per_sec`**: The training throughput measured as total processed tokens divided by step wall-clock execution time.
- **`elapsed_s`**: Cumulative training wall-clock time elapsed in seconds since training started.
- **`peak_mem_mb`**: Peak memory consumption allocated by the framework (MPS/RAM) in megabytes during the step.

## Committing Experiment Logs

Small CSV and JSON metric files in this folder are intended to be committed to version control to preserve experiment history and enable plotting training curves in reports.

