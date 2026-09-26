# How to Use the CSV to GMIC Converter

## Basic Usage

Convert your CSV file to GMIC pickle format:

```bash
python csv_to_gmic_converter.py \
    --input-csv sample_gmic_data.csv \
    --output-dir ./gmic_output
```

## With Enhanced Labeling (Recommended)

If your CSV includes tumor laterality information:

```bash
python csv_to_gmic_converter.py \
    --input-csv sample_gmic_data.csv \
    --output-dir ./gmic_output \
    --tumor-laterality-col tumor_laterality \
    --enhanced-labels
```

## Command Line Options

- `--input-csv` - Path to your CSV file (required)
- `--output-dir` - Directory where pickle files will be saved (required)
- `--tumor-laterality-col` - Column name containing tumor side ('L', 'R', or empty)
- `--enhanced-labels` - Use enhanced labeling logic with tumor laterality
- `--prefer-view-labels` - Use view-level labels over exam-level labels (default: True)

## Output

The converter will create separate pickle files for each data split:
- `exam_list_train.pkl`
- `exam_list_dev.pkl` 
- `exam_list_test.pkl`

## Example with Your Data Format

Based on the `sample_gmic_data.csv` format with `exam_level_label` and `view_level_label` columns:

```bash
python csv_to_gmic_converter.py \
    --input-csv sample_gmic_data.csv \
    --output-dir ./output \
    --tumor-laterality-col tumor_laterality \
    --enhanced-labels \
    --prefer-view-labels
```

This will use the view-level labels to create accurate breast-side labels for GMIC.