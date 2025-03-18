# Reproducibility Plot for High-Throughput Chemical Genomics

This Python script generates reproducibility plots using hexbin visualization to assess data quality in high-throughput chemical genomics screens. It helps researchers evaluate replicate consistency by calculating Pearson correlation and confidence intervals.

## Features
- Processes normalized chemical-genomics screening data.
- Computes Pearson correlation between replicates.
- Generates a high-resolution reproducibility plot with a custom color scale.
- Outputs the results as a `.tiff` file.

## Usage
```bash
python reproducibility_plot.py input_data.csv
