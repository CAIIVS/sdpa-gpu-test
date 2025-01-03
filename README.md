# SDPA Test Runner

This project is designed to run SDPA tests using configurations specified in JSON format. It utilizes the `submitit` library to manage job submissions and generate reports in Markdown format.

## Results

| GPU  | Report                          |
|------|---------------------------------|
| V100 | [report](./outputs/v100sxm.md)  |
| H100 | [report](./outputs/h100pcie.md) |
| H200 | [report](./outputs/h200sxm.md)  |


## Installation

1. Clone the repository:
   ```bash
   git clone git@github.com:CAIIVS/sdpa-gpu-test.git
   cd sdpa-gpu-test.git
   ```

## Usage

1. Prepare a JSON configuration file specifying the test parameters.

2. Run the main script with the configuration file:
   ```bash
   sbatch runner.sh configs/v100.json
   ```

3. The results will be saved in the `outputs` directory as a Markdown report.

## Configuration

The JSON configuration file should include the following fields:
- `gpu`: The type of GPU to use.
- `input_dims`: List of input dimensions.
- `input_channels`: List of input channels.
- `input_lengths`: List of input lengths.
- `backend_configs`: List of backend configurations.
- `output_path`: (Optional) Directory to save the output report.
- `timeout`: (Optional) Timeout for each job in minutes.
- `account`: (Optional) SLURM account to use.
- `partition`: (Optional) SLURM partition to use.
