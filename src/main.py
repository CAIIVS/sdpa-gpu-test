import argparse
import itertools
import json
from pathlib import Path

import submitit

from src.report import generate_markdown_report
from src.sdpa_test import run_sdpa_test


def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)


def generate_combinations(dims, channels, lengths):
    combinations = []
    for dim, channel, length in itertools.product(dims, channels, lengths):
        input_size = [dim, channel] + [length] * dim
        combinations.append(input_size)
    return combinations


def main(config):
    # setup config
    gpu = config.get("gpu")
    input_dims = config.get("input_dims")
    input_channels = config.get("input_channels")
    input_lengths = config.get("input_lengths")
    backend_configs = config.get("backend_configs")
    output_path = Path(config.get("output_path", "outputs")) / f"{gpu}.md"

    # setup slurm executor
    executor = submitit.AutoExecutor(
        folder="log_test",
    )
    executor.update_parameters(
        slurm_setup=[
            "module load python/3.11.9",
            "VENV=gputest module load uv",
            "uv sync",
        ],
        slurm_cpus_per_task=1,
        slurm_mem_per_cpu="16G",
        slurm_timeout_min=config.get("timeout", 10),
        slurm_partition=config.get("partition", "p_gpu_all"),
        slurm_gres=f"gpu:{gpu}:1",
        slurm_account=config.get("account", "cai_ivs"),
    )

    # execute jobs
    jobs = []
    for input_size in generate_combinations(input_dims, input_channels, input_lengths):
        for backend_config in backend_configs:
            job = executor.submit(run_sdpa_test, input_size, backend_config)
            jobs.append(job)

    # generate report
    generate_markdown_report(config, [job.result() for job in jobs], output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run SDPA tests with configurations from a JSON file"
    )
    parser.add_argument("--config", type=str, help="Path to the JSON config file")

    args = parser.parse_args()
    config = load_config(args.config)

    main(config)
