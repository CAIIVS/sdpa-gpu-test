import pandas as pd


def generate_markdown_report(config, results, output_path):
    df = pd.DataFrame(results)
    with open(output_path, "w") as f:
        f.write(
            f"# {config.get('gpu')} Scaled Dot Product Attention (SDPA) Test Results \n\n"
        )
        f.write(
            f"This report shows the results of testing different SDPA configurations with various input sizes on a {config.get('gpu')} GPU.\n\n"
        )
        f.write("## Table Explanation:\n")
        f.write("- **dimensions**: The dimensions of the input tensor\n")
        f.write("- **chanels**: The channels of the input tensor\n")
        f.write("- **lengths**: The lengths of the input tensor in every dimension\n")
        f.write("- **success**: Whether the SDPA operation completed successfully\n")
        f.write(
            "- **peak memory (GB)**: Maximum GPU memory usage during the SDPA operation\n"
        )
        f.write(
            "- **backend mem**: Whether the memory-efficient SDPA implementation was enabled\n"
        )
        f.write(
            "- **backend flash**: Whether the Flash Attention implementation was enabled\n"
        )
        f.write(
            "- **backend math**: Whether the math SDPA implementation was enabled\n"
        )
        f.write("- **error**: Error message if the operation failed, None otherwise\n")
        for dim, group in df.groupby("dimensions"):
            f.write(f"## Results for {dim} dimensions:\n\n")
            f.write(group.to_markdown(index=False))
            f.write("\n\n")
