import torch
from chuchichaestli.models.attention.self_attention import SelfAttention


def run_sdpa_test(input_size, backend_config):
    # cuda backend options
    torch.backends.cuda.enable_mem_efficient_sdp(backend_config["mem"])
    torch.backends.cuda.enable_flash_sdp(backend_config["flash"])
    torch.backends.cuda.enable_math_sdp(backend_config["math"])

    try:
        # selfattention block and input
        device = torch.device("cuda")
        self_att = SelfAttention(n_channels=input_size[0]).to(device)
        input = torch.rand(input_size, device=device).unsqueeze(0)

        _ = self_att.forward(input, None)
        peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
        return {
            "dimensions": input_size[0],
            "channels": input_size[1],
            "length": input_size[2],
            "success": True,
            "peak memory": round(peak_memory, 3),
            "backend mem": backend_config["mem"],
            "backend flash": backend_config["flash"],
            "backend math": backend_config["math"],
            "error": None,
        }
    except Exception as e:
        return {
            "dimensions": input_size[0],
            "channels": input_size[1],
            "length": input_size[2],
            "success": False,
            "peak memory": None,
            "backend mem": backend_config["mem"],
            "backend flash": backend_config["flash"],
            "backend math": backend_config["math"],
            "error": str(e).replace("\n", " "),
        }
