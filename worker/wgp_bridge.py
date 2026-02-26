"""Bridge module for importing and wrapping wgp.py.

wgp.py executes significant module-level code on import (arg parsing, GPU
detection, config file I/O).  This module patches the environment so that
import succeeds cleanly inside a headless worker process.
"""

import json
import logging
import os
import sys
from typing import List, Optional, Tuple

from PIL import Image

from .config import WorkerConfig

logger = logging.getLogger(__name__)

# Will be set after bootstrap
wgp = None


def bootstrap_wgp(config: WorkerConfig):
    """Prepare the environment and import wgp.py.

    Must be called exactly once, before any other wgp_bridge function.
    """
    global wgp

    # 1. CWD must be the project root — wgp.py opens relative paths like
    #    "models/_settings.json" and "defaults/*.json".
    os.chdir(config.wan2gp_root)
    logger.info("CWD set to %s", config.wan2gp_root)

    # 2. Ensure output directory exists
    os.makedirs(config.output_dir, exist_ok=True)

    # 3. Create/update wgp_config.json so that module-level config loading
    #    picks up our desired settings.
    config_path = os.path.join(config.wan2gp_root, "wgp_config.json")
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            server_cfg = json.load(f)
    else:
        server_cfg = {}

    server_cfg["attention_mode"] = config.attention_mode
    server_cfg["transformer_quantization"] = config.quantization
    server_cfg["text_encoder_quantization"] = config.quantization
    server_cfg["save_path"] = config.output_dir
    server_cfg["image_save_path"] = config.output_dir
    server_cfg["audio_save_path"] = config.output_dir
    server_cfg["notification_sound_enabled"] = 0
    server_cfg["mmaudio_mode"] = 0

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(server_cfg, f)
    logger.info("Wrote %s", config_path)

    # 4. Patch sys.argv so _parse_args() succeeds
    sys.argv = [
        "wgp.py",
        "--profile", str(config.mmgp_profile),
        "--attention", config.attention_mode,
        "--output-dir", config.output_dir,
    ]

    # 5. Add project root to sys.path
    if config.wan2gp_root not in sys.path:
        sys.path.insert(0, config.wan2gp_root)

    # 6. Import wgp (triggers module-level init)
    logger.info("Importing wgp...")
    import wgp as _wgp
    wgp = _wgp

    # 7. Override save paths at module level to be sure
    wgp.server_config["save_path"] = config.output_dir
    wgp.server_config["image_save_path"] = config.output_dir
    wgp.server_config["audio_save_path"] = config.output_dir
    wgp.save_path = config.output_dir
    wgp.image_save_path = config.output_dir
    wgp.audio_save_path = config.output_dir

    # 8. Ensure ffmpeg is available (needed for video encoding)
    from shared.ffmpeg_setup import download_ffmpeg
    download_ffmpeg()

    logger.info("wgp bootstrap complete")


def preload_model(model_type: str):
    """Load the model into GPU memory once. Subsequent generations skip loading."""
    assert wgp is not None, "Call bootstrap_wgp() first"

    logger.info("Preloading model: %s", model_type)
    output_type = wgp.get_output_type_for_model(model_type, 0)
    wgp.wan_model, wgp.offloadobj = wgp.load_models(model_type, output_type=output_type)
    wgp.transformer_type = model_type
    wgp.reload_needed = False
    logger.info("Model preloaded successfully")


def create_state() -> dict:
    """Create a fresh CLI-mode state dict (isolated per job)."""
    return {
        "gen": {
            "queue": [],
            "in_progress": False,
            "file_list": [],
            "file_settings_list": [],
            "audio_file_list": [],
            "audio_file_settings_list": [],
            "selected": 0,
            "audio_selected": 0,
            "prompt_no": 0,
            "prompts_max": 0,
            "repeat_no": 0,
            "total_generation": 1,
            "window_no": 0,
            "total_windows": 0,
            "progress_status": "",
            "process_status": "process:main",
        },
        "loras": [],
    }


def build_task(job_params: dict, prompt: str, image_start: Image.Image) -> dict:
    """Build a task dict that validate_task() / process_tasks_cli() expect.

    Starts from primary_settings defaults, applies i2v_2_2 preset defaults,
    then overlays user-supplied job params.
    """
    assert wgp is not None, "Call bootstrap_wgp() first"

    params = wgp.primary_settings.copy()

    # I2V 2.2 preset defaults (from defaults/i2v_2_2.json)
    params["model_type"] = "i2v_2_2"
    params["guidance_phases"] = 2
    params["guidance_scale"] = 3.5
    params["guidance2_scale"] = 3.5
    params["flow_shift"] = 5
    params["switch_threshold"] = 900
    params["masking_strength"] = 0.1
    params["denoising_strength"] = 0.9

    # Apply user overrides from the job
    resolution = job_params.get("resolution")
    if resolution:
        params["resolution"] = resolution

    for key in (
        "video_length", "num_inference_steps", "guidance_scale",
        "guidance2_scale", "guidance_phases", "flow_shift", "seed",
        "batch_size", "negative_prompt", "repeat_generation",
    ):
        if key in job_params:
            params[key] = job_params[key]

    # Set the input image
    params["image_start"] = [image_start]

    task = {
        "params": params,
        "prompt": prompt,
    }
    return task


def run_generation(task: dict, state: dict) -> Tuple[List[str], bool]:
    """Run video generation using process_tasks_cli.

    Returns (output_files, success).
    """
    assert wgp is not None, "Call bootstrap_wgp() first"

    state["gen"]["queue"] = [task]
    success = wgp.process_tasks_cli([task], state)

    output_files = list(state["gen"].get("file_list", []))
    return output_files, success
