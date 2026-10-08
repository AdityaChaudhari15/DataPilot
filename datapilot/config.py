"""All tunable settings live here, so every run can be reproduced from one object."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib, json


@dataclass(frozen=True)
class ModelConfig:
    name: str = "google/gemma-4-31b-it"
    base_url: str = "https://integrate.api.nvidia.com/v1"
    api_key_env: str = "NVIDIA_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout_s: float = 180.0
    max_retries: int = 3
    # NVIDIA NIM docs: Gemma 4 reasoning is OFF unless chat_template_kwargs.enable_thinking is true.
    thinking: bool = False
    # Set False if the endpoint rejects the "system" role (v1 used a HumanMessage prefix instead).
    use_system_role: bool = True


@dataclass(frozen=True)
class RunConfig:
    recursion_limit: int = 40   # hard cap on graph steps (v1 had none)
    workers: int = 1            # keep 1 for latency studies; >1 only for accuracy-only sweeps
    reps: int = 1
    seed: int = 0
    sc_samples: int = 3         # self-consistency votes
    sc_temperature: float = 0.7


def config_fingerprint(*cfgs) -> str:
    blob = json.dumps([asdict(c) for c in cfgs], sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]
