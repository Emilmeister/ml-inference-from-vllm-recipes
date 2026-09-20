import json
from pathlib import Path

import pytest

from vllm_cloudru.translator import UnsupportedRecipe, translate_rendering

# AICODE-NOTE: Fixtures are minimized from recipes.vllm.ai on 2026-09-20.
# Keep both argv and head_argv/worker_argv shapes when refreshing them.
FIXTURES = Path(__file__).parents[1] / "fixtures" / "recipes"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def test_translates_single_node_rendering():
    translated = translate_rendering(load_fixture("single-node.json"))

    assert translated.model_id == "LiquidAI/LFM2.5-8B-A1B"
    assert translated.arguments == [
        {
            "key": "reasoning-parser",
            "value": "qwen3",
            "parameterType": "PARAMETER_TYPE_ARG_KV",
        },
        {
            "key": "enable-auto-tool-choice",
            "value": True,
            "parameterType": "PARAMETER_TYPE_ARG_FLAG",
        },
        {
            "key": "tool-call-parser",
            "value": "lfm2",
            "parameterType": "PARAMETER_TYPE_ARG_KV",
        },
    ]
    assert translated.environment == []
    assert translated.recommended_gpu_count == 8
    assert translated.recommended_hardware == "H100"


def test_rejects_multi_node_rendering():
    with pytest.raises(UnsupportedRecipe, match="нескольких процессов"):
        translate_rendering(load_fixture("multi-node.json"))
