"""Shared pipeline config: prompt templates and versioned constants."""

import pytest

from gigacity_contracts import CONFIG_VERSION, PromptVersion, Task, prompt_template
from gigacity_contracts.settings import PROMPTS


def test_every_task_has_a_v1_prompt():
    for task in Task:
        assert (task, PromptVersion.V1) in PROMPTS


def test_v1_prompt_matches_ab_baseline():
    template = prompt_template(Task.COLLISION_YN, PromptVersion.V1)
    assert template.frame_caption(1, 1200) == "Frame 1 at 1.2 s:"
    assert template.question(None) == (
        "Did any traffic collision occur during the shown observation window?"
        " Answer only Yes or No."
    )
    assert template.question("Did the truck flip?") == (
        "Did the truck flip? Answer only Yes or No."
    )


def test_prompt_registry_is_read_only():
    with pytest.raises(TypeError):
        PROMPTS[(Task.COLLISION_YN, PromptVersion.V1)] = None


def test_config_version_is_set():
    assert CONFIG_VERSION.strip()
