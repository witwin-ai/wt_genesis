"""Generated motion assets get a bounded file name, whatever the prompt length.

R47 acceptance: a ~230-character natural-language motion prompt became the
.npz file stem and the Project path exceeded the Windows 260-character limit,
so the person was never created.
"""
import pytest

from wt_genesis.adapter import scene_animation as adapter
from test_scene_animation import _animated_backend_for_frames
from test_scene_authoring_safety import _existing_scene, synthetic_skin  # noqa: F401

LONG_PROMPT = (
    "A person on a flat level bedroom floor takes a few small slow even steps forward, "
    "with an upright relaxed posture, comes to a natural stop, then stands in place and "
    "waves one hand. No stairs or furniture contact. Complete sequence about nine seconds."
)


def _create(scene, project_dir, *, prompt, **kwargs):
    return adapter.create_humanoid_animation(
        scene=scene, project_dir=project_dir, prompt=prompt,
        motion_length=.2, device=-1, backend=_animated_backend_for_frames(4), **kwargs)


@pytest.fixture
def names(monkeypatch):
    seen = []
    real = adapter.generate_motion_asset

    def capture(*args, **kwargs):
        seen.append(kwargs.get("output_name"))
        return real(*args, **kwargs)

    monkeypatch.setattr(adapter, "generate_motion_asset", capture)
    return seen


def test_long_prompt_gets_a_short_deterministic_asset_name(tmp_path, names):
    scene, _ = _existing_scene()
    _create(scene, tmp_path, prompt=LONG_PROMPT)
    _create(_existing_scene()[0], tmp_path / "again", prompt=LONG_PROMPT)

    first, second = names
    assert first == second
    assert len(first) <= 64
    assert first.startswith("a_person_on_a_flat_level_bedroom_floor")
    files = list((tmp_path / "assets" / "motions").glob("*.npz"))
    assert [path.stem for path in files] == [first]


def test_distinct_long_prompts_with_a_shared_prefix_do_not_collide(tmp_path, names):
    _create(_existing_scene()[0], tmp_path, prompt=LONG_PROMPT)
    _create(_existing_scene()[0], tmp_path, prompt=LONG_PROMPT.replace("one hand", "both hands"))
    assert names[0] != names[1]


def test_explicit_output_name_is_kept(tmp_path, names):
    _create(_existing_scene()[0], tmp_path, prompt=LONG_PROMPT, output_name="wave_walk")
    assert names == ["wave_walk"]
