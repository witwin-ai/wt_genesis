"""The generate_humanoid_route_animation agent tool.

The audit's agents could only pass ``generate_humanoid_animation`` a prompt
and a position: "walk to the sofa and sit" never worked. This tool takes the
planned route and the end pose; it refuses on MDM instead of writing the
route into the prompt.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from tests.test_kimodo_path import ROUTE, _route_following_generator, _scene_with_tiny_skin
from tests.test_route_animation import SOFA, _box, _people, _seat_skin, _seeded
from tests.test_tools import _FakeContext


def _context(scene):
    ctx = _FakeContext()
    ctx.api = SimpleNamespace(scene=scene, server=None)
    return ctx


def _tool(scene, *, calls=None, generator=None, geometry=(SOFA,)):
    from wt_genesis.tools import make_generate_humanoid_route_animation_tool

    return make_generate_humanoid_route_animation_tool(
        _context(scene), generator=generator or _route_following_generator(
            [] if calls is None else calls),
        geometry_reader=lambda _scene: list(geometry), skin=_seat_skin(), foot_lock=False,
        body_validator=lambda *_args: {"ok": True, "status": "clear_at_samples", "samples": 1,
                                       "affected_samples": 0, "contacts": []})


def _call(tool, args):
    return asyncio.run(tool.call(args))


def test_on_mdm_a_route_is_refused_not_written_into_a_prompt(tmp_path, monkeypatch):
    from witwin_server.tools.base import ToolError

    monkeypatch.setenv("WITWIN_PROJECT_DIR", str(tmp_path))
    monkeypatch.setenv("WITWIN_GENESIS_BACKEND", "mdm_tiny")
    scene = _scene_with_tiny_skin(monkeypatch)
    calls = []
    with pytest.raises(ToolError) as refused:
        _call(_tool(scene, calls=calls), {"waypoints": [list(p) for p in ROUTE], "end": "sit",
                                          "target_object_id": "sofa1"})

    assert refused.value.code == "kimodo_backend_required"
    assert "Kimodo" in str(refused.value)
    assert refused.value.detail["active_backend"] == "mdm_tiny"
    assert calls == [] and _people(scene) == []


def test_walk_to_the_sofa_and_sit(tmp_path, monkeypatch):
    monkeypatch.setenv("WITWIN_PROJECT_DIR", str(tmp_path))
    monkeypatch.setenv("WITWIN_GENESIS_BACKEND", "kimodo")
    scene = _scene_with_tiny_skin(monkeypatch)
    calls = []
    result = _call(_tool(scene, calls=calls), {
        "waypoints": [list(p) for p in ROUTE], "end": "sit", "target_object_id": "sofa1",
        "speed_mps": 1.0, "seed": 4, "rest_to_clip_end": False})

    assert result["end"] == "sit"
    assert _people(scene) == [result["body_object_id"]]
    assert result["target"]["object_id"] == "sofa1"
    assert result["target"]["sit_at"] == pytest.approx([3.0, 4.16])
    assert "seat_contact" in result["key_events"]
    assert result["attempts"] == [{"seed": 4, "status": "kept"}]
    assert result["scene_binding"]["radar_target_contract"]["required_playback_mode"] \
        == "baked_transforms"
    assert result["binding"]["pivot"] == "smpl_pelvis_j0"
    assert any(item["type"] == "end-effector" for item in calls[0]["constraints"])


def test_an_end_facing_direction_becomes_a_kimodo_heading(tmp_path, monkeypatch):
    import math

    monkeypatch.setenv("WITWIN_PROJECT_DIR", str(tmp_path))
    monkeypatch.setenv("WITWIN_GENESIS_BACKEND", "kimodo")
    scene = _scene_with_tiny_skin(monkeypatch)
    result = _call(_tool(scene), {"waypoints": [list(p) for p in ROUTE],
                                  "end_facing": [1.0, 0.0], "rest_to_clip_end": False})

    assert result["plan"]["end_heading"] == pytest.approx(math.pi / 2)


@pytest.mark.parametrize("args, geometry, code", [
    ({"end": "sit", "target_object_id": "armchair9"}, (SOFA,), "route_target_unsupported"),
    ({"end": "lie", "target_object_id": "sofa1"}, (SOFA,), "route_target_unsupported"),
    ({"end": "fall", "waypoints": [[0.0, 0.0], [2.0, 0.0]]},
     (_box("table1", (3.0, 0.0, -0.5), (3.6, 0.75, 0.5)),), "fall_zone_blocked"),
    ({"speed_mps": 2.5, "waypoints": [[0.0, 0.0], [30.0, 0.0]]}, (), "invalid_route"),
])
def test_refusals_carry_codes_the_scenario_layer_translates(tmp_path, monkeypatch, args,
                                                            geometry, code):
    from witwin_server.tools.base import ToolError

    monkeypatch.setenv("WITWIN_PROJECT_DIR", str(tmp_path))
    monkeypatch.setenv("WITWIN_GENESIS_BACKEND", "kimodo")
    scene = _scene_with_tiny_skin(monkeypatch)
    with pytest.raises(ToolError) as refused:
        _call(_tool(scene, geometry=geometry), {"waypoints": [list(p) for p in ROUTE], **args})

    assert refused.value.code == code
    if code == "fall_zone_blocked":
        assert refused.value.detail["blockers"][0]["object_id"] == "table1"


def test_every_refused_take_is_reported(tmp_path, monkeypatch):
    from witwin_server.tools.base import ToolError

    monkeypatch.setenv("WITWIN_PROJECT_DIR", str(tmp_path))
    monkeypatch.setenv("WITWIN_GENESIS_BACKEND", "kimodo")
    scene = _scene_with_tiny_skin(monkeypatch)
    with pytest.raises(ToolError) as refused:
        _call(_tool(scene, generator=_seeded(set())),
              {"waypoints": [list(p) for p in ROUTE], "attempts": 2, "rest_to_clip_end": False})

    assert refused.value.code == "humanoid_generation_rejected"
    assert [row["status"] for row in refused.value.detail["attempts"]] == ["rejected"] * 2
    assert _people(scene) == []


def test_the_schema_takes_a_route_and_an_end_pose_not_a_position():
    from wt_genesis.tools import make_generate_humanoid_route_animation_tool

    tool = make_generate_humanoid_route_animation_tool(_FakeContext())
    schema = tool.input_schema

    assert schema["required"] == ["waypoints"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["end"]["enum"] == ["stand", "sit", "lie", "fall"]
    assert "position" not in schema["properties"]
    assert all(prop.get("description") for prop in schema["properties"].values())
    assert tool.requires_confirmation is True


def test_the_route_body_check_does_not_relocate_the_walk(monkeypatch):
    # Moving or turning the whole track would take the person off its route.
    import wt_genesis.tools as genesis_tools

    captured = {}

    def report(scene, args, project_root):
        captured.update(args=args)
        return {"status": "clear_at_samples"}

    module = SimpleNamespace(body_collision_report=report)
    monkeypatch.setattr(genesis_tools.importlib, "import_module", lambda _name: module)
    monkeypatch.setattr(genesis_tools, "default_project_dir", lambda: "project-root")

    result = genesis_tools._motion_matching_route_body_validation(object(), "human", 0.5, 4.0)

    assert result["status"] == "clear_at_samples"
    assert captured["args"]["object_id"] == "human"
    assert (captured["args"]["start"], captured["args"]["end"]) == (0.5, 4.0)
