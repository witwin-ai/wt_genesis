"""Genesis Tool tags drive the ProjectServer Tool selection.

Since S7c the Genesis retrieval vocabulary lives only in the Tool tags in
``tools.py``; the core ``select_tools`` turns them into packs and hints. The
prompts come from the deleted Studio hints and tests (``toolRetrieval.ts``,
``radioMapRouting.test.ts``).
"""
from __future__ import annotations

import pytest

from witwin_server.api.testing import select_tools

PREFIX = "witwin.genesis."
WRITERS = ("generate_smpl_motion", "generate_humanoid_animation", "generate_humanoid_route_animation")
HINT_REASONS = ("hint:", "explicit-hint:")


def _core(name, description, tier="read"):
    # Stand-ins for first-party Tools that compete for the same slots.
    return {
        "name": name, "description": description, "permissionTier": tier,
        "sideEffects": tier != "read", "requiresConfirmation": tier == "destructive",
        "inputSchema": {"type": "object", "properties": {"scene_id": {"type": "string"}}},
    }


@pytest.fixture(scope="module")
def inventory():
    from wt_genesis import tools as genesis_tools

    ctx = object()
    plugin = [
        genesis_tools.make_check_genesis_runtime_tool(ctx),
        genesis_tools.make_generate_smpl_motion_tool(ctx),
        genesis_tools.make_generate_humanoid_animation_tool(ctx),
        genesis_tools.make_generate_humanoid_route_animation_tool(ctx),
    ]
    # The loader prefixes plugin Tools with the plugin id.
    return [{**tool.to_schema(), "name": PREFIX + tool.name} for tool in plugin] + [
        _core("scene.list_objects", "List objects in a Scene with exact ids."),
        _core("scene.snapshot", "Read a Scene snapshot."),
        _core("scene.create_object", "Create a primitive Scene object.", "scene_write"),
        _core("scene.update_transform", "Move, rotate and/or scale an existing object.", "scene_write"),
        _core("scene.delete_object", "Delete an object.", "destructive"),
        _core("result.read", "Read a stored Result."),
        _core("project.fs_read", "Read a project file."),
    ]


def _select(inventory, prompt):
    selection = select_tools(inventory, prompt)
    return {entry.name: entry.reasons for entry in selection.trace.selected}


@pytest.mark.parametrize("prompt", [
    "Add a person who waves and stretches.",
    "Have someone walk across the room",
    "I want a person standing next to the window",
    "生成一个人，在房间里走一圈",
    "让一个人坐下",
])
def test_person_requests_open_the_humanoid_writer(inventory, prompt):
    selected = _select(inventory, prompt)
    assert PREFIX + "generate_humanoid_animation" in selected
    assert any(r.startswith("explicit-hint:") for r in selected[PREFIX + "generate_humanoid_animation"])


def test_a_route_request_opens_the_route_writer(inventory):
    selected = _select(inventory, "The person walks to the sofa and sits on it")
    assert PREFIX + "generate_humanoid_route_animation" in selected


@pytest.mark.parametrize("prompt", ["Generate SMPL motion of a jump", "Use Genesis for text to smpl motion"])
def test_smpl_requests_open_the_motion_exporter(inventory, prompt):
    assert PREFIX + "generate_smpl_motion" in _select(inventory, prompt)


def test_person_words_do_not_join_the_scene_pack(inventory):
    # Genesis sits in the core Scene pack; its person words must not make every
    # scene.* Tool relevant to a person request.
    selected = _select(inventory, "Add a person who waves")
    for name, reasons in selected.items():
        if name.startswith("scene."):
            assert not any(r.startswith("pack:witwin.scene:") for r in reasons), (name, reasons)


def test_scene_requests_still_reach_genesis_through_the_scene_pack(inventory):
    selected = _select(inventory, "What is in the scene?")
    assert PREFIX + "check_genesis_runtime" in selected
    for name in WRITERS:
        assert PREFIX + name not in selected, name


@pytest.mark.parametrize("prompt", [
    # A read-only mention of a person names no action: the writers stay hidden.
    "What does the person in the scene look like?",
    "How many humans are there?",
])
def test_read_only_person_mentions_keep_writers_hidden(inventory, prompt):
    selected = _select(inventory, prompt)
    for name in WRITERS:
        assert PREFIX + name not in selected, (prompt, name)


@pytest.mark.parametrize("prompt", [
    "Move the cube up by one meter",
    "Run a radio map of this scene",
    "Show me the UCSD map",
    "Make the cat jump onto the bed",
    "Cancel radar simulation",
])
def test_unrelated_requests_do_not_match_genesis_vocabulary(inventory, prompt):
    selected = _select(inventory, prompt)
    for name in WRITERS:
        assert PREFIX + name not in selected, (prompt, name)
    for name, reasons in selected.items():
        if name.startswith(PREFIX):
            assert not any(r.startswith(HINT_REASONS) for r in reasons), (prompt, name, reasons)
