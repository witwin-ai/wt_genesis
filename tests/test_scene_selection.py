from types import SimpleNamespace

import pytest

from wt_genesis import tools


def test_explicit_scene_wins_over_active_scene():
    first, second = SimpleNamespace(scene_id='first'), SimpleNamespace(scene_id='second')
    server = SimpleNamespace(get_scene={'first': first, 'second': second}.get)
    ctx = SimpleNamespace(api=SimpleNamespace(scene=first, server=server))
    assert tools._get_scene(ctx, 'second') is second
    assert tools._get_scene(ctx) is first
    for missing in ['closed', ' ']:
        with pytest.raises(tools.ToolError) as caught:
            tools._get_scene(ctx, missing)
        assert caught.value.code == 'missing_scene'


def test_closed_scene_rejects_before_generation(monkeypatch):
    ctx = SimpleNamespace(scene=SimpleNamespace(scene_id='other'))
    def unexpected(*args, **kwargs):
        pytest.fail('must not generate after an invalid target')
    monkeypatch.setattr(tools, 'create_humanoid_animation', unexpected)
    with pytest.raises(tools.ToolError):
        tools._run_generate_animation(ctx, {'scene_id': 'closed', 'prompt': 'walk'}, None)
