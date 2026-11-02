from types import SimpleNamespace

import pytest

from wt_genesis import tools


def test_the_current_scene_is_the_target():
    current = SimpleNamespace(scene_id='first')
    ctx = SimpleNamespace(api=SimpleNamespace(scene=current, server=SimpleNamespace(scene=current)))
    assert tools._get_scene(ctx) is current
    assert tools._get_scene(ctx, 'first') is current


def test_another_scene_is_refused_not_redirected():
    # The ProjectServer loads one scene only; a request for another room must not
    # land in the current one.
    ctx = SimpleNamespace(api=SimpleNamespace(scene=SimpleNamespace(scene_id='first')))
    with pytest.raises(tools.ToolError) as caught:
        tools._get_scene(ctx, 'second')
    assert caught.value.code == 'scene_not_current'
    assert 'switch to scene' in str(caught.value)
    with pytest.raises(tools.ToolError) as caught:
        tools._get_scene(ctx, ' ')
    assert caught.value.code == 'missing_scene'


def test_a_non_current_scene_rejects_before_generation(monkeypatch):
    ctx = SimpleNamespace(scene=SimpleNamespace(scene_id='other'))
    def unexpected(*args, **kwargs):
        pytest.fail('must not generate after an invalid target')
    monkeypatch.setattr(tools, 'create_humanoid_animation', unexpected)
    with pytest.raises(tools.ToolError):
        tools._run_generate_animation(ctx, {'scene_id': 'closed', 'prompt': 'walk'}, None)
