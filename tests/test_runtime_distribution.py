from importlib.metadata import PackageNotFoundError

import torch

from wt_genesis import tools


def test_windows_rayd_distribution_is_not_reported_missing(monkeypatch):
    versions = {'witwin-radar': '0.3.0', 'witwin-channel': '0.5.0', 'rayd-torch': '0.8.0'}
    def version(name):
        if name not in versions:
            raise PackageNotFoundError(name)
        return versions[name]
    monkeypatch.setattr(tools.metadata, 'version', version)
    monkeypatch.setattr(tools.importlib, 'import_module', lambda _: object())
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    result = tools._inspect_radar_pipeline_runtime()
    assert result['ready']
    assert result['versions']['rayd'] == '0.8.0'
    assert result['distributions']['rayd'] == 'rayd-torch'

    def broken_import(name):
        if name == 'rayd.torch':
            raise OSError('native DLL missing')
        return object()
    monkeypatch.setattr(tools.importlib, 'import_module', broken_import)
    result = tools._inspect_radar_pipeline_runtime()
    assert not result['ready']
    assert result['import_errors']['rayd.torch'] == 'OSError: native DLL missing'
