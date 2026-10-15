# Genesis Plugin

`wt_genesis` generates an SMPL animation, authors it into the current Scene,
and keys the Timeline. WT Radar 0.4 consumes the existing hydrated
`SkinnedMesh` and baked Transform tracks; Genesis does not add a second
Radar-specific target component.

## Optional Kimodo backend

Genesis keeps the same Agent tools and canonical SMPL asset contract when the
generator is changed. To opt into the isolated NVIDIA Kimodo backend before
starting Studio, set:

```powershell
$env:WITWIN_GENESIS_BACKEND = "kimodo"
$env:WITWIN_KIMODO_PYTHON = "C:\Users\WQZ\AppData\Local\anaconda3\envs\witwin-kimodo\python.exe"
$env:WITWIN_KIMODO_ROOT = "C:\WiTwin\repos\kimodo"
$env:TEXT_ENCODER_DEVICE = "cpu"
```

The default Kimodo checkpoint is `Kimodo-SOMA-RP-v1.1`. Its 77-joint SOMA
output is retargeted at the integration boundary to the existing 24-joint SMPL
timeline; WT Genesis itself is not modified. Set `WITWIN_KIMODO_MODEL` only for
an intentional checkpoint comparison. The original `mdm_tiny` backend remains
the default unless `WITWIN_GENESIS_BACKEND=kimodo` is present.

`check_genesis_runtime` reports `active_backend` and a separate
`kimodo_runtime` diagnosis. Kimodo requires local Hugging Face authentication
with access to its gated Llama text encoder. Never put an HF token in a Project,
scene, prompt, source file, or Agent message; authenticate in the isolated
environment with `hf auth login`.

The SOMA-to-SMPL path preserves the major body chains and resamples rotations
with spherical interpolation. It is an integration retarget, not a claim of
motion-quality equivalence; quality must be judged from a generated Studio
timeline and compared against the same prompt/seed on the original backend.

## Preflight and scene animation

Run `witwin.genesis.check_genesis_runtime` first. Its `pipeline_ready` field is
a package/model/native-library/CUDA prerequisite check, **not** evidence of a
successful human Radar measurement. `human_radar_validation: not_run` makes
that boundary explicit. Supply licensed SMPL assets yourself; do not substitute
dummy files to pass the check. On Windows, an explicit `WITWIN_GENESIS_DATA_DIR`
set before server startup avoids per-application user-cache redirection.

Use `scene_id` to select an exact already-open room. A closed or unknown room
fails without falling back to another room. Scene authoring validates motion
and builds the skin before inserting objects, then rolls back newly authored
objects/tracks if attachment or keyframe recording fails. It preserves existing
objects, timeline length, frame rate and playhead. Generated motion assets are
retained for diagnosis rather than deleted after an authoring failure.

## Mesh-based Radar animation interface

For the current `witwin.radar.submit_animation_measurement` tool path:

1. Enable Genesis, Humanoid and Radar and inspect runtime readiness.
2. Generate the human with an explicit `scene_id` and
   `playback_mode: "baked_transforms"`. Retain the returned
   `scene_binding.radar_target_contract` and target object ID.
3. Retain the returned body object ID and motion interval. Use the installed
   Radar tools to ensure a sensor and plan the measurement against that body;
   inspect the plan before submitting. Do not invent object IDs or tool names.
4. Submit, wait, verify the result, then prepare replay/export using those
   tools' returned IDs. A generation receipt is not a measurement receipt.

The opt-in mode validates complete root and bone tracks, omits only the newly
generated `SMPLMotion.current_frame` track, and verifies the actual stored keys.
This prevents an integer-frame controller from overwriting interpolated bone
transforms. The default `legacy_controller` preserves the previous behavior.
Source NPZ and controller component are retained in both modes.

Limitations: baked mode is authoritative for the rendered skinned mesh, not
time-varying `SMPLBody.pose`/`PlatformGeometry.pose` parameters. Parametric SMPL
animation export is not supported in that mode. The current Radar animation
adapter still enforces its normal single-animated-target checks; independently
animated cat and human together are not automatically supported. Interface
tests are not a validation of human motion quality, collision avoidance or
calibrated human electromagnetic scattering. A real licensed-model human
generation and GPU measurement must be tested separately.

WT Radar 0.4 selects stable sites and owns the target site budget on its Radar
settings object. A target-side `RadarScattererComponent`, `radar_surface_sites`,
or `radar_surface_seed` is not part of this interface.

The Genesis runtime tool reports asset/package readiness separately from the
Radar/Channel/RayD/CUDA runtime. It recognizes the Windows `rayd-torch`
distribution, while still checking the real `rayd.torch` native import.

## Local plugin checkout

Studio never scans sibling Plugin repositories implicitly. Add this checkout
through the Plugins panel or start the ProjectServer with an explicit path:

```powershell
python -m witwin_server.server_entry serve --project E:\Projects\Demo --plugin-path E:\Code\witwin-plugins\wt_genesis --plugin-path E:\Code\witwin-plugins\wt_radar
```
