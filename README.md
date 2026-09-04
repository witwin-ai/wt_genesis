# Genesis Plugin

`wt_genesis` generates an SMPL animation, authors it into the current Scene,
and keys the Timeline. The optional Radar integration adds stable surface
scatter sites to the generated skinned body.

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
   `playback_mode: "baked_transforms"`. Omit `radar_surface_sites` unless the
   installed Radar Plugin actually exposes `RadarScattererComponent`.
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

## Optional surface-scatter producer interface

The sequence below requires an installed Radar version that supplies
`RadarScattererComponent` and the corresponding Timeline producer. It is not
interchangeable with the mesh-based animation tool path above. If that optional
component is absent, requesting `radar_surface_sites` fails before generation.

1. Enable the Genesis, Humanoid, and Radar Plugins for the Project.
2. Call `witwin.genesis.generate_humanoid_animation` with
   `radar_surface_sites` (128 is a practical starting point) and an optional
   `radar_surface_seed`.
3. Add/configure a `Radar` component in the Scene if one is not already
   present.
4. Use **Bake over timeline**. The `witwin.radar.simulate` producer samples the
   complete Timeline, derives stable surface velocities, and emits the selected
   channel (range-Doppler by default).
5. Open the registered dataset in Timeline Replay. The generated Radar
   component's `signal_source` is attached automatically, so the existing
   heatmap follows the replay playhead.

The Genesis runtime tool reports asset/package readiness separately from the
Radar/Channel/RayD/CUDA runtime. It recognizes the Windows `rayd-torch`
distribution, while still checking the real `rayd.torch` native import.

## Local plugin checkout

Studio never scans sibling Plugin repositories implicitly. Add this checkout
through the Plugins panel or start the ProjectServer with an explicit path:

```powershell
python -m witwin_server.server_entry serve --project E:\Projects\Demo --plugin-path E:\Code\witwin-plugins\wt_genesis --plugin-path E:\Code\witwin-plugins\wt_radar
```
