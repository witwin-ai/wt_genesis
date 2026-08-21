# Genesis Plugin

`wt_genesis` generates an SMPL animation, authors it into the current Scene,
and keys the Timeline. The optional Radar integration adds stable surface
scatter sites to the generated skinned body.

## Genesis to Radar sequence

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

The Genesis runtime tool `witwin.genesis.check_genesis_runtime` reports Genesis
asset/package readiness separately from the Radar/Channel/RayD/CUDA pipeline.

## Local plugin checkout

Studio never scans sibling Plugin repositories implicitly. Add this checkout
through the Plugins panel or start the ProjectServer with an explicit path:

```powershell
python -m witwin_server.server_entry serve --project E:\Projects\Demo --plugin-path E:\Code\witwin-plugins\wt_genesis --plugin-path E:\Code\witwin-plugins\wt_radar
```
