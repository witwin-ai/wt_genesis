---
name: text-to-smpl-motion
description: Generate SMPL humanoid motion animations from text prompts using wt_genesis and bind the generated motion to wt_humanoid. Use when the user asks for text-to-motion, text-generated human animation, SMPL animation, humanoid movement generation, or MDM-based motion.
version: 1
scope: plugin
---

Use `witwin.genesis.generate_humanoid_animation` when the user asks to create a humanoid/SMPL motion in the scene, add it to the timeline, or play it back.
Use `witwin.genesis.generate_smpl_motion` only when the user asks for a standalone motion asset file.

Workflow:

1. For scene animation requests, call `witwin.genesis.generate_humanoid_animation` with the user's prompt.
2. Pass `autoplay: true` if the user asks to play the timeline immediately.
3. The tool creates an `SMPLBody`, an `SMPLMotion`, binds the generated `.npz` file, and keys `SMPLMotion.current_frame` plus the body object's `Transform.position` and `Transform.rotation` root motion on the timeline.
4. If the user only wants a reusable file, call `witwin.genesis.generate_smpl_motion`; use its returned `motion_path` with `pose_key='smpl_pose'`, `betas_key='shape'`, and `trans_key='root'`.
5. If the user asks for a baked Radar response, pass `radar_surface_sites` (default recommendation: 128) and optionally `radar_surface_seed` to `generate_humanoid_animation`. Ensure a configured `Radar` component is in the Scene, then use the Timeline bake producer `witwin.radar.simulate`.
6. After bake completion, open the dataset in Timeline Replay. The Radar `signal_source` binding follows the playhead automatically; use the range-Doppler channel for the standard heatmap unless the user asks for raw or point-cloud output.

Prefer visible Project assets under `assets/motions/` for generated motion files. Do not write generated motion assets into `.witwin/` unless the user explicitly wants disposable runtime output.
