---
name: text-to-smpl-motion
description: Generate SMPL humanoid motion animations from text prompts using wt_genesis and bind the generated motion to wt_humanoid. Use when the user asks for text-to-motion, text-generated human animation, SMPL animation, humanoid movement generation, or MDM-based motion.
version: 1
scope: plugin
---

Use `witwin.genesis.generate_humanoid_animation` when the user asks to create a humanoid/SMPL motion in the scene, add it to the timeline, or play it back.
Use `witwin.genesis.generate_smpl_motion` only when the user asks for a standalone motion asset file.

Workflow:

1. Inspect `witwin.genesis.check_genesis_runtime` before generation. Missing licensed models are blockers; do not invent model paths or create placeholder files. Runtime readiness is not an end-to-end measurement result.
2. For scene animation requests, call `witwin.genesis.generate_humanoid_animation` with the user's prompt and the exact already-open `scene_id`. Unknown/closed scenes must not silently fall back to another room.
3. Pass `autoplay: true` only if the user asks to play immediately. The result identifies the new body/controller, source asset and actual motion interval. Preserve existing cats and their timelines.
4. For the current mesh-based Radar animation tools, explicitly select `playback_mode: "baked_transforms"`. This writes validated root/bone Transform tracks without a competing controller track. Source SMPL parameter components remain initial-pose metadata; do not describe this as dynamic parametric SMPL export.
5. Use the installed `witwin.radar.ensure_sensor`, `plan_animation_measurement`, `submit_animation_measurement`, `get_simulation`, `verify_result`, `prepare_replay` and `export_result` tools according to their schemas. Carry forward the returned body/operation IDs. If a required tool is unavailable, report that blocker instead of guessing a replacement. Existing multiple-animated-target restrictions still apply; do not silently delete another actor's animation.
6. WT Radar 0.4 consumes the returned `SkinnedMesh` target and baked Transform tracks directly. Pass the returned target object ID to the Radar tools; site selection and `target_site_budget` belong to the Radar settings object. Do not invent or attach a target-side `RadarScattererComponent`.
7. Report generation, replay preparation, GPU measurement validation and export as separate results. Do not claim collision-free or physically calibrated human scattering from a successful import or interface test.
8. If the user only wants a reusable file, call `witwin.genesis.generate_smpl_motion`; use its returned `motion_path` with `pose_key='smpl_pose'`, `betas_key='shape'`, and `trans_key='root'`.

Prefer visible Project assets under `assets/motions/` for generated motion files. Do not write generated motion assets into `.witwin/` unless the user explicitly wants disposable runtime output.
