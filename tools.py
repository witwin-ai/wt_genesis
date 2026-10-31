"""Agent tools for wt_genesis."""
from __future__ import annotations

import asyncio
import importlib
from importlib import metadata
import os
import threading
from typing import Any

from packaging.specifiers import SpecifierSet

from witwin_server.plugins.paths import default_project_dir
from witwin_server.tools.base import Tool, ToolError, tool
from witwin_server.utils.logging import get_logger

from witwin.genesis.assets import inspect_runtime
from witwin.genesis.motion_export import Backend, generate_motion_asset
from .adapter.kimodo_backend import configured_backend, configured_backend_name, inspect_kimodo_runtime
from .adapter.kimodo_path import heading_from_direction
from .adapter.route_animation import (
    END_POSES, MAX_MOTION_S, FallZoneBlocked, RouteBackendUnavailable, RouteGenerationRejected,
    RouteTargetUnsupported, create_route_animation, require_route_backend,
)
from .adapter.scene_animation import create_humanoid_animation
from .adapter.trajectory_clearance import HumanoidTrajectoryBlocked

logger = get_logger("Genesis")


_HUMANOID_ANIMATION_RETRIEVAL_TAGS = (
    "pack:witwin.scene",
    "word-intent:human",
    "word-intent:person",
    "word-intent:someone",
    "intent:人物",
    "intent:人类",
    "intent:一个人",
    "hint-term:human animation",
    "hint-term:person in the room",
    "hint-term:human in the room",
    "hint-explicit-intent:add one human",
    "hint-explicit-intent:add a human",
    "hint-explicit-intent:create a human",
    "hint-explicit-intent:put a human",
    "hint-explicit-intent:someone walking",
    "hint-explicit-intent:someone stroll",
    "hint-explicit-intent:person walking",
    "hint-explicit-intent:person stroll",
    "hint-explicit-intent:生成一个人",
    "hint-explicit-intent:添加一个人",
    "hint-explicit-intent:人物行走",
    "hint-requires-pack",
    "hint-boost:20",
)


_DESCRIPTION = (
    "Generate an SMPL humanoid motion asset from a text prompt using the configured Genesis model backend. "
    "Use this when the user asks for text-to-human-motion, text-to-SMPL, or generated humanoid animation. "
    "The tool writes a .npz file under the Project's assets/motions directory. The returned path can be "
    "loaded by wt_humanoid's SMPLMotion component with pose_key='smpl_pose', betas_key='shape', trans_key='root'."
)

_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "prompt": {
            "type": "string",
            "minLength": 1,
            "description": "Natural-language motion prompt, for example 'a person walks forward and waves'.",
        },
        "motion_length": {
            "type": "number",
            "minimum": 0.1,
            "maximum": 9.8,
            "description": "Motion duration in seconds. MDM HumanML output is capped to 196 frames at 20 fps.",
        },
        "seed": {"type": "integer", "description": "Random seed. Default 10."},
        "device": {
            "type": "integer",
            "description": "CUDA device index, or -1 for CPU. Default 0.",
        },
        "guidance_param": {
            "type": "number",
            "minimum": 0.1,
            "description": "Classifier-free guidance scale. Default 2.5.",
        },
        "smplify_iters": {
            "type": "integer",
            "minimum": 1,
            "maximum": 100,
            "description": "SMPLify iterations per sequence. Default 20.",
        },
        "output_name": {
            "type": "string",
            "description": "Optional filename stem for the generated .npz asset.",
        },
    },
    "required": ["prompt"],
    "additionalProperties": False,
}

_ANIMATION_DESCRIPTION = (
    "Generate an SMPL humanoid animation from a text prompt, either create a new person or reuse an "
    "existing Genesis SMPL body by body_object_id, and place or continue it in the current Studio scene. "
    "bind the generated motion to wt_humanoid SMPLBody/SMPLMotion components, and key the timeline "
    "so timeline playback drives the humanoid pose plus root Transform position/rotation. Use this "
    "for user requests like 'generate a human animation', 'make a person walk', or 'create an SMPL "
    "motion in the scene'."
)

_ANIMATION_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        **_INPUT_SCHEMA["properties"],
        "scene_id": {
            "type": "string", "minLength": 1,
            "description": "Exact already-open target scene. A missing scene never falls back to another room.",
        },
        "playback_mode": {
            "type": "string", "enum": ["legacy_controller", "baked_transforms"],
            "default": "legacy_controller",
            "description": "Use baked_transforms for the current mesh-based Radar animation input: only root/bone Transform tracks drive playback. SMPL parameter components remain source metadata, not time-evaluated parametric solver input. Legacy retains the upstream controller track.",
        },
        "body_name": {
            "type": "string",
            "description": "Optional scene object name for the generated SMPL body.",
        },
        "body_object_id": {
            "type": "string",
            "minLength": 1,
            "description": "Existing Genesis SMPL body to animate in place. Omit to create a new humanoid.",
        },
        "motion_name": {
            "type": "string",
            "description": "Optional scene object name for the generated SMPL motion controller.",
        },
        "position": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 3,
            "maxItems": 3,
            "description": "Scene position for the generated humanoid, as [x, y, z]. Default [0, 0, 0].",
        },
        "start_time": {
            "type": "number",
            "minimum": 0,
            "description": "Timeline time in seconds when this humanoid animation begins. Default 0.",
        },
        "autoplay": {
            "type": "boolean",
            "description": "Whether to start timeline playback after creating the animation. Default false.",
        },
        "loop": {
            "type": "boolean",
            "description": "Whether the SMPL motion and timeline should loop. Default true.",
        },
    },
    "required": ["prompt"],
    "additionalProperties": False,
}

_ROUTE_RETRIEVAL_TAGS = (
    "pack:witwin.scene",
    "word-intent:person",
    "hint-term:walk to",
    "hint-term:sit on",
    "hint-term:lie on the bed",
    "hint-term:falls down",
    "hint-explicit-intent:person walks to the sofa",
    "hint-explicit-intent:person sits on the bed",
    "hint-explicit-intent:走到沙发坐下",
    "hint-requires-pack",
)

_ROUTE_DESCRIPTION = (
    "Generate a person (Kimodo) who walks a planned floor route and ends in a pose: stand, sit on a "
    "seat, lie on a bed, or fall to the floor. "
    "When: the person must go somewhere in the room or end on furniture, e.g. 'walk to the sofa "
    "and sit'. Give the route as floor waypoints [x, z] (or [x, y, z]) from start to the stop in "
    "front of the target; the route is followed with root constraints, not described in text. "
    "Preconditions: the Kimodo backend (on MDM the tool refuses with kimodo_backend_required); "
    "sit and lie need target_object_id, a seat or bed in the scene, and the route ending on the "
    "floor in front of it; lie needs head_direction; fall needs 1.8 m of clear floor ahead of the "
    "last step. "
    "Result: the body, its interval, route following error, key event times (arrival, "
    "seat/back contact, impact), the end pose's rest on its support, the foot clean-up, the rest "
    "idle to the clip end and every Kimodo take. "
    "Limits: one adult SMPL body, at most 10 s of motion before the rest idle, no stairs, no "
    "carried objects, no hand contact; prompts and style_hint change style, never the path."
)

_POINT = {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 3}

_ROUTE_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "scene_id": {
            "type": "string", "minLength": 1,
            "description": "Exact already-open target scene. A missing scene never falls back to another room.",
        },
        "waypoints": {
            "type": "array", "items": _POINT, "minItems": 2,
            "description": "Floor route in scene metres, start first: [x, z] or [x, y, z] points (y is "
                           "the floor height). The last point is where the walk stops (in front of "
                           "the seat or bed for sit and lie).",
        },
        "end": {
            "type": "string", "enum": list(END_POSES), "default": "stand",
            "description": "How the route ends: stand, sit (on target_object_id), lie (on the bed "
                           "target_object_id) or fall (to the floor ahead of the last step).",
        },
        "target_object_id": {
            "type": "string", "minLength": 1,
            "description": "Scene object the person sits or lies on (required for sit and lie).",
        },
        "sit_at": {
            **_POINT, "maxItems": 2,
            "description": "Optional sit point [x, z] on the seat; default 0.16 m in from the seat "
                           "edge nearest the last waypoint.",
        },
        "head_direction": {
            **_POINT, "maxItems": 2,
            "description": "For lie: direction [x, z] from the bed's centre to its head end.",
        },
        "support_height_m": {
            "type": "number", "minimum": 0.0, "maximum": 1.5,
            "description": "Optional seat or mattress top above the floor in metres; default "
                           "measured on the target's mesh.",
        },
        "end_facing": {
            **_POINT, "maxItems": 2,
            "description": "Optional direction [x, z] a standing person faces at the end (a sit "
                           "faces out of the seat by default).",
        },
        "prompt": {
            "type": "string", "minLength": 1,
            "description": "Optional one-sentence motion text; default chosen for the end pose.",
        },
        "style_hint": {
            "type": "string", "minLength": 1,
            "description": "Optional style appended to the prompt, e.g. 'casually, hands relaxed'.",
        },
        "speed_mps": {
            "type": "number", "exclusiveMinimum": 0.0, "maximum": 3.0, "default": 1.0,
            "description": "Walking speed in m/s. Default 1.0.",
        },
        "start_hold_s": {
            "type": "number", "minimum": 0.0, "maximum": 5.0, "default": 0.5,
            "description": "Seconds standing at the start before walking. Default 0.5.",
        },
        "end_hold_s": {
            "type": "number", "minimum": 0.0, "maximum": 6.0,
            "description": "Seconds of motion after arriving (to turn and sit, lie down, fall); "
                           "default 1.5 stand, 3 sit, 5 lie, 3 fall.",
        },
        "start_time": {
            "type": "number", "minimum": 0.0,
            "description": "Timeline time in seconds when the person starts. Default 0.",
        },
        "rest_to_clip_end": {
            "type": "boolean", "default": True,
            "description": "Keep the person resting alive (breathing, looking round; pelvis and legs "
                           "still) from the motion's end to the scene's clip end. Default true.",
        },
        "rest_until_s": {
            "type": "number", "minimum": 0.0,
            "description": "Optional timeline time to rest until, instead of the clip end.",
        },
        "max_motion_s": {
            "type": "number", "exclusiveMinimum": 0.0, "maximum": MAX_MOTION_S,
            "default": MAX_MOTION_S,
            "description": "Longest generated motion in seconds (walk plus end hold). Default 10.",
        },
        "seed": {"type": "integer", "description": "Kimodo seed of the first take. Default 10."},
        "attempts": {
            "type": "integer", "minimum": 1, "maximum": 4, "default": 2,
            "description": "Kimodo takes tried (seed, seed + 1, ...) until one passes the take "
                           "checks. Default 2.",
        },
        "body_name": {"type": "string", "description": "Optional scene object name for the body."},
        "output_name": {"type": "string",
                        "description": "Optional filename stem for the generated .npz asset."},
    },
    "required": ["waypoints"],
    "additionalProperties": False,
}


def make_generate_humanoid_route_animation_tool(
    ctx: Any, *, generator: Any = None, geometry_reader: Any = None, skin: Any = None,
    foot_lock: bool = True, body_validator: Any = None,
) -> Tool:
    @tool(
        name="generate_humanoid_route_animation",
        description=_ROUTE_DESCRIPTION,
        input_schema=_ROUTE_INPUT_SCHEMA,
        output_schema={"type": "object"},
        side_effects=True,
        requires_confirmation=True,
        permission_scope="genesis.generate_scene_animation",
        permission_tier="execute",
        timeout=1800.0,
        tags=_ROUTE_RETRIEVAL_TAGS,
    )
    async def generate_humanoid_route_animation(args: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(
            _run_route_animation, ctx, args, generator=generator,
            geometry_reader=geometry_reader, skin=skin, foot_lock=foot_lock,
            body_validator=body_validator)

    return generate_humanoid_route_animation


def _run_route_animation(ctx: Any, args: dict[str, Any], *, generator: Any, geometry_reader: Any,
                         skin: Any, foot_lock: bool, body_validator: Any) -> dict[str, Any]:
    backend = _active_backend_name()
    try:
        require_route_backend(backend)
    except RouteBackendUnavailable as exc:
        raise ToolError(str(exc), code="kimodo_backend_required",
                        detail={"active_backend": backend,
                                "in_place_tool": "generate_humanoid_animation"}) from exc
    scene = _get_scene(ctx, args.get("scene_id"))
    if scene is None:
        raise ToolError("no active scene", code="no_scene")
    server = _get_server(ctx)
    rest_until: Any = "clip_end" if args.get("rest_to_clip_end", True) else None
    if "rest_until_s" in args:
        rest_until = float(args["rest_until_s"])
    end_heading = None
    if args.get("end_facing") is not None:
        try:
            end_heading = heading_from_direction(args["end_facing"])
        except ValueError as exc:
            raise ToolError(str(exc), code="invalid_route") from exc
    try:
        result = create_route_animation(
            scene=scene,
            project_dir=default_project_dir(),
            waypoints=args["waypoints"],
            end=str(args.get("end", "stand")),
            target_object_id=args.get("target_object_id"),
            sit_at=args.get("sit_at"),
            head_direction=args.get("head_direction"),
            support_height_m=args.get("support_height_m"),
            geometry_reader=geometry_reader or _motion_matching_geometry_reader(),
            prompt=args.get("prompt"),
            style_hint=args.get("style_hint"),
            speed_mps=float(args.get("speed_mps", 1.0)),
            start_hold_s=float(args.get("start_hold_s", 0.5)),
            end_hold_s=(float(args["end_hold_s"]) if "end_hold_s" in args else None),
            end_heading=end_heading,
            rest_until_s=rest_until,
            foot_lock=foot_lock,
            max_motion_s=float(args.get("max_motion_s", MAX_MOTION_S)),
            attempts=int(args.get("attempts", 2)),
            seed=int(args.get("seed", 10)),
            output_name=args.get("output_name"),
            body_name=args.get("body_name"),
            start_time=(float(args["start_time"]) if "start_time" in args else None),
            body_validator=body_validator or _motion_matching_route_body_validation,
            generator=generator,
            skin=skin,
            mesh_assets=getattr(server, "mesh_assets", None),
            broadcast=_get_broadcast_for_server(server),
            serialize_object=_get_serializer(server, scene),
        )
    except RouteTargetUnsupported as exc:
        raise ToolError(str(exc), code="route_target_unsupported") from exc
    except FallZoneBlocked as exc:
        raise ToolError(str(exc), code="fall_zone_blocked", detail={"blockers": exc.blockers}) from exc
    except HumanoidTrajectoryBlocked as exc:
        raise ToolError(str(exc), code="humanoid_trajectory_blocked",
                        detail={"blockers": exc.blockers,
                                "attempts": getattr(exc, "attempts", None)}) from exc
    except RouteGenerationRejected as exc:
        raise ToolError(str(exc), code="humanoid_generation_rejected",
                        detail={"attempts": exc.attempts}) from exc
    except ValueError as exc:
        raise ToolError(str(exc), code="invalid_route") from exc
    except RuntimeError as exc:
        if "Kimodo runtime is not ready" not in str(exc):
            raise
        raise ToolError(str(exc), code="kimodo_runtime_unavailable") from exc
    binding = _finalize_scene_animation(server, scene, result.path.animation)
    payload = result.to_dict()
    payload["scene_id"] = binding["scene_id"]
    payload["scene_binding"] = binding["scene_binding"]
    return payload


def _motion_matching_geometry_reader():
    """Motion Matching's live scene geometry (footprints and mesh tops), or
    None when it is not installed (sit and lie are then refused)."""
    try:
        module = importlib.import_module("witwin_motion_matching.scene_geometry")
        return getattr(module, "scene_geometry")
    except (ImportError, AttributeError):
        return None


def _motion_matching_route_body_validation(
    scene: Any,
    object_id: str,
    start_time: float,
    end_time: float,
) -> dict[str, Any]:
    """The final-LBS body check for a route walk, without relocation: moving
    or turning the whole track would take the person off its route."""
    try:
        module = importlib.import_module("witwin_motion_matching.body_collision_service")
        validate = getattr(module, "body_collision_report")
    except (ImportError, AttributeError) as exc:
        raise ToolError(
            "humanoid body safety requires the Motion Matching collision provider",
            code="humanoid_clearance_unavailable",
        ) from exc
    try:
        return validate(
            scene,
            {
                "object_id": object_id,
                "start": float(start_time),
                "end": float(end_time),
                "fps": 30.0,
                "max_samples": 10000,
                "max_contacts": 200,
                "tolerance_m": 0.0001,
                "volume_model": "convex_parts",
            },
            default_project_dir(),
        )
    except Exception as exc:  # provider exception type is intentionally behind the port
        if str(exc) != "no exact static environment geometry":
            raise
        return {
            "ok": True,
            "status": "clear_at_samples",
            "samples": 0,
            "affected_samples": 0,
            "contacts": [],
            "limitations": ["The scene contained no static obstacle triangles to test."],
        }


def make_check_genesis_runtime_tool(ctx: Any) -> Tool:
    @tool(
        name="check_genesis_runtime",
        description="Inspect Genesis package, CUDA, downloadable model, and licensed SMPL asset readiness.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        output_schema={"type": "object"},
        permission_scope="genesis.runtime.read",
        permission_tier="read",
    )
    def check_genesis_runtime(args: dict[str, Any]) -> dict[str, object]:
        del args
        payload = inspect_runtime().to_dict()
        active_backend = _active_backend_name()
        kimodo_runtime = inspect_kimodo_runtime()
        radar_runtime = _inspect_radar_pipeline_runtime()
        payload["active_backend"] = active_backend
        payload["kimodo_runtime"] = kimodo_runtime
        payload["radar_runtime"] = radar_runtime
        generator_ready = (
            bool(kimodo_runtime["ready"] and not payload.get("missing_smpl_files"))
            if active_backend == "kimodo"
            else bool(payload["ready"])
        )
        payload["pipeline_ready"] = bool(generator_ready and radar_runtime["ready"])
        # Preserve the legacy key, but do not present a package check as a
        # successful scene measurement or confirmation of optional interfaces.
        payload["readiness_scope"] = "packages_models_native_imports_and_cuda_only"
        payload["human_radar_validation"] = "not_run"
        return payload

    return check_genesis_runtime


def make_generate_smpl_motion_tool(ctx: Any, *, backend: Backend | None = None) -> Tool:
    @tool(
        name="generate_smpl_motion",
        description=_DESCRIPTION,
        input_schema=_INPUT_SCHEMA,
        output_schema={"type": "object"},
        side_effects=True,
        requires_confirmation=True,
        permission_scope="genesis.generate",
        permission_tier="execute",
        timeout=900.0,
    )
    async def generate_smpl_motion(args: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(_run_generate, args, backend)

    return generate_smpl_motion


def make_generate_humanoid_animation_tool(ctx: Any, *, backend: Backend | None = None) -> Tool:
    @tool(
        name="generate_humanoid_animation",
        description=_ANIMATION_DESCRIPTION,
        input_schema=_ANIMATION_INPUT_SCHEMA,
        output_schema={"type": "object"},
        side_effects=True,
        requires_confirmation=True,
        permission_scope="genesis.generate_scene_animation",
        permission_tier="execute",
        timeout=900.0,
        tags=_HUMANOID_ANIMATION_RETRIEVAL_TAGS,
    )
    async def generate_humanoid_animation(args: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(_run_generate_animation, ctx, args, backend)

    return generate_humanoid_animation


def _run_generate(args: dict[str, Any], backend: Backend | None) -> dict[str, Any]:
    backend = backend if backend is not None else configured_backend()
    result = generate_motion_asset(
        default_project_dir(),
        prompt=args["prompt"],
        motion_length=float(args.get("motion_length", 6.0)),
        seed=int(args.get("seed", 10)),
        device=int(args.get("device", 0)),
        guidance_param=float(args.get("guidance_param", 2.5)),
        smplify_iters=int(args.get("smplify_iters", 20)),
        output_name=args.get("output_name"),
        backend=backend,
    )
    payload = result.to_dict()
    payload["humanoid_binding"] = {
        "component": "SMPLMotion",
        "motion_path": payload["motion_path"],
        "pose_key": payload["pose_key"],
        "betas_key": payload["betas_key"],
        "trans_key": payload["trans_key"],
        "fps": payload["fps"],
    }
    return payload


def _run_generate_animation(ctx: Any, args: dict[str, Any], backend: Backend | None) -> dict[str, Any]:
    backend = backend if backend is not None else configured_backend()
    scene = _get_scene(ctx, args.get("scene_id"))
    if scene is None:
        raise ToolError("no active scene", code="no_scene")
    server = _get_server(ctx)
    try:
        result = create_humanoid_animation(
            scene=scene,
            project_dir=default_project_dir(),
            prompt=args["prompt"],
            motion_length=float(args.get("motion_length", 6.0)),
            seed=int(args.get("seed", 10)),
            device=int(args.get("device", 0)),
            guidance_param=float(args.get("guidance_param", 2.5)),
            smplify_iters=int(args.get("smplify_iters", 20)),
            output_name=args.get("output_name"),
            body_name=args.get("body_name"),
            body_object_id=args.get("body_object_id"),
            motion_name=args.get("motion_name"),
            position=args.get("position"),
            start_time=(float(args["start_time"]) if "start_time" in args else None),
            autoplay=bool(args.get("autoplay", False)),
            loop=bool(args.get("loop", True)),
            playback_mode=str(args.get("playback_mode", "legacy_controller")),
            backend=backend,
            mesh_assets=getattr(server, "mesh_assets", None),
            broadcast=_get_broadcast_for_server(server),
            serialize_object=_get_serializer(server, scene),
            body_validator=_motion_matching_body_validation,
        )
    except HumanoidTrajectoryBlocked as exc:
        raise ToolError(
            str(exc),
            code="humanoid_trajectory_blocked",
            detail={"blockers": exc.blockers},
        ) from exc
    return _finalize_scene_animation(server, scene, result)


def _active_backend_name() -> str:
    return configured_backend_name()


def _finalize_scene_animation(server: Any, scene: Any, result: Any) -> dict[str, Any]:
    _save_generated_scene(server, scene)
    payload = result.to_dict()
    payload["scene_id"] = getattr(scene, "scene_id", None)
    payload["scene_binding"] = {
        "body_object_id": payload["body_object_id"],
        "motion_object_id": payload["motion_object_id"],
        "timeline_track": payload["timeline_track"],
        "radar_target_contract": {
            "version": "studio_skinned_mesh_baked_transforms_v1",
            "target_object_id": payload["body_object_id"],
            "component": "SkinnedMesh",
            "required_playback_mode": "baked_transforms",
            "site_budget_owner": "witwin.radar.target_site_budget",
        },
    }
    return payload


def _motion_matching_body_validation(
    scene: Any,
    object_id: str,
    start_time: float,
    end_time: float,
) -> dict[str, Any]:
    """Run the shared final-LBS validator before the Genesis transaction commits."""
    try:
        module = importlib.import_module("witwin_motion_matching.body_collision_service")
        # Motion Matching owns collision recovery. Genesis still owns only the
        # generated motion and scene binding; the bridge may translate the
        # unchanged root track and revalidate it against the furnished room.
        validate = getattr(module, "body_collision_report_with_relocation")
    except (ImportError, AttributeError) as exc:
        raise ToolError(
            "humanoid body safety requires the Motion Matching collision provider",
            code="humanoid_clearance_unavailable",
        ) from exc
    try:
        return validate(
            scene,
            {
                "object_id": object_id,
                "start": float(start_time),
                "end": float(end_time),
                "fps": 30.0,
                "max_samples": 10000,
                "max_contacts": 200,
                "tolerance_m": 0.0001,
                "volume_model": "convex_parts",
            },
            default_project_dir(),
        )
    except Exception as exc:  # provider exception type is intentionally behind the port
        if str(exc) != "no exact static environment geometry":
            raise
        return {
            "ok": True,
            "status": "clear_at_samples",
            "samples": 0,
            "affected_samples": 0,
            "contacts": [],
            "limitations": ["The scene contained no static obstacle triangles to test."],
        }


def _inspect_radar_pipeline_runtime() -> dict[str, Any]:
    import torch

    requirements = {
        "witwin-radar": (">=0.3,<0.5", "witwin.radar"),
        "witwin-channel": (">=0.5,<0.6", "witwin.channel"),
        "rayd": (">=0.8,<0.9", "rayd.torch"),
    }
    versions = {}
    distributions = {}
    issues = []
    import_errors = {}
    for package_name, (required, module_name) in requirements.items():
        version = None
        # The Windows RayD wheel is distributed as rayd-torch but imports as
        # rayd.torch. Keep the existing logical diagnostic key and report the
        # distribution actually found; an import check is still mandatory.
        candidates = ("rayd", "rayd-torch") if package_name == "rayd" else (package_name,)
        for distribution in candidates:
            try:
                version = metadata.version(distribution)
            except metadata.PackageNotFoundError:
                continue
            distributions[package_name] = distribution
            break
        versions[package_name] = version
        if version is None:
            issues.append(f"{package_name} is not installed")
        elif version not in SpecifierSet(required):
            issues.append(f"{package_name} {version} is incompatible; expected {required}")
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001 - diagnostic must report native import failure
            import_errors[module_name] = f"{type(exc).__name__}: {exc}"
    cuda_available = bool(torch.cuda.is_available())
    if not cuda_available:
        issues.append("CUDA is unavailable for Radar sequence solving")
    return {
        "ready": not issues and not import_errors,
        "cuda_available": cuda_available,
        "cuda_device_count": int(torch.cuda.device_count()),
        "versions": versions,
        "distributions": distributions,
        "import_errors": import_errors,
        "issues": issues,
    }

def _get_scene(ctx: Any, scene_id: str | None = None) -> Any:
    if scene_id is not None:
        requested = str(scene_id).strip()
        if not requested:
            raise ToolError("scene_id must not be empty", code="missing_scene")
        server = _get_server(ctx)
        getter = getattr(server, "get_scene", None)
        if callable(getter):
            scene = getter(requested)
        else:
            current = _get_scene(ctx)
            scene = current if str(getattr(current, "scene_id", "")) == requested else None
        if scene is None:
            raise ToolError(f"scene not found or no longer open: {requested}", code="missing_scene")
        return scene
    api = getattr(ctx, "api", None)
    scene = getattr(api, "scene", None)
    if scene is not None:
        return scene
    return getattr(ctx, "scene", None)


def _get_server(ctx: Any) -> Any:
    api = getattr(ctx, "api", None)
    try:
        return getattr(api, "server", None)
    except Exception:  # noqa: BLE001 - unbound api has no server
        return None


def _get_broadcast(ctx: Any):
    server = _get_server(ctx)
    return _get_broadcast_for_server(server)


def _get_broadcast_for_server(server: Any):
    broadcast = getattr(server, "_safe_broadcast", None)
    return broadcast if callable(broadcast) else None


def _get_serializer(server: Any, scene: Any):
    serialize = getattr(server, "_serialize_object", None)
    if not callable(serialize):
        return None

    def _serialize(obj: Any) -> dict[str, Any]:
        return serialize(obj, scene)

    return _serialize


def _save_generated_scene(server: Any, scene: Any) -> None:
    persistence = getattr(server, "persistence", None)
    if persistence is None:
        return
    if scene is getattr(server, "scene", None):
        save_scene = getattr(persistence, "save_scene", None)
        if callable(save_scene):
            save_scene(force=True)
            return
    save_scene_file = getattr(persistence, "save_scene_file", None)
    if callable(save_scene_file):
        save_scene_file(scene)


def register(ctx: Any) -> None:
    """contributes.tools hook: register Genesis text-to-SMPL tools."""
    ctx.tools.register(make_check_genesis_runtime_tool(ctx))
    ctx.tools.register(make_generate_smpl_motion_tool(ctx))
    ctx.tools.register(make_generate_humanoid_animation_tool(ctx))
    ctx.tools.register(make_generate_humanoid_route_animation_tool(ctx))
    if configured_backend_name() == "mdm_tiny_cached" and os.environ.get("WITWIN_GENESIS_PREWARM", "1") != "0":
        # Load the model in the background so the first request is warm.
        from .adapter.mdm_cached_backend import prewarm

        threading.Thread(target=_prewarm_quietly, args=(prewarm,), daemon=True,
                         name="genesis-mdm-prewarm").start()


def _prewarm_quietly(prewarm: Any) -> None:
    try:
        prewarm()
    except Exception as exc:  # noqa: BLE001 - a failed warm-up leaves the cold path
        logger.warning(f"Genesis MDM Tiny warm-up failed: {exc}")
