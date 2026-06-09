"""Agent tools for wt_genesis."""
from __future__ import annotations

import asyncio
from typing import Any

from witwin_server.plugins.paths import default_project_dir
from witwin_server.tools.base import Tool, ToolError, tool

from .adapter.motion_export import Backend, generate_motion_asset
from .adapter.scene_animation import create_humanoid_animation


_DESCRIPTION = (
    "Generate an SMPL humanoid motion asset from a text prompt using the Genesis MDM Tiny runtime. "
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
    "Generate an SMPL humanoid animation from a text prompt, place it in the current Studio scene, "
    "bind the generated motion to wt_humanoid SMPLBody/SMPLMotion components, and key the timeline "
    "so timeline playback drives the humanoid pose plus root Transform position/rotation. Use this "
    "for user requests like 'generate a human animation', 'make a person walk', or 'create an SMPL "
    "motion in the scene'."
)

_ANIMATION_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        **_INPUT_SCHEMA["properties"],
        "body_name": {
            "type": "string",
            "description": "Optional scene object name for the generated SMPL body.",
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
    )
    async def generate_humanoid_animation(args: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(_run_generate_animation, ctx, args, backend)

    return generate_humanoid_animation


def _run_generate(args: dict[str, Any], backend: Backend | None) -> dict[str, Any]:
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
    scene = _get_scene(ctx)
    if scene is None:
        raise ToolError("no active scene", code="no_scene")

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
        motion_name=args.get("motion_name"),
        position=args.get("position"),
        autoplay=bool(args.get("autoplay", False)),
        loop=bool(args.get("loop", True)),
        backend=backend,
        mesh_assets=getattr(_get_server(ctx), "mesh_assets", None),
        broadcast=_get_broadcast(ctx),
    )
    payload = result.to_dict()
    payload["scene_binding"] = {
        "body_object_id": payload["body_object_id"],
        "motion_object_id": payload["motion_object_id"],
        "timeline_track": payload["timeline_track"],
    }
    return payload


def _get_scene(ctx: Any) -> Any:
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
    broadcast = getattr(server, "_safe_broadcast", None)
    return broadcast if callable(broadcast) else None


def register(ctx: Any) -> None:
    """contributes.tools hook: register Genesis text-to-SMPL tools."""
    ctx.tools.register(make_generate_smpl_motion_tool(ctx))
    ctx.tools.register(make_generate_humanoid_animation_tool(ctx))
