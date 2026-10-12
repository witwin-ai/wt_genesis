"""Agent tools for wt_genesis."""
from __future__ import annotations

import asyncio
import importlib
from importlib import metadata
from typing import Any

from packaging.specifiers import SpecifierSet

from witwin_server.plugins.paths import default_project_dir
from witwin_server.tools.base import Tool, ToolError, tool

from witwin.genesis.assets import inspect_runtime
from witwin.genesis.motion_export import Backend, generate_motion_asset
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
        radar_runtime = _inspect_radar_pipeline_runtime()
        payload["radar_runtime"] = radar_runtime
        payload["pipeline_ready"] = bool(payload["ready"] and radar_runtime["ready"])
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
    scene = _get_scene(ctx, args.get("scene_id"))
    if scene is None:
        raise ToolError("no active scene", code="no_scene")
    server = _get_server(ctx)
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
        playback_mode=str(args.get("playback_mode", "legacy_controller")),
        backend=backend,
        mesh_assets=getattr(server, "mesh_assets", None),
        broadcast=_get_broadcast_for_server(server),
        serialize_object=_get_serializer(server, scene),
    )
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
