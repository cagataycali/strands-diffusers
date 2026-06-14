"""Dynamic pipeline/model/scheduler registry — 100% diffusers coverage, zero hardcoding.

The single source of truth is diffusers' own `_import_structure` — the lazy map of
every public symbol it exposes (307 pipelines, 87 models, 54 schedulers as of
0.38). We read it at runtime, so when diffusers adds a new pipeline (e.g. a fresh
Cosmos world-foundation model), strands-diffusers supports it automatically — no
code change required.

Same philosophy as `use_aws` (wraps boto3 dynamically), `use_lerobot` (wraps
lerobot) and `use_transformers` (wraps the transformers task taxonomy): discover,
don't hardcode.

Diffusers has no single "task taxonomy" like transformers' SUPPORTED_TASKS, so we
derive structure from three places, all dynamic:

1. `diffusers._import_structure`        → every public class, grouped by submodule.
2. The `AutoPipelineFor*` mappings      → the canonical task → pipeline-class maps
   (text2image / image2image / inpainting), diffusers' closest thing to tasks.
3. Class-name heuristics                → group the long tail of pipelines by the
   modality their name implies (TextToImage / ImageToVideo / VideoToWorld / ...).
"""

from __future__ import annotations

import importlib
import inspect
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional


@lru_cache(maxsize=1)
def _import_structure() -> Dict[str, List[str]]:
    import diffusers

    ist = getattr(diffusers, "_import_structure", None) or {}
    # Flatten dotted keys but keep the group name (submodule) for context.
    return {k: list(v) for k, v in ist.items()}


@lru_cache(maxsize=1)
def all_symbols() -> Dict[str, str]:
    """Every public diffusers symbol → its kind (pipeline/model/scheduler/other)."""
    out: Dict[str, str] = {}
    for syms in _import_structure().values():
        for s in syms:
            out[s] = _classify(s)
    return out


def _classify(name: str) -> str:
    if name.endswith("Pipeline"):
        return "pipeline"
    if name.endswith("Scheduler") or name.endswith("SchedulerOutput"):
        return "scheduler"
    if name.endswith("Output"):
        return "output"
    if "Model" in name or name.startswith("Autoencoder") or name.endswith("Transformer"):
        return "model"
    return "other"


@lru_cache(maxsize=1)
def pipelines() -> List[str]:
    return sorted(n for n, k in all_symbols().items() if k == "pipeline")


@lru_cache(maxsize=1)
def models() -> List[str]:
    return sorted(n for n, k in all_symbols().items() if k == "model")


@lru_cache(maxsize=1)
def schedulers() -> List[str]:
    return sorted(n for n, k in all_symbols().items() if k == "scheduler"
                  and not n.endswith("Output"))


# ───────────────────────── AutoPipeline task maps ─────────────────────────

@lru_cache(maxsize=1)
def auto_pipeline_tasks() -> Dict[str, Dict[str, str]]:
    """diffusers' canonical task → {model-family: pipeline-class} maps.

    These are the only first-class "tasks" diffusers ships: text2image,
    image2image, inpainting. Returned as plain class-name strings.
    """
    from diffusers.pipelines.auto_pipeline import (
        AUTO_TEXT2IMAGE_PIPELINES_MAPPING,
        AUTO_IMAGE2IMAGE_PIPELINES_MAPPING,
        AUTO_INPAINT_PIPELINES_MAPPING,
    )

    def _names(m):
        return {family: cls.__name__ for family, cls in m.items()}

    return {
        "text-to-image": _names(AUTO_TEXT2IMAGE_PIPELINES_MAPPING),
        "image-to-image": _names(AUTO_IMAGE2IMAGE_PIPELINES_MAPPING),
        "inpainting": _names(AUTO_INPAINT_PIPELINES_MAPPING),
    }


# ───────────────────────── modality grouping (derived) ─────────────────────────

# Ordered (pattern → modality) rules applied to a pipeline class name. First match
# wins. Purely name-derived so new pipelines slot in without code changes.
_MODALITY_RULES = (
    # Explicit transition names first (most specific).
    (r"TextToImage|Text2Image", "text-to-image"),
    (r"TextToVideo|Text2Video|TextToWorld", "text-to-video"),
    (r"ImageToVideo|Image2Video", "image-to-video"),
    (r"VideoToVideo|Video2Video", "video-to-video"),
    (r"VideoToWorld|World", "video-to-world"),
    (r"ImageToImage|Image2Image", "image-to-image"),
    (r"Inpaint", "inpainting"),
    (r"Upscale|SuperResolution", "upscaling"),
    (r"TextToAudio|Text2Audio|Audio|Music|Speech|TTS|Bark|MusicGen|Stable[Aa]udio", "audio"),
    # Image families that share a name-stem with a video family (e.g. HunyuanDiT,
    # HunyuanImage) — classify as image BEFORE the broad video catch-all below.
    (r"DiT|HunyuanImage|HunyuanDiT", "image"),
    # Broad video/world model families (named after the architecture, not a task).
    (r"Video|Animate|Cosmos|Wan|Hunyuan|Mochi|Allegro|LTX|SkyReels|CogVideo|Latte|Genie", "video"),
    (r"ControlNet|Adapter|IP", "controlled-image"),
    # Abbreviated transition tokens — MUST precede architecture-family rules so a
    # family-named video pipeline (Kandinsky5I2V, *T2V) isn't grabbed as image.
    (r"I2V", "image-to-video"),
    (r"T2V", "text-to-video"),
    (r"V2V", "video-to-video"),
    # Transition variants on family-named pipelines (task encoded as a suffix).
    (r"Img2Img|Image2Image", "image-to-image"),
    (r"Fill", "inpainting"),
    # NOTE: no bare "Edit" rule — editing can be image OR video (e.g. ChronoEdit
    # is image-to-video). Let such names fall through to their family/Video rule.
    # Architecture-named image-gen families (task implicit = text-to-image-class).
    (r"StableDiffusion|Flux|CogView|Bria|Chroma|AuraFlow|Amused|Kandinsky|"
     r"PixArt|Sana|Lumina|DeepFloyd|Wuerstchen|Kolors|HiDream|Janus|OmniGen|"
     r"Marigold|VisualCloze", "image"),
    # Architecture-named audio families.
    (r"AudioLDM|StableAudio|MusicGen|Musicgen|AceStep|Dance|Spectrogram", "audio"),
    (r"Image", "image"),
)


def modality_of(pipeline_name: str) -> str:
    for pat, mod in _MODALITY_RULES:
        if re.search(pat, pipeline_name):
            return mod
    return "other"


def tasks_by_modality() -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for p in pipelines():
        groups.setdefault(modality_of(p), []).append(p)
    for v in groups.values():
        v.sort()
    return groups


# World-foundation-model / action-capable pipelines (the ones the agent cares
# about for robotics). Detected by name — Cosmos*, *World*, action-conditioned.
def world_foundation_models() -> List[str]:
    return sorted(p for p in pipelines()
                  if re.search(r"Cosmos|World|Wan|Hunyuan|Genie", p))


# ───────────────────────── resolution & introspection ─────────────────────────

def resolve_attr(dotted: str, root_module: str = "diffusers") -> Any:
    """Resolve a dotted path against diffusers (or a submodule).

    Examples:
        resolve_attr("StableDiffusionPipeline")
        resolve_attr("DiffusionPipeline.from_pretrained")
        resolve_attr("Cosmos3OmniPipeline")          # from-source builds
        resolve_attr("utils.export_to_video")
        resolve_attr("schedulers.UniPCMultistepScheduler.from_config")
    """
    full = dotted if dotted.startswith(root_module + ".") else f"{root_module}.{dotted}"

    try:
        return importlib.import_module(full)
    except ImportError:
        pass

    # Fast path: attribute(s) on the root module (diffusers uses lazy __getattr__
    # that raises AttributeError, not ImportError, for non-module attrs).
    try:
        root = importlib.import_module(root_module)
        obj = root
        for attr in dotted.split("."):
            obj = getattr(obj, attr)
        return obj
    except AttributeError:
        pass

    segments = full.split(".")
    for i in range(len(segments), 0, -1):
        try:
            mod = importlib.import_module(".".join(segments[:i]))
        except Exception:
            continue
        obj = mod
        try:
            for attr in segments[i:]:
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            break

    root = importlib.import_module(root_module)
    obj = root
    for attr in dotted.split("."):
        obj = getattr(obj, attr)
    return obj


def pipeline_info(name: str) -> Optional[Dict[str, Any]]:
    """Modality + __call__ signature for one pipeline class (lazily resolved)."""
    if name not in all_symbols():
        # Known from-source WFM/pipeline classes (e.g. Cosmos3OmniPipeline ships in
        # diffusers>0.38 from source). Degrade gracefully instead of erroring like a
        # typo would — the tool resolves these dynamically once the install has them.
        if re.search(r"Pipeline$", name) and re.search(
                r"Cosmos|World|Wan|Hunyuan|Genie|Omni", name):
            return {
                "name": name,
                "kind": "pipeline",
                "modality": modality_of(name),
                "available": False,
                "note": (f"'{name}' is not in this diffusers build "
                         "(likely a from-source >0.38 class). Install with: "
                         "pip install 'git+https://github.com/huggingface/diffusers' "
                         "— use_diffusers resolves it dynamically once present."),
            }
        return None
    info: Dict[str, Any] = {
        "name": name,
        "kind": all_symbols()[name],
        "modality": modality_of(name),
    }
    try:
        cls = resolve_attr(name)
        call = getattr(cls, "__call__", None)
        if call is not None:
            info["call_params"] = _sig_params(call)
        fp = getattr(cls, "from_pretrained", None)
        if fp is not None and getattr(fp, "__doc__", None):
            info["from_pretrained_doc"] = fp.__doc__[:400]
        if cls.__doc__:
            info["doc"] = cls.__doc__[:600]
    except Exception as e:  # resolution may fail on from-source-only classes
        info["note"] = f"class not resolvable in this diffusers build: {e}"
    return info


def describe(obj: Any, max_doc: int = 600) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "kind": type(obj).__name__,
        "name": getattr(obj, "__name__", str(obj)[:80]),
    }
    if inspect.isclass(obj):
        info["methods"] = [
            m for m in dir(obj)
            if not m.startswith("_") and callable(getattr(obj, m, None))
        ][:40]
        for ctor in ("from_pretrained", "__call__", "__init__"):
            fn = getattr(obj, ctor, None)
            if fn is not None:
                try:
                    info[f"{ctor}_params"] = _sig_params(fn)
                    if fn.__doc__:
                        info[f"{ctor}_doc"] = fn.__doc__[:max_doc]
                except (ValueError, TypeError):
                    continue
    elif callable(obj):
        try:
            info["params"] = _sig_params(obj)
        except (ValueError, TypeError):
            pass
        if obj.__doc__:
            info["doc"] = obj.__doc__[:max_doc]
    elif inspect.ismodule(obj):
        info["public"] = [n for n in dir(obj) if not n.startswith("_")][:50]
    else:
        info["value"] = str(obj)[:200]
    return info


def _sig_params(fn: Any) -> Dict[str, Dict[str, Any]]:
    sig = inspect.signature(fn)
    return {
        name: {
            "default": ("REQUIRED" if p.default is inspect.Parameter.empty
                        else str(p.default)),
            "annotation": (None if p.annotation is inspect.Parameter.empty
                           else str(p.annotation)),
        }
        for name, p in sig.parameters.items()
        if name not in ("self", "args", "kwargs")
    }
