"""Scoped Forge Neo bridge for a validated local FLUX.3 package."""
from __future__ import annotations

from functools import wraps
import inspect
from pathlib import Path
import math
import secrets
import sys
import threading

from . import protection
from . import workflow_support

TAG = "[Invisible-FLUX3]"
_installed = False
_undo = []
_registered = {}
_created_cards = []
_lock = threading.RLock()
_owned_process_wrappers = set()


def _setting(name, default=None):
    try:
        from modules import shared
        return getattr(shared.opts, name, default)
    except Exception:
        return default


def configured_asset():
    if bool(_setting('flux3_comfy_enabled',False)):
        from modules.paths_internal import models_path
        from .comfy_bridge import configured_asset as comfy_asset
        return comfy_asset(_setting,models_path)
    model_dir = str(_setting("flux3_model_dir", "") or "").strip()
    if not model_dir:
        return None
    from .assets import validate_model_dir
    return validate_model_dir(model_dir, str(_setting("flux3_quantization", "bf16")))


def _remove_created_cards(sd_models):
    for ci in _created_cards:
        for table_name in ("checkpoints_list", "checkpoint_aliases"):
            table = getattr(sd_models, table_name, {})
            for key, value in list(table.items()):
                if value is ci:
                    table.pop(key, None)
    _created_cards.clear()


def register_checkpoint():
    """Expose only a real weight file from a package that passed validation."""
    from modules import sd_models
    _remove_created_cards(sd_models)
    _registered.clear()
    try:
        asset = configured_asset()
    except (OSError, ValueError) as error:
        print(f"{TAG} package unavailable: {error}")
        return None
    if not asset:
        return None
    path = Path(asset["checkpoint_path"]).resolve()
    if not path.is_file():
        print(f"{TAG} validated package did not provide a real checkpoint file")
        return None
    key = str(path).casefold()
    _registered[key] = dict(asset)
    existing = next((ci for ci in sd_models.checkpoints_list.copy().values()
                     if str(Path(ci.filename).resolve()).casefold() == key), None)
    if existing is not None:
        return existing
    ci = sd_models.CheckpointInfo(str(path))
    label = str(asset.get("display_name") or f"FLUX.3 â€” {asset['quantization'].upper()}")
    ci.name = label
    ci.title = label + (f" [{ci.shorthash}]" if getattr(ci, "shorthash", None) else "")
    ci.short_title = ci.title
    ci.ids += [ci.name, ci.title, str(path)]
    ci.register()
    _created_cards.append(ci)
    return ci


def selected(p=None, name=None):
    from modules import sd_models, shared
    override = getattr(p, "override_settings", {}) or {}
    name = name or override.get("sd_model_checkpoint") or shared.opts.sd_model_checkpoint
    ci = sd_models.checkpoint_aliases.get(name) or sd_models.checkpoints_list.get(name)
    if ci is None:
        ci = next((item for item in sd_models.checkpoints_list.copy().values()
                   if name in (item.name, item.title, getattr(item, "short_title", None))), None)
    if ci is None:
        matches=[asset for asset in _registered.values() if name in (Path(asset['checkpoint_path']).name,asset.get('display_name'))]
        if len(matches)>1:raise ValueError('Ambiguous registered FLUX.3 checkpoint alias; select its full dropdown name.')
        if matches:return matches[0]
        if bool(_setting('flux3_comfy_enabled',False)) and 'flux3-image-klein-' in str(name).lower():
            raise ValueError('Experimental matched Comfy profile is unavailable or differs from the selected family/precision. Validate its local checkpoint, encoder and VAE before Generate.')
        return None
    asset=_registered.get(str(Path(ci.filename).resolve()).casefold())
    parent=Path(ci.filename).parent
    if asset is None and bool(_setting('flux3_comfy_enabled',False)) and parent.name.casefold()=='flux-3' and parent.parent.name.casefold()=='stable-diffusion':
        raise ValueError('Experimental matched Comfy profile is unavailable or differs from the selected family/precision. Validate its local checkpoint, encoder and VAE before Generate.')
    return asset


def _replace(owner, name, replacement):
    previous = getattr(owner, name)
    setattr(owner, name, replacement)
    _undo.append((owner, name, previous, replacement))


def _release():
    for module_name in ('runtime','comfy_bridge'):
        try:
            from importlib import import_module
            import_module('.'+module_name,__package__).unload()
        except Exception as error:
            print(f"{TAG} {module_name} unload warning: {error}")


def refresh():
    _release()
    try:
        register_checkpoint()
    except Exception as error:
        print(f"{TAG} refresh failed: {error}")


def install():
    global _installed
    if _installed:
        return True
    try:
        from modules import processing, scripts, sd_models, script_callbacks, shared
        inspect.signature(processing.process_images).bind(object())
        inspect.signature(scripts.ScriptRunner.run).bind(object(), object(), 0)
        process, run, listing = processing.process_images, scripts.ScriptRunner.run, sd_models.list_models

        @wraps(run)
        def wrapped_run(runner, p, *args, **kwargs):
            asset = selected(p) if _installed else None
            if asset is not None:
                return generate(p, asset)
            _release()
            return run(runner, p, *args, **kwargs)

        @wraps(process)
        def wrapped_process(p, *args, **kwargs):
            asset = selected(p) if _installed else None
            if asset is not None:
                return generate(p, asset)
            _release()
            return process(p, *args, **kwargs)

        @wraps(listing)
        def wrapped_list(*args, **kwargs):
            result = listing(*args, **kwargs)
            if _installed:
                register_checkpoint()
            return result

        _replace(scripts.ScriptRunner, "run", wrapped_run)
        _replace(processing, "process_images", wrapped_process)
        _owned_process_wrappers.add(wrapped_process)
        _replace(sd_models, "list_models", wrapped_list)
        for module_name in ("modules.api.api", "modules.txt2img", "modules.img2img"):
            module = sys.modules.get(module_name)
            if module is not None and getattr(module, "process_images", None) is process:
                _replace(module, "process_images", wrapped_process)
        register_checkpoint()
        option = shared.opts.data_labels.get("sd_model_checkpoint")
        if option is not None:
            previous_change = option.onchange

            def checkpoint_changed():
                _release()
                if previous_change is not None:
                    return previous_change()
            _replace(option, "onchange", checkpoint_changed)
        script_callbacks.on_script_unloaded(uninstall)
        _installed = True
        return True
    except Exception as error:
        uninstall()
        print(f"{TAG} Forge hook drift; bridge disabled: {error}")
        return False


def bind_entrypoints(*_):
    """Bind late-imported UI/API aliases after Forge has finished loading scripts."""
    if not _installed:
        return
    with _lock:
        for name in ("modules.processing", "modules.api.api", "modules.txt2img", "modules.img2img"):
            owner = sys.modules.get(name)
            current = getattr(owner, "process_images", None) if owner else None
            if not callable(current) or current in _owned_process_wrappers:
                continue
            def make_dispatch(original):
                @wraps(original)
                def dispatch(p, *args, **kwargs):
                    asset = selected(p) if _installed else None
                    if asset is not None:
                        return generate(p, asset)
                    _release()
                    return original(p, *args, **kwargs)
                return dispatch
            dispatch = make_dispatch(current)
            _replace(owner, "process_images", dispatch)
            _owned_process_wrappers.add(dispatch)


def uninstall():
    global _installed
    _release()
    for owner, name, previous, replacement in reversed(_undo):
        if getattr(owner, name, None) is replacement:
            setattr(owner, name, previous)
    _undo.clear()
    _owned_process_wrappers.clear()
    try:
        from modules import sd_models
        _remove_created_cards(sd_models)
    except Exception:
        pass
    _registered.clear()
    _installed = False


def register_settings():
    import gradio as gr
    from modules import shared
    section = ("flux3", "Project Invisible FLUX.3")
    options = {
        'flux3_comfy_enabled': shared.OptionInfo(False,'Experimental matched Comfy profiles (requires verified local components; separate from the original FP8 route)',section=section),
        'flux3_comfy_family': shared.OptionInfo('base','Experimental Comfy model family',gr.Dropdown,{'choices':['base','distilled']},section=section),
        'flux3_comfy_precision': shared.OptionInfo('int8_convrot','Experimental Comfy checkpoint storage',gr.Dropdown,{'choices':['int8_convrot','bf16']},section=section),
        'flux3_comfy_encoder_precision': shared.OptionInfo('int8_convrot','Experimental Comfy Qwen3-VL-4B storage',gr.Dropdown,{'choices':['int8_convrot','bf16']},section=section),
        'flux3_comfy_alpha': shared.OptionInfo(16.0,'Experimental Comfy schedule alpha',gr.Number,section=section),
        'flux3_comfy_seed_bits': shared.OptionInfo('64','Experimental Comfy seed width',gr.Dropdown,{'choices':['32','64']},section=section),
        "flux3_workflow_controls": shared.OptionInfo(False, "Experimental workflow controls (GPU validation pending)", section=section),
        "flux3_alpha": shared.OptionInfo(6.93, "Experimental FLUX.3 schedule alpha", gr.Number, section=section),
        "flux3_seed_bits": shared.OptionInfo("32", "Experimental seed width", gr.Dropdown, {"choices": ["32", "64"]}, section=section),
        "flux3_layout_elements": shared.OptionInfo("[]", "Experimental layout JSON: id, bbox [top,left,bottom,right] (0-1000), desc; use matching <id> in caption", gr.Textbox, {"lines": 5}, section=section),
        "flux3_model_dir": shared.OptionInfo("", "Local official FLUX.3 package folder", section=section),
        "flux3_quantization": shared.OptionInfo("bf16", "FLUX.3 precision", gr.Dropdown,
                                                {"choices": ["bf16", "fp8"]}, section=section),
        "flux3_memory_mode": shared.OptionInfo("balanced", "FLUX.3 memory mode", gr.Dropdown,
                                               {"choices": ["balanced", "resident"]}, section=section),
        "flux3_caption_acknowledged": shared.OptionInfo(False,
            "I understand FLUX.3 expects a complete caption; this extension does not pretend to run PUP/prompt enhancement.",
            section=section),
        "flux3_reference_acknowledged": shared.OptionInfo(False,
            "I understand img2img denoising strength is not used; input images condition FLUX.3 directly.", section=section),
        "flux3_spectrum": shared.OptionInfo(False, "Spectrum acceleration (experimental; may change image details)", section=section),
        "flux3_attention_backend": shared.OptionInfo("auto", "FLUX.3 attention", gr.Dropdown,
            {"choices": ["auto", "forge", "torch"]}, section=section),
    }
    for key, option in options.items():
        if key not in shared.opts.data_labels:
            option.onchange = refresh
            shared.opts.add_option(key, option)


def _as_list(value, count, label):
    values = value if isinstance(value, list) else [value] * count
    if len(values) < count:
        raise ValueError(f"{label} list must contain one value per output image")
    return values


def _validate_processing(p, model_family="distilled"):
    if not bool(_setting("flux3_caption_acknowledged", False)):
        raise ValueError("Open Settings > Project Invisible FLUX.3 and acknowledge the full-caption requirement.")
    negatives = getattr(p, "negative_prompt", "")
    negatives = negatives if isinstance(negatives, list) else [negatives]
    if any(str(value or "").strip() for value in negatives):
        raise ValueError("FLUX.3 does not support negative prompts. Clear the negative prompt box.")
    cfg = float(getattr(p, "cfg_scale", 1.0))
    if model_family == 'base':
        if not math.isfinite(cfg) or not 0 <= cfg <= 20:
            raise ValueError('Experimental Base CFG must be finite0..20.')
    elif not math.isfinite(cfg) or abs(cfg - 1.0) > 1e-6:
        raise ValueError("FLUX.3 requires CFG scale 1. Set CFG to 1.")
    if float(getattr(p, "subseed_strength", 0.0) or 0.0) != 0.0:
        raise ValueError("FLUX.3 variation seed strength is not supported. Set it to 0.")
    if getattr(p, "enable_hr", False):
        raise ValueError("FLUX.3 Hires fix is not supported.")
    if (getattr(p, "image_mask", None) is not None or getattr(p, "mask", None) is not None) and not protection.enabled(p):
        raise ValueError("FLUX.3 masks and inpainting are not supported. Explicit source protection can composite a supplied mask after generation; enable it separately.")


def _resolve_adapters(prompt, existing=None):
    """Resolve Forge's native <lora:name:strength> syntax without activating SD LoRAs."""
    from modules import extra_networks
    clean, parsed = extra_networks.parse_prompt(prompt)
    if not parsed and isinstance(existing, dict):
        parsed = existing
    if set(parsed) - {"lora"}:
        raise ValueError("FLUX.3 supports only native <lora:name:strength> adapter tags.")
    params_list = list(parsed.get("lora", []))
    if len(params_list) > 8:
        raise ValueError("FLUX.3 supports at most 8 adapters per request.")
    if not params_list:
        return clean, []
    import networks
    if not getattr(networks, "available_networks", None):
        networks.list_available_networks()
    specs = []
    for params in params_list:
        args = list(getattr(params, "positional", []))
        if not args or len(args) > 2 or getattr(params, "named", {}):
            raise ValueError("Use <lora:name:strength> without block modifiers.")
        name = args[0]
        item = (networks.available_networks.get(name)
                or getattr(networks, "available_network_aliases", {}).get(name))
        if item is None:
            raise ValueError("Adapter not found: " + name)
        strength = float(args[1]) if len(args) == 2 else 1.0
        if not math.isfinite(strength) or not -4.0 <= strength <= 4.0:
            raise ValueError("Adapter strength must be finite and between -4 and 4.")
        specs.append({"path": str(Path(item.filename).resolve()), "strength": strength})
    return clean, specs


def _seeds(p, count):
    if bool(_setting("flux3_workflow_controls", False)):
        return workflow_support.seeds(getattr(p, "seed", -1), count, _setting("flux3_seed_bits", "32"))
    raw = getattr(p, "seed", -1)
    if isinstance(raw, list):
        values = _as_list(raw, count, "Seed")
        return [secrets.randbits(32) if int(v) < 0 else int(v) % 2**32 for v in values]
    base = int(raw if raw is not None else -1)
    if base < 0:
        base = secrets.randbits(32)
    return [(base + i) % 2**32 for i in range(count)]


def generate(p, asset):
    from modules import images, processing, shared
    from . import runtime
    output, seeds, prompts, infotexts = [], [], [], []
    info = ""
    source_protection = None
    with _lock:
        try:
            comfy_profile=asset.get('comfy_profile')
            _validate_processing(p,asset.get('model_family','distilled'))
            experimental_workflow = bool(_setting("flux3_workflow_controls", False))
            workflow_alpha = workflow_support.alpha(_setting("flux3_alpha", 6.93)) if experimental_workflow else 6.93
            workflow_bits = workflow_support.seed_bits(_setting("flux3_seed_bits", "32")) if experimental_workflow else 32
            layout_rows = workflow_support.elements(_setting("flux3_layout_elements", "[]")) if experimental_workflow else []
            if comfy_profile is not None:
                experimental_workflow=True
                workflow_alpha=workflow_support.alpha(_setting('flux3_comfy_alpha',16.0))
                workflow_bits=workflow_support.seed_bits(_setting('flux3_comfy_seed_bits','64'))
                if layout_rows:raise ValueError('Layout boxes are not accepted on experimental Comfy profiles.')
                overrides=getattr(p,'override_settings',{}) or {}
                if overrides.get('forge_additional_modules') or overrides.get('sd_vae') not in (None,'','Automatic','None'):
                    raise ValueError('Matched Comfy profiles own their pinned Qwen3-VL encoder and FLUX2 VAE; incompatible external component overrides are not supported.')
                if bool(_setting('flux3_spectrum',False)):
                    raise ValueError('Spectrum is not validated for experimental Comfy profiles.')
                if getattr(p,'sampler_name','Euler')!='Euler' or getattr(p,'scheduler',None) not in (None,'Automatic','Simple'):
                    raise ValueError('Experimental Comfy profiles implement Euler with Simple schedule only.')
            source_protection = protection.prepare(p)
            if source_protection and not getattr(p, "do_not_save_samples", False) and shared.opts.samples_save and shared.opts.samples_format.lower() != "png":
                raise ValueError("Source protection requires PNG saving for pixel-exact protected source pixels.")
            count = int(getattr(p, "batch_size", 1)) * int(getattr(p, "n_iter", 1))
            if not 1 <= count <= 64:
                raise ValueError("Choose between 1 and 64 images per request.")
            width, height = int(p.width), int(p.height)
            if not (64 <= width <= 4096 and 64 <= height <= 4096 and width % 16 == 0 and height % 16 == 0):
                raise ValueError("FLUX.3 width and height must be 64â€“4096 pixels and divisible by 16.")
            if not 1 <= int(p.steps) <= 100:
                raise ValueError("FLUX.3 steps must be between 1 and 100.")
            positive = _as_list(getattr(p, "prompt", ""), count, "Prompt")
            negative = _as_list(getattr(p, "negative_prompt", ""), count, "Negative prompt")
            seed_values = (workflow_support.seeds(getattr(p,'seed',-1),count,workflow_bits) if comfy_profile is not None else _seeds(p,count))
            refs = [image.convert("RGB") for image in (getattr(p, "init_images", None) or []) if image is not None]
            if refs and not bool(_setting("flux3_reference_acknowledged", False)):
                raise ValueError("Acknowledge FLUX.3 reference-image behavior in Settings before img2img use.")
            prepared = []
            for index in range(count):
                prompt = shared.prompt_styles.apply_styles_to_prompt(str(positive[index]), getattr(p, "styles", []))
                neg = shared.prompt_styles.apply_negative_styles_to_prompt(str(negative[index] or ""), getattr(p, "styles", []))
                if neg.strip():
                    raise ValueError("FLUX.3 does not support negative prompts, including negative prompt styles.")
                prompt, adapters = _resolve_adapters(prompt, getattr(p, "extra_network_data", None))
                prompt = workflow_support.caption(prompt, layout_rows)
                prepared.append((prompt, adapters))
            if comfy_profile is not None and any(adapters for _,adapters in prepared):
                raise ValueError('Adapters are not validated for experimental Comfy profiles; clear target LoRA tags.')
            from backend import memory_management
            memory_management.unload_all_models()
            shared.state.job_count = count
            shared.state.sampling_steps = int(p.steps)
            p.sd_model_name = Path(asset["checkpoint_path"]).stem
            p.sampler_name = 'FLUX.3 Base Euler/simple' if asset.get('model_family')=='base' else 'FLUX.3 official flow-map'

            def cancelled():
                return bool(shared.state.interrupted or shared.state.skipped)

            for index in range(count):
                if cancelled():
                    break
                prompt, adapters = prepared[index]
                seed = seed_values[index]

                def progress(step, total):
                    shared.state.sampling_step = int(step)
                    shared.state.sampling_steps = int(total)
                    shared.state.textinfo = f"FLUX.3: step {step}/{total}"

                request = {"model_dir": asset["model_dir"], "quantization": asset["quantization"],
                           "memory_mode": str(_setting("flux3_memory_mode", "balanced")), "prompt": prompt,
                           "width": int(p.width), "height": int(p.height), "steps": int(p.steps),
                           "seed": seed, "images": refs, "adapters": adapters}
                request.update({"alpha": workflow_alpha, "seed_bits": workflow_bits})
                request["attention_backend"] = str(_setting("flux3_attention_backend", "auto"))
                request["spectrum"] = bool(_setting("flux3_spectrum", False))
                engine=runtime
                if comfy_profile is not None:
                    from . import comfy_bridge
                    engine=comfy_bridge
                    request.update(comfy_profile=comfy_profile,cfg=float(p.cfg_scale),exact_workflow=True,reference_edit=bool(refs),negative_prompt='')
                image, metadata = engine.generate(request, progress, cancelled)
                runner = getattr(p, "scripts", None)
                if runner is not None and hasattr(runner, "postprocess_image"):
                    from modules.scripts import PostprocessImageArgs
                    pp = PostprocessImageArgs(image, index)
                    runner.postprocess_image(p, pp)
                    image = pp.image
                shared.state.assign_current_image(image)
                extras = {"FLUX.3 precision": asset["quantization"],
                          "FLUX.3 memory mode": request["memory_mode"], "FLUX.3 caption mode": "full caption",
                          "FLUX.3 adapters": len(adapters),
                          "FLUX.3 workflow controls experimental": experimental_workflow,
                          "FLUX.3 layout elements": len(layout_rows),
                          "FLUX.3 reference mode": "direct conditioning; denoising strength unused" if refs else "none"}
                extras.update({f"FLUX.3 {key}": value for key, value in (metadata or {}).items()
                               if isinstance(value, (str, int, float, bool))})
                if source_protection:
                    extras.update({"FLUX.3 source protection":"supplied raw mask after generation; not native inpainting",
                                   "FLUX.3 protection mask sha256":source_protection.mask_sha256,
                                   "FLUX.3 protected black pixels":source_protection.protected_pixels,
                                   "FLUX.3 protection mask controls":"raw alpha; no blur, crop or inversion"})
                p.extra_generation_params.update(extras)
                info = (f"{prompt}\nSteps: {p.steps}, Sampler: {p.sampler_name}, CFG scale: {float(p.cfg_scale)}, Seed: {seed}, "
                        f"Size: {p.width}x{p.height}, Model: {p.sd_model_name}, FLUX.3 precision: {asset['quantization']}, "
                        f"FLUX.3 memory mode: {request['memory_mode']}, FLUX.3 adapters: {len(adapters)}, "
                        f"FLUX.3 alpha: {workflow_alpha}, FLUX.3 seed bits: {workflow_bits}, FLUX.3 layout elements: {len(layout_rows)}, "
                        f"FLUX.3 prompt upsampled: false, FLUX.3 references: {len(refs)}, "
                        f"FLUX.3 attention: {metadata.get('attention_backend', 'torch-sdpa')}, "
                        f"FLUX.3 Spectrum forecast steps: {metadata.get('spectrum_forecast_steps', 0)}")
                if source_protection:
                    info += f", FLUX.3 source protection: supplied-mask post-composite (not native inpainting), Mask SHA256: {source_protection.mask_sha256}"
                if not source_protection and not getattr(p, "do_not_save_samples", False) and shared.opts.samples_save:
                    images.save_image(image, p.outpath_samples, "", seed, prompt,
                                      shared.opts.samples_format, info=info, p=p)
                output.append(image)
                seeds.append(seed)
                prompts.append(prompt)
                infotexts.append(info)
                shared.state.nextjob()
        except getattr(runtime, "Cancelled", InterruptedError):
            info = "FLUX.3 generation cancelled."
        except Exception as error:
            info = f"FLUX.3: {error}"
            print(f"{TAG} {info}")
            _release()
            if not output:
                # Processed.js() omits its free-form info string. Propagate the
                # failure so API/UI callers get the reason instead of [] success.
                raise RuntimeError(info) from error
        if not output:
            comments = getattr(p, "comments", None)
            if isinstance(comments, dict):
                comments[info] = 1
        result = processing.Processed(p, output, seeds[0] if seeds else -1, info,
            all_prompts=prompts or None, all_negative_prompts=[""] * len(prompts) or None,
            all_seeds=seeds or None, infotexts=infotexts or None)
        runner = getattr(p, "scripts", None)
        if runner is not None and hasattr(runner, "postprocess"):
            runner.postprocess(p, result)
        if source_protection:
            if len(result.images) != len(seeds):
                raise ValueError("Source protection cannot safely align callbacks that change the output count.")
            result.images = [protection.apply(image, source_protection) for image in result.images]
            if result.images:
                shared.state.assign_current_image(result.images[-1])
            if not getattr(p, "do_not_save_samples", False) and shared.opts.samples_save:
                for index, image in enumerate(result.images):
                    saved_path, _ = images.save_image(image, p.outpath_samples, "", seeds[index], prompts[index],
                                                      "png", info=infotexts[index], p=p)
                    protection.verify_saved(saved_path, image)
        return result
