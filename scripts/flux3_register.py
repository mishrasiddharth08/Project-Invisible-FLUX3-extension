"""Register FLUX.3 with Forge's existing checkpoint, Generate, API and gallery flow."""
from pathlib import Path
import sys

EXT = Path(__file__).resolve().parent.parent
if str(EXT) not in sys.path:
    sys.path.insert(0, str(EXT))

try:
    from modules import script_callbacks
    from flux3_forge import bridge

    script_callbacks.on_ui_settings(bridge.register_settings)
    script_callbacks.on_app_started(bridge.bind_entrypoints)
    bridge.install()
    from flux3_forge import presets
    presets.register()
    script_callbacks.on_before_ui(presets.register)
except (ImportError, AttributeError) as error:
    print(f"[Invisible-FLUX3] Forge UI unavailable: {error}")


from modules import scripts
import gradio as gr

class Script(scripts.Script):
    flux3_source_protection_control = True

    def title(self):
        return "FLUX3 source protection"

    def show(self, is_img2img):
        return scripts.AlwaysVisible if is_img2img else False

    def ui(self, is_img2img):
        with gr.Accordion("FLUX.3 source protection", open=False):
            enabled = gr.Checkbox(value=False, label="Composite supplied mask after generation (not native masked inpainting)")
            gr.Markdown("Opt-in for FLUX.3 only. Supply a grayscale mask: white uses generation, black copies exact source RGB pixels; gray blends. No region is inferred. Source/output/mask must match dimensions. Raw mask alpha is used; native blur, crop, inversion and denoising-strength controls do not control this compositor. PNG required when saving. It protects outside the mask; it cannot guarantee shapes or texture inside white areas.")
        return [enabled]
