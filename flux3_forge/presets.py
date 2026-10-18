"""Native FLUX3 preset. The packaged pipeline owns matching TE/VAE selection."""
from enum import Enum

NAME = 'flux3'

def _member(enum_cls: type[Enum]):
    if NAME in enum_cls.__members__: return enum_cls[NAME]
    member = object.__new__(enum_cls)
    member._name_, member._value_ = NAME, max(m.value for m in enum_cls)+1
    member.__objclass__, member._sort_order_ = enum_cls, len(enum_cls._member_names_)
    type.__setattr__(enum_cls, NAME, member)
    enum_cls._member_map_[NAME] = member; enum_cls._member_names_.append(NAME)
    enum_cls._value2member_map_[member.value] = member
    if hasattr(enum_cls,'_hashable_values_'): enum_cls._hashable_values_.append(member.value)
    return member

def register():
    from modules import shared
    from modules_forge import presets
    from . import bridge
    # Never invent a checkpoint or select a missing/unvalidated package.
    checkpoint = bridge.register_checkpoint()
    if checkpoint is None: return False
    arch = _member(presets.PresetArch)
    presets.SAMPLERS[arch] = 'Euler'; presets.SCHEDULERS[arch] = 'Automatic'
    presets.STEPS[arch] = 16; presets.CFG[arch] = 1.0
    templates = {}; presets.register(templates)
    for key,info in templates.items():
        if (key.startswith(NAME+'_') or key.endswith('_'+NAME)) and key not in shared.opts.data_labels:
            shared.opts.add_option(key,info)
    current = shared.opts.data.get('forge_checkpoint_flux3')
    if current not in (*checkpoint.ids,checkpoint.name,checkpoint.title,checkpoint.filename):
        shared.opts.data['forge_checkpoint_flux3'] = checkpoint.name
    # No external native components: the validated package supplies Qwen3VL/VAE.
    shared.opts.data['forge_additional_modules_flux3'] = []
    return True
