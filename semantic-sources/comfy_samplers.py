# Reference excerpt: Comfy-Org/ComfyUI98cc0bf84b7cb1383654c5b402e7612d875d6d22; GPL-3.0, see LICENSE.COMFY
def simple_scheduler(model_sampling, steps):
    s = model_sampling
    sigs = []
    ss = len(s.sigmas) / steps
    for x in range(steps):
        sigs += [float(s.sigmas[-(1 + int(x * ss))])]
    sigs += [0.0]
    return torch.FloatTensor(sigs)
