"""Pure-data FLUX3 workflow controls. Never imports inference frameworks."""
import copy
import json
import math
import secrets

DEFAULT_ALPHA = 6.93

def alpha(value=DEFAULT_ALPHA):
    if isinstance(value,bool): raise ValueError('FLUX3 alpha must be a finite positive number')
    value=float(value)
    if not math.isfinite(value) or not 0 < value <= 1000:
        raise ValueError('FLUX3 alpha must be finite and in (0,1000]')
    return value

def seed_bits(value=32):
    if isinstance(value,bool) or str(value) not in ('32','64'):
        raise ValueError('FLUX3 seed width must be32 or64 bits')
    return int(value)

def normalize_seed(value, bits=32):
    bits=seed_bits(bits)
    if isinstance(value,bool): raise ValueError('Seeds must be integers, not booleans')
    if bits==64 and isinstance(value,float) and (not math.isfinite(value) or not value.is_integer() or abs(value)>2**53-1):
        raise ValueError('64-bit seeds supplied as floats must be exact safe integers; use an integer/string for larger seeds')
    if isinstance(value,float) and (not math.isfinite(value) or not value.is_integer()):
        raise ValueError('Seeds must be finite integers')
    return int(value) % (2**bits)

def seeds(value,count,bits=32,randbits=None):
    bits=seed_bits(bits);randbits=randbits or secrets.randbits
    if isinstance(value,list):
        if len(value)<count: raise ValueError('Seed list must contain one value per output image')
        return [randbits(bits) if int(v)<0 else normalize_seed(v,bits) for v in value[:count]]
    value=-1 if value is None else value
    base=randbits(bits) if int(value)<0 else normalize_seed(value,bits)
    return [(base+i)%(2**bits) for i in range(count)]

def controls(request):
    bits=seed_bits(request.get('seed_bits',32))
    return {'alpha':alpha(request.get('alpha',DEFAULT_ALPHA)),
            'seed_bits':bits,'seed':normalize_seed(request.get('seed',0),bits)}

def elements(value):
    if isinstance(value,str):
        if len(value.encode('utf-8'))>65536: raise ValueError('FLUX3 layout JSON exceeds64KiB')
        value=json.loads(value)
    if not isinstance(value,list) or len(value)>64: raise ValueError('FLUX3 layout must be a JSON array of at most64 rows')
    rows=copy.deepcopy(value);identities=set()
    for i,row in enumerate(rows):
        if not isinstance(row,dict):raise ValueError('Each layout row must be an object')
        row.setdefault('id',f'obj_{i+1}');identity=row['id'];bbox=row.get('bbox')
        if not isinstance(identity,str) or not identity or identity in identities:raise ValueError('Layout ids must be nonempty and unique')
        identities.add(identity)
        if not isinstance(bbox,list) or len(bbox)!=4 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1000 for v in bbox):
            raise ValueError('bbox must be finite [top,left,bottom,right] coordinates on the0-1000 canvas')
        if not bbox[0]<bbox[2] or not bbox[1]<bbox[3]:raise ValueError('bbox corners must have positive area')
        if not isinstance(row.get('desc'),str):raise ValueError('Each layout row needs a desc string')
    # Reject non-JSON/nonfinite auxiliary fields, preserve supplied provenance/text.
    encoded=json.dumps(rows,ensure_ascii=False,allow_nan=False,separators=(',',':'))
    if len(encoded.encode('utf-8'))>65536:raise ValueError('FLUX3 layout JSON exceeds64KiB')
    return rows

def caption(prompt,rows):
    rows=elements(rows)
    return prompt if not rows else prompt.rstrip()+' '+json.dumps(rows,ensure_ascii=False,allow_nan=False,separators=(',',':'))

def repair_prompt_links(document):
    """Reconnect optional PUP to its own diffusion branch; never execute nodes.

    Preserve the CURRENT linked prompt source, not stale cached widget text.
    PUP remains bypassed/off so no optional model is implicitly loaded.
    """
    result=copy.deepcopy(document);nodes={n['id']:n for n in result['nodes']};definitions={g['id']:g for g in result['definitions']['subgraphs']}
    links={link[0]:link for link in result['links']};last=max(links,default=0);changes=[]
    def connect(target,slot,source,source_slot):
        nonlocal last
        old=nodes[target]['inputs'][slot].get('link')
        if old is not None:
            previous=links.pop(old);outputs=nodes[previous[1]]['outputs'][previous[2]]
            outputs['links']=[i for i in (outputs.get('links') or []) if i!=old]
        last+=1;links[last]=[last,source,source_slot,target,slot,'STRING'];nodes[target]['inputs'][slot]['link']=last
        outputs=nodes[source]['outputs'][source_slot];outputs.setdefault('links',[])
        if outputs['links'] is None:outputs['links']=[]
        outputs['links'].append(last)
    for target,pup in [(128,127),(156,174)]:
        if target not in nodes or pup not in nodes:raise ValueError('Reference workflow structure differs; explicit review required')
        if nodes[target]['type'] not in definitions or nodes[pup]['type'] not in definitions:raise ValueError('Unknown subgraph identity')
        if definitions[nodes[target]['type']]['name']!='Diffusion (FLUX 3 Klein)' or definitions[nodes[pup]['type']]['name']!='Prompt Upsampler':raise ValueError('Unexpected subgraph contract')
        old=links[nodes[target]['inputs'][0]['link']]
        if old[1]==pup:raise ValueError('Prompt handoff is already repaired; do not stack patches')
        source,source_slot=old[1:3];connect(pup,0,source,source_slot);connect(target,0,pup,0)
        nodes[pup]['mode']=4;nodes[pup]['widgets_values'][1]=False
        changes.append({'diffusion_node':target,'upsampler_node':pup,'preserved_prompt_source':[source,source_slot],'upsampler_default':'bypassed/off; explicit enable plus verified dedicated asset required'})
    result['links']=list(links.values());result['last_link_id']=last;result['revision']=int(result.get('revision',0))+1
    result.setdefault('extra',{})['project_invisible_workflow_patch']={'changes':changes,'gpu_validated':False,'note':'Prompt wiring repair only; model names are unverified references, not identities. INT8/Base/PUP are not enabled in Forge by importing this JSON. No reference-image edit or native inpainting path exists in this graph.'}
    return result
