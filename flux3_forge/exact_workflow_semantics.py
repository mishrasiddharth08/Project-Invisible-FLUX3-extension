"""Reviewed node semantics for the supplied FLUX3 graph; no node-package execution.

Resolution: Comfy-Org/ComfyUI98cc0bf84b7cb1383654c5b402e7612d875d6d22.
Prompt list: yolain/ComfyUI-Easy-Use4ba22aa1ec2dd606995574d93d70c6c8a3744859.
No model download, registration, pickle or arbitrary-code evaluation.
"""
import copy,hashlib,json,math

RATIOS={'1:1 (Square)':(1,1),'2:3 (Portrait Photo)':(2,3),'3:2 (Photo)':(3,2),'3:4 (Portrait Standard)':(3,4),'4:3 (Standard)':(4,3),'9:16 (Portrait Widescreen)':(9,16),'16:9 (Widescreen)':(16,9),'21:9 (Ultrawide)':(21,9)}
ALIASES={
 'flux3_image_klein_int8_convrot.safetensors':('base','flux3-image-klein-base-int8-convrot.safetensors',5807427339,'450f47e2f4811aaa396fe646ba76741f1c2f31b8'),
 'flux3_image_klein_sd_int8_convrot.safetensors':('distilled','flux3-image-klein-int8-convrot.safetensors',5827874779,'fd455429b3e87418302c0717e3ba68fe3aca525d')}
def resolution(aspect_ratio,megapixels,multiple):
 if aspect_ratio not in RATIOS:raise ValueError('Unknown reviewed aspect ratio')
 if isinstance(megapixels,bool) or not isinstance(megapixels,(int,float)) or not math.isfinite(megapixels) or not .1<=megapixels<=16:raise ValueError('Megapixels must be0.1..16')
 if isinstance(multiple,bool) or not isinstance(multiple,int) or not 8<=multiple<=128 or multiple%4:raise ValueError('Multiple must be8..128 in steps of4')
 ratio=RATIOS[aspect_ratio];unit=math.sqrt(megapixels*(1024**2)/(ratio[0]*ratio[1]))
 return tuple(int(round(unit*side/multiple)*multiple) for side in ratio)
def prompt_list(values,optional=None):
 if not isinstance(values,list) or len(values)!=5 or any(not isinstance(s,str) for s in values):raise ValueError('Exactly five literal string fields are required')
 if optional is not None and (not isinstance(optional,list) or any(not isinstance(s,str) for s in optional)):raise ValueError('Only reviewed optional string-list inputs are accepted')
 prompts=list(optional or [])+[s for s in values if s!='']
 if len(prompts)>64 or sum(len(s.encode('utf-8')) for s in prompts)>1024**2:raise ValueError('Bounded prompt list exceeds64 items/1MiB')
 return prompts  # Preserve whitespace, duplicates and input order; never join.
def checkpoint_alias(name):
 if name not in ALIASES:raise ValueError('Checkpoint name has no publisher-verified alias')
 family,current,size,pointer=ALIASES[name]
 return dict(model_family=family,current_filename=current,size_bytes=size,publisher_pointer_git_oid=pointer)
def verify_lfs_payload(path,contract):
 """Verify payload against publisher Git-LFS pointer when unauth metadata masks SHA256."""
 from pathlib import Path
 path=Path(path)
 if path.stat().st_size!=contract['size_bytes']:raise ValueError('Checkpoint byte size differs from publisher')
 with path.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
 pointer=f'version https://git-lfs.github.com/spec/v1\noid sha256:{sha}\nsize {path.stat().st_size}\n'.encode()
 oid=hashlib.sha1(b'blob '+str(len(pointer)).encode()+b'\0'+pointer).hexdigest()
 if oid!=contract['publisher_pointer_git_oid']:raise ValueError('Payload does not match pinned publisher Git-LFS object')
 return sha
def execution_plan(document,branch=128,*,allow_bypassed=False):
 d=copy.deepcopy(document);nodes={n['id']:n for n in d['nodes']};defs={g['id']:g for g in d['definitions']['subgraphs']};links={r[0]:r for r in d['links']}
 if branch not in [128,156] or branch not in nodes:raise ValueError('Only reviewed supplied diffusion branches accepted')
 n=nodes[branch]
 if n.get('mode',0)!=0 and not allow_bypassed:raise ValueError('Branch is bypassed; explicit opt-in required')
 definition=defs[n['type']];fields=[x['name'] for x in definition['inputs']]
 if len(fields)!=len(n['widgets_values']):raise ValueError('Subgraph schema changed')
 params=dict(zip(fields,n['widgets_values']));contract=checkpoint_alias(params['unet_name'])
 if (contract['model_family']=='distilled' and params['cfg']!=1) or not math.isfinite(params['cfg']):raise ValueError('CFG/model family conflict')
 if params['batch_size']!=1:raise ValueError('Exact batch>1 RNG/latent support is not validated')
 dims=[]
 for slot,output in [(1,0),(2,1)]:
  r=links[n['inputs'][slot]['link']];source=nodes[r[1]]
  if source['type']!='ResolutionSelector' or r[2]!=output:raise ValueError('Resolution source contract changed')
  dims.append(resolution(*source['widgets_values'])[output])
 r=links[n['inputs'][0]['link']];seen=set()
 while nodes[r[1]]['type']=='Reroute':
  if r[1] in seen:raise ValueError('Prompt reroute cycle')
  seen.add(r[1]);r=links[nodes[r[1]]['inputs'][0]['link']]
 source=nodes[r[1]]
 if source['type']!='easy promptList' or r[2]!=1:raise ValueError('Only reviewed direct promptList output1 route is accepted; PUP-enabled plan separate')
 prompts=prompt_list(source['widgets_values'])
 if not prompts:raise ValueError('No prompts to execute')
 seed=params['seed']
 if isinstance(seed,bool) or not isinstance(seed,int) or not 0<=seed<2**64:raise ValueError('Exact64-bit integer seed required')
 requests=[dict(prompt=p,seed=seed,seed_bits=64,batch_size=1,width=dims[0],height=dims[1],steps=params['steps'],cfg=params['cfg'],alpha=params['alpha'],sampler=params['sampler_name'],scheduler=params['scheduler'],exact_workflow=True) for p in prompts]
 return {'experimental':True,'real_weight_acceptance':False,'branch':branch,'model':contract,'width':dims[0],'height':dims[1],'sampler':params['sampler_name'],'scheduler':params['scheduler'],'steps':params['steps'],'cfg':params['cfg'],'alpha':params['alpha'],'seed':seed,'seed_bits':64,'requests':requests,'prompt_list_behavior':'One execution per string, scalar seed broadcast unchanged','PUP':'bypassed in original','boxes':'bypassed/empty in original','reference_images':'none; graph LoadImage branch is comparison only','precision':'int8_convrot DiT and Qwen3VL4B conditioning; matching FLUX2 VAE','execution':'NOT RUN; local matching assets and fully validated runtime required'}
