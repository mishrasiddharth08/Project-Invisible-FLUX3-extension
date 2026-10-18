"""Disabled-by-default preview/save adapter. No ML imports or global Forge mutation."""
import copy,hashlib,html,io,json,os,re,struct,tempfile,zlib
from pathlib import Path
SIGNATURE=b'\x89PNG\r\n\x1a\n'
class PlanCancelled(InterruptedError):pass
def gui_preview(plan):
 if not plan.get('experimental') or not plan.get('requests'):raise ValueError('Reviewed experimental plan required')
 first=plan['requests'][0]
 seed=first['seed']
 if isinstance(seed,bool) or not isinstance(seed,int) or not 0<=seed<2**64:raise ValueError('Invalid64-bit seed')
 values={'txt2img_width':plan['width'],'txt2img_height':plan['height'],'txt2img_steps':plan['steps'],'txt2img_cfg_scale':plan['cfg']}
 if seed<=2**53-1:values['txt2img_seed']=seed
 return {'can_execute':False,'requires_explicit_experimental_optin':True,'native_control_values':values,'seed_text':str(seed),'prompts':copy.deepcopy(plan['requests']),'summary_html':'<p>Experimental preview only: '+html.escape(plan['model']['model_family'])+'; '+str(len(plan['requests']))+' prompts; '+str(plan['width'])+' x '+str(plan['height'])+'. Matching assets and backend validation required.</p>'}
def gradio_updates(preview,gradio):
 return {name:gradio.update(value=value) for name,value in preview['native_control_values'].items()}
def _chunk(kind,data):return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
def png_chunks(raw):
 if not isinstance(raw,bytes) or len(raw)>128*1024**2 or not raw.startswith(SIGNATURE):raise ValueError('Bounded PNG bytes required')
 rows=[];offset=len(SIGNATURE)
 while offset<len(raw):
  if offset+12>len(raw):raise ValueError('Truncated PNG chunk')
  length=struct.unpack('>I',raw[offset:offset+4])[0];kind=raw[offset+4:offset+8];end=offset+12+length
  if end>len(raw):raise ValueError('Truncated PNG payload')
  payload=raw[offset+8:offset+8+length];crc=struct.unpack('>I',raw[offset+8+length:end])[0]
  if zlib.crc32(kind+payload)&0xffffffff!=crc:raise ValueError('PNG checksum mismatch')
  rows.append((kind,payload));offset=end
  if len(rows)>10000:raise ValueError('PNG chunk count exceeds bounded parser limit')
  if kind==b'IEND':break
 if offset!=len(raw) or not rows or rows[0][0]!=b'IHDR' or len(rows[0][1])!=13 or rows[-1]!=(b'IEND',b'') or not any(k==b'IDAT' for k,v in rows):raise ValueError('Invalid PNG structure')
 return rows
def annotate_png(raw,metadata,expected_size):
 rows=png_chunks(raw);size=struct.unpack('>II',rows[0][1][:8])
 if tuple(expected_size)!=size:raise ValueError('Generated dimensions differ from the exact plan')
 text=json.dumps(metadata,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode('utf-8')
 if len(text)>1024**2:raise ValueError('PNG metadata exceeds1MiB')
 result=[]
 for kind,payload in rows[:-1]:
  if kind in (b'tEXt',b'iTXt',b'zTXt') and payload.split(b'\0',1)[0] in [b'Project Invisible exact workflow',b'parameters']:continue
  result.append(_chunk(kind,payload))
 result.append(_chunk(b'iTXt',b'Project Invisible exact workflow\0\0\0\0\0'+text))
 request=metadata.get('request',{})
 if all(k in request for k in ['prompt','steps','cfg','seed','width','height']):
  infotext=f"{request['prompt']}\nSteps: {request['steps']}, Sampler: Euler, Schedule type: Simple, CFG scale: {request['cfg']}, Seed: {request['seed']}, Size: {request['width']}x{request['height']}, Exact workflow: experimental"
  result.append(_chunk(b'iTXt',b'parameters\0\0\0\0\0'+infotext.encode('utf-8')))
 result.append(_chunk(b'IEND',b''))
 return SIGNATURE+b''.join(result)
def save_png(raw,directory,prefix,metadata,expected_size):
 if not isinstance(prefix,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',prefix):raise ValueError('Safe filename prefix required; no paths or template execution')
 encoded=annotate_png(raw,metadata,expected_size);root=Path(directory).resolve();root.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile(dir=root,prefix='.exact-workflow-',suffix='.partial',delete=False) as f:
  temporary=Path(f.name);f.write(encoded);f.flush();os.fsync(f.fileno())
 try:
  for counter in range(1,1000000):
   target=root/f'{prefix}_{counter:05d}_.png'
   try:
    if os.name=='nt':os.rename(temporary,target)  # Windows refuses an existing destination.
    else:os.link(temporary,target)  # POSIX rename could overwrite, so use exclusive link.
    break
   except FileExistsError:continue
  else:raise ValueError('Output counter exhausted')
 finally:temporary.unlink(missing_ok=True)
 return {'path':str(target),'png_sha256':hashlib.sha256(encoded).hexdigest(),'metadata':copy.deepcopy(metadata)}
def run_plan(plan,engine,directory,*,allow_experimental=False,cancelled=None,prefix='FLUX3_exact'):
 if allow_experimental is not True:raise ValueError('Exact-workflow execution is disabled; explicit experimental opt-in required')
 if not plan.get('experimental') or not plan.get('requests'):raise ValueError('Reviewed experimental plan required')
 results=[]
 try:
  for index,request in enumerate(copy.deepcopy(plan['requests'])):
   if cancelled and cancelled():raise PlanCancelled('Cancelled before next prompt; completed files retained')
   image,actual=engine.generate(request,cancelled=cancelled)
   if cancelled and cancelled():raise PlanCancelled('Cancelled before saving; no false completed PNG')
   if not isinstance(actual,dict) or actual.get('model_family')!=plan['model']['model_family']:raise ValueError('Runtime model family differs from reviewed plan')
   for key in ['seed','steps','cfg','alpha','width','height']:
    if actual.get(key)!=request[key]:raise ValueError('Runtime metadata differs from requested '+key)
   buffer=io.BytesIO();image.save(buffer,format='PNG')
   metadata={'schema':1,'experimental':True,'full_chain_certified':False,'prompt_index':index,'model_identity':plan['model'],'request':request,'actual':actual}
   results.append(save_png(buffer.getvalue(),directory,prefix,metadata,(request['width'],request['height'])))
  return results
 finally:engine.unload()  # Only this explicitly supplied extension-owned engine.

def gallery_results(results):
 return [(row['path'],f"Prompt {row['metadata']['prompt_index']+1}") for row in results]
