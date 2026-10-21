"""Download only the pinned public observer snapshot on remote storage."""
from pathlib import Path
import concurrent.futures, hashlib, json, os, time, urllib.request
ROOT=Path('/orcd/pool/003/kirilb/Steganography/assets/qwen35_27b_fp8')
REV='97f5941bf617e31c5e237364a8602ce3f03a551a'
DEST=ROOT/REV
DEST.mkdir(parents=True,exist_ok=True)
def atomic(path,data):
 tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');os.replace(tmp,path)
def hashed(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
url='https://huggingface.co/api/models/Qwen/Qwen3.5-27B-FP8/tree/'+REV+'?recursive=true'
items=json.load(urllib.request.urlopen(url,timeout=60))
items=[{k:x[k] for k in ('path','size','oid','lfs') if k in x} for x in items if x['type']=='file' and x['path']!='.gitattributes']
assert len([x for x in items if x['path'].endswith('.safetensors')])==11
atomic(ROOT/'DOWNLOAD_STATUS.json',dict(status='downloading',revision=REV,started_at=time.time(),files=len(items)))
def one(item):
 name=item['path'];p=DEST/name
 assert '/' not in name and '..' not in name
 if not p.exists() or p.stat().st_size!=item['size']:
  temp=p.with_name(name+'.partial')
  with urllib.request.urlopen('https://huggingface.co/Qwen/Qwen3.5-27B-FP8/resolve/'+REV+'/'+name,timeout=180) as src,temp.open('wb') as out:
   while True:
    b=src.read(8*1024*1024)
    if not b:break
    out.write(b)
  assert temp.stat().st_size==item['size'],name
  os.replace(temp,p)
 actual=hashed(p)
 if 'lfs' in item:
  assert actual==item['lfs']['oid'],name
 else:
  data=p.read_bytes();assert hashlib.sha1(('blob '+str(len(data))+'\0').encode()+data).hexdigest()==item['oid'],name
 print('VERIFIED',name,item['size'],flush=True)
 return dict(path=name,size=p.stat().st_size,sha256=actual,hub_oid=item['oid'])
try:
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: files=list(pool.map(one,items))
 atomic(ROOT/'MODEL_MANIFEST.json',dict(repo='Qwen/Qwen3.5-27B-FP8',revision=REV,path=str(DEST),files=files,verified_at=time.time()))
 atomic(ROOT/'DOWNLOAD_STATUS.json',dict(status='complete',revision=REV,completed_at=time.time(),files=len(files)))
except BaseException as e:
 atomic(ROOT/'DOWNLOAD_STATUS.json',dict(status='failed',revision=REV,error=repr(e),time=time.time()))
 raise
