"""Pin and verify observer artifacts once on ORCD before generating run configurations."""
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

scratch=Path('/orcd/scratch/orcd/003/kirilb/Steganography/assets')
legacy=json.loads((scratch/'icrl/ASSETS.json').read_text())['models']
models={size:legacy[key] for size,key in [('4','small'),('9','observer')]}
for size,rev in [('0.8','2fc06364715b967f1860aea9cf38778875588b17'),('2','15852e8c16360a2fea060d615a32b45270f8a8fc'),('27','fc05daec18b0a78c049392ed2e771dde82bdf654')]:
    repo=f'Qwen/Qwen3.5-{size}B'
    root=scratch/('icrl' if size=='27' else 'nov02')/repo.split('/')[1]/rev
    info=HfApi().model_info(repo,revision=rev,files_metadata=True)
    assert info.sha==rev
    names=[f.rfilename for f in info.siblings if '/' not in f.rfilename and (f.rfilename.endswith(('.safetensors','.json','.jinja','.txt')))]
    if size!='27':
        snapshot_download(repo,revision=rev,local_dir=str(root),allow_patterns=names,max_workers=3)
    hashes={}
    for f in info.siblings:
        if f.rfilename not in names:continue
        p=root/f.rfilename
        with p.open('rb') as stream:
            h=hashlib.file_digest(stream,'sha256').hexdigest()
        if f.lfs:
            assert f.lfs.sha256==h
        else:
            b=p.read_bytes()
            assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==f.blob_id
        hashes[f.rfilename]=h
    models[size]=dict(repo_id=repo,revision=rev,path=str(root),artifact_sha256=hashes)
Path('observer_assets.json').write_text(json.dumps(models,indent=2))
