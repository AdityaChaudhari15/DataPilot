import json, io, contextlib, sys, traceback
from dataclasses import dataclass, field, asdict
from typing import Any
nb=json.load(open('/mnt/user-data/uploads/single_multi_ai_agents_research.ipynb'))
cells=[''.join(c['source']) for c in nb['cells']]
ns={'__name__':'nb'}
import time, re, math, random, pandas as pd, numpy as np
ns.update(dict(time=time,re=re,math=math,pd=pd,np=np,field=field,dataclass=dataclass,Any=Any))
failed=[]
for i in range(27,96):
    src=cells[i]
    if '!pip' in src or 'userdata' in src: continue
    buf=io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(src,f'cell{i}','exec'),ns)
    except Exception as e:
        failed.append((i,type(e).__name__,str(e)[:80]))
for f in failed: print('skip/fail',f)
groups={k:v for k,v in ns.items() if isinstance(v,list) and v and hasattr(v[0],'task_id')}
for k,v in groups.items(): print(k,len(v))
json.dump({k:[dict(vars(t)) for t in v] for k,v in groups.items()},open('tasks_all_raw.json','w'),indent=1,default=str)
