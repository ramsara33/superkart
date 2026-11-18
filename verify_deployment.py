"""Run against an explicitly selected live API; retain every response and batch row."""
from pathlib import Path
import os, json, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests
import joblib

ROOT=Path(__file__).resolve().parent
base=os.environ.get('SUPERKART_API_URL','').rstrip('/')
if not base:
    raise SystemExit('Set SUPERKART_API_URL to the actual API URL. No remote deployment is assumed.')
model=joblib.load(ROOT/'backend/superkart_model.joblib')
batch=pd.read_csv(ROOT/'Batch_Data_SuperKart.csv')
evidence=[]
def record(name,method,endpoint,expected,**kwargs):
    start=time.perf_counter()
    result=requests.request(method,base+endpoint,timeout=30,**kwargs)
    body=result.json()
    item={'test':name,'url':base+endpoint,'method':method,'expected_status':expected,'actual_status':result.status_code,
        'latency_ms':round((time.perf_counter()-start)*1000,3),'response':body,
        'request_id':result.headers.get('X-Request-ID'),'model_id':result.headers.get('X-Model-ID'),
        'passed':result.status_code==expected}
    evidence.append(item)
    assert item['passed'], item
    return result
record('health','GET','/',200)
payload=batch.iloc[0].to_dict()
response=record('single','POST','/v1/predict',200,json=payload)
assert np.isclose(response.json()['Sales'],model.predict(batch.iloc[[0]])[0])
evidence[-1]['input']=payload
response=record('batch','POST','/v1/predictbatch',200,files={'file':('batch.csv',batch.to_csv(index=False).encode(),'text/csv')})
expected=np.round(model.predict(batch),2)
assert len(response.json())==len(batch)
assert np.allclose([response.json()[str(i)] for i in range(len(batch))],expected)
evidence[-1]['input']=batch.to_dict('records')
record('missing_field','POST','/v1/predict',400,json={k:v for k,v in payload.items() if k!='Product_MRP'})
record('invalid_numeric','POST','/v1/predict',400,json={**payload,'Product_Weight':-1})
report={'executed_at_utc':datetime.now(timezone.utc).isoformat(),'base_url':base,
    'scope':'local HTTP' if '127.0.0.1' in base or 'localhost' in base else 'remote HTTP',
    'all_passed':all(i['passed'] for i in evidence),'tests':evidence,
    'batch_rows':[{'row':i,'input':batch.iloc[i].to_dict(),'expected_local_prediction':float(expected[i]),'actual_api_prediction':response.json()[str(i)],'passed':bool(np.isclose(expected[i],response.json()[str(i)]))} for i in range(len(batch))]}
out=ROOT/'evidence'/('local_http_verification.json' if report['scope']=='local HTTP' else 'remote_http_verification.json')
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(report,indent=2))
print(json.dumps({'scope':report['scope'],'tests':len(evidence),'batch_rows':len(batch),'all_passed':report['all_passed'],'report':str(out)},indent=2))
