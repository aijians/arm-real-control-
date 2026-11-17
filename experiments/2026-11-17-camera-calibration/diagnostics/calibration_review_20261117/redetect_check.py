from pathlib import Path
import json,cv2,numpy as np
cv2.setNumThreads(2)
out=Path(__file__).parent;ev=out/'board_evidence';cal=json.loads((out/'input_intrinsics.json').read_text());meta=json.loads((ev/'samples.json').read_text());K=np.array(cal['camera_matrix']);D=np.array(cal['distortion_coefficients'])
obj=np.zeros((48,3),np.float64);obj[:,:2]=np.mgrid[0:8,0:6].T.reshape(-1,2)*6
rows=[]
for name in ['sample_001.png','sample_018.png','sample_022.png']:
 im=cv2.imread(str(ev/name));gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
 found,p=cv2.findChessboardCornersSB(gray,(8,6),cv2.CALIB_CB_NORMALIZE_IMAGE|cv2.CALIB_CB_ACCURACY)
 row=dict(file=name,full_resolution_SB_found=bool(found))
 if found:
  old=np.array(next(x for x in meta['samples'] if x['file']==name)['corners']).reshape(6,8,2)
  grid=p.reshape(6,8,2);variants=[grid,grid[::-1],grid[:,::-1],grid[::-1,::-1]];grid=min(variants,key=lambda q:np.mean((q-old)**2));p=grid.reshape(-1,1,2).astype(np.float64)
  delta=np.linalg.norm(grid-old,axis=2)
  ok,rv,tv=cv2.solvePnP(obj,p,K,D,flags=cv2.SOLVEPNP_ITERATIVE);pr=cv2.projectPoints(obj,rv,tv,K,D)[0];err=np.linalg.norm(pr.reshape(-1,2)-p.reshape(-1,2),axis=1)
  row.update(delta_saved_to_SB_rms_px=float(np.sqrt(np.mean(delta**2))),delta_saved_to_SB_max_px=float(delta.max()),pose_rms_px=float(np.sqrt(np.mean(err**2))),pose_max_px=float(err.max()),corners=p.reshape(-1,2).tolist())
 rows.append(row)
(out/'full_resolution_corner_review.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2));print(json.dumps([{k:v for k,v in r.items() if k!='corners'} for r in rows],ensure_ascii=False,indent=2))
