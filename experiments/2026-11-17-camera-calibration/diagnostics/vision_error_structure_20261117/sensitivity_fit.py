import json,sys,math
import numpy as np
D0=sys.argv[1]
cal=json.load(open(f"{D0}/used_calibration.json"))
pts=[]
for n,(rb,ro) in {"0001":(113.5,60.),"0002":(113.5,80.),"0005":(105.,75.)}.items():
    r=json.load(open(f"{D0}/result_{n}.json"))
    sl=r["pixels"]["sliders"]
    wx={}
    for k,v in r["world"].items():
        pass
    pts.append((n,"蓝",sl["blue"]["pixel_xy"],rb)); pts.append((n,"橙",sl["yellow"]["pixel_xy"],ro))
def undist(u,v,K,Dc,it=200):
    fx,fy,cx,cy=K[0][0],K[1][1],K[0][2],K[1][2]; k1,k2,p1,p2,k3=Dc
    xd,yd=(u-cx)/fx,(v-cy)/fy; x,y=xd,yd
    for _ in range(it):
        r2=x*x+y*y; rad=1+k1*r2+k2*r2*r2+k3*r2**3
        dx=2*p1*x*y+p2*(r2+2*x*x); dy=p1*(r2+2*y*y)+2*p2*x*y
        x=(xd-dx)/rad; y=(yd-dy)/rad
    return x,y
def rot(axis,deg):
    a=math.radians(deg); c,s=math.cos(a),math.sin(a)
    if axis=="x": return np.array([[1,0,0],[0,c,-s],[0,s,c]])
    if axis=="y": return np.array([[c,0,s],[0,1,0],[-s,0,c]])
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])
def world_x(p,K=None,Dc=None,R=None,C=None,Y=None,Rcam=np.eye(3)):
    K=K or cal["K"]; Dc=Dc or cal["D"]; R=np.array(R or cal["R_world_from_camera"]); C=C or cal["C_world_mm"]; Y=cal["plane_y_mm"] if Y is None else Y
    x,y=undist(p[0],p[1],K,Dc); d=R@(Rcam@np.array([x,y,1.]))
    t=(Y-C[1])/d[1]; return C[0]+t*d[0], C[2]+t*d[2]
base=[world_x(p[2]) for p in pts]
print("photo color pixel  worldX  ref  err  Z_at_plane  depth(Cx-X)")
for (n,c,px,ref),(X,Z) in zip(pts,base):
    print(f"{n} {c} {px} {X:9.4f} {ref:6.1f} {X-ref:7.3f} {Z:8.2f} {cal['C_world_mm'][0]-X:7.2f}")
def show(name,vals):
    d=[v[0]-b[0] for v,b in zip(vals,base)]
    print(f"{name:34s}"+" ".join(f"{x:+6.2f}" for x in d))
print("\nshift of world X per perturbation  (order: 1蓝 1橙 2蓝 2橙 5蓝 5橙)")
C=cal["C_world_mm"]; K=cal["K"]
show("Cx +1 mm",[world_x(p[2],C=[C[0]+1,C[1],C[2]]) for p in pts])
show("Cy +1 mm (camera->rail dY -1)",[world_x(p[2],C=[C[0],C[1]+1,C[2]]) for p in pts])
show("Cz +1 mm",[world_x(p[2],C=[C[0],C[1],C[2]+1]) for p in pts])
show("rail plane Y +1 mm",[world_x(p[2],Y=cal['plane_y_mm']+1) for p in pts])
Kf=[[K[0][0]*1.01,0,K[0][2]],[0,K[1][1]*1.01,K[1][2]],[0,0,1]]
show("fx,fy +1%",[world_x(p[2],K=Kf) for p in pts])
show("cx +10 px",[world_x(p[2],K=[[K[0][0],0,K[0][2]+10],[0,K[1][1],K[1][2]],[0,0,1]]) for p in pts])
show("cy +10 px",[world_x(p[2],K=[[K[0][0],0,K[0][2]],[0,K[1][1],K[1][2]+10],[0,0,1]]) for p in pts])
show("D = 0 (no distortion)",[world_x(p[2],Dc=[0,0,0,0,0]) for p in pts])
for ax in "xyz":
    show(f"camera rot about cam-{ax} +0.5deg",[world_x(p[2],Rcam=rot(ax,.5)) for p in pts])

err=np.array([X-p[3] for (X,_),p in zip(base,pts)])
shapes={
 "Cx (camera X / optical-centre depth)":[world_x(p[2],C=[C[0]+1,C[1],C[2]])[0]-b[0] for p,b in zip(pts,base)],
 "dY camera-rail (Cy or plane Y)":[world_x(p[2],C=[C[0],C[1]+1,C[2]])[0]-b[0] for p,b in zip(pts,base)],
 "focal length":[world_x(p[2],K=Kf)[0]-b[0] for p,b in zip(pts,base)],
 "principal point cx":[world_x(p[2],K=[[K[0][0],0,K[0][2]+10],[0,K[1][1],K[1][2]],[0,0,1]])[0]-b[0] for p,b in zip(pts,base)],
 "camera yaw (cam-y)":[world_x(p[2],Rcam=rot('y',.5))[0]-b[0] for p,b in zip(pts,base)],
}
print("\nobserved errors:",np.round(err,3),"mean %.3f sd %.3f"%(err.mean(),err.std(ddof=1)))
print("same-reference repeat (1蓝 vs 2蓝, both 113.5): %.3f mm"%(err[0]-err[2]))
print("\nsingle-source fit: err = k*shape (no free constant)")
for name,s in shapes.items():
    s=np.array(s); k=(s@err)/(s@s); r=err-k*s
    print(f"  {name:38s} k={k:+8.3f} unit  residual RMS {math.sqrt((r@r)/len(r)):.3f}  residual {np.round(r,2)}")
print("\nconstant offset + one depth-scaled source:")
for name,s in list(shapes.items())[1:]:
    A=np.column_stack([np.ones(6),np.array(s)]); sol,*_=np.linalg.lstsq(A,err,rcond=None); r=err-A@sol
    print(f"  offset {sol[0]:+7.2f} mm + {name:30s} k={sol[1]:+7.3f}  residual RMS {math.sqrt((r@r)/6):.3f}")
