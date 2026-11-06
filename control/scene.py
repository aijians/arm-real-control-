"""Fixed-scene polygon envelopes, independent of selected contact points.

The JSON contains solid component outlines with their separate height ranges.
This preserves the opening between the gripper plates. No CAD, mesh library,
search graph, or simulator is loaded on the board.
"""
from __future__ import annotations
import math
import numpy as np


def _rotate(points, yaw):
    c,s=math.cos(yaw),math.sin(yaw)
    return points @ np.array([[c,s],[-s,c]])


def _inside_convex(point, polygon, tolerance=0.):
    edges=np.roll(polygon,-1,axis=0)-polygon
    cross=edges[:,0]*(point[1]-polygon[:,1])-edges[:,1]*(point[0]-polygon[:,0])
    return bool(np.all(cross > tolerance*np.linalg.norm(edges,axis=1)))


def _overlap(first, second, tolerance=0.):
    """Separating axes for two convex solid polygons; tangency is permitted."""
    for polygon in (first,second):
        edges=np.roll(polygon,-1,axis=0)-polygon
        normals=np.column_stack((-edges[:,1],edges[:,0]))
        normals/=np.linalg.norm(normals,axis=1)[:,None]
        a,b=first@normals.T,second@normals.T
        if np.any(np.minimum(a.max(0),b.max(0))-np.maximum(a.min(0),b.min(0))<=tolerance):
            return False
    return True


def _circle_overlap(polygon, center, radius, tolerance=0.):
    if _inside_convex(center,polygon):return True
    edges=np.roll(polygon,-1,axis=0)-polygon
    t=np.clip(np.sum((center-polygon)*edges,axis=1)/np.sum(edges*edges,axis=1),0.,1.)
    return bool(np.min(np.linalg.norm(polygon+t[:,None]*edges-center,axis=1)) < radius-tolerance)


def _box(center, size):
    center=np.asarray(center,float);half=np.asarray(size,float)/2
    return center+np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*half


class SimpleScene:
    def __init__(self, config):
        self.config=config
        self.arm=config['arm']
        self.limits=np.asarray(self.arm['joint_limits_rad'],float)
        self.d=np.asarray(self.arm['elbow_translation_mm'],float)
        self.bodies=[]
        for item in config['bodies']:
            poly=np.asarray(item['polygon_xy_mm'],float)
            poly=poly[np.any(np.abs(poly-np.roll(poly,1,axis=0))>1e-10,axis=1)]
            if len(poly)>=3:self.bodies.append((item,poly))
        self.workspace=np.asarray(config['workspace_xy_mm'],float)
        self.obstacles=[]
        for obstacle in config['obstacles']:
            center=np.asarray(obstacle['center_mm'],float);size=np.asarray(obstacle['size_mm'],float)
            self.obstacles.append((obstacle['name'],_box(center[:2],size[:2]),center[2]-size[2]/2,center[2]+size[2]/2))
        self.contact_tolerance=float(config.get('contact_tolerance_mm',.04))
        self.slider_shapes=config.get('slider_primitives',[])

    def transformed_bodies(self,q):
        yaw=self.arm['shoulder_static_yaw_rad']+q[0]
        elbow=_rotate(self.d[None],yaw)[0]
        forearm_yaw=yaw+self.arm['elbow_static_yaw_rad']+q[1]
        return [(item,_rotate(poly,yaw if item['part']=='upper' else forearm_yaw)+(0 if item['part']=='upper' else elbow))
                for item,poly in self.bodies]

    def check(self,q,sliders=None,contact=None):
        q=np.asarray(q,float)
        if q.shape!=(2,) or not np.isfinite(q).all() or np.any(q<self.limits[:,0]-1e-9) or np.any(q>self.limits[:,1]+1e-9):
            return {'obstacle':'joint_limit'}
        target_color=contact.get('color') if isinstance(contact,dict) else contact
        reference_tolerance=(contact.get('reference_contact_tolerance_mm',0.) if isinstance(contact,dict) else 0.)
        if type(reference_tolerance) not in (int,float) or not math.isfinite(reference_tolerance) or not 0<=reference_tolerance<=3.:
            return {'obstacle':'invalid_reference_contact_tolerance'}
        for item,poly in self.transformed_bodies(q):
            if np.any(poly.min(0)<self.workspace[0]-self.contact_tolerance) or np.any(poly.max(0)>self.workspace[1]+self.contact_tolerance):
                return {'obstacle':'workspace_boundary','part':item['part'],'body':item['name']}
            zlo,zhi=item['z_mm'];lo,hi=poly.min(0),poly.max(0)
            for name,shape,olo,ohi in self.obstacles:
                if zhi<=olo or ohi<=zlo:continue
                if np.any(hi<=shape.min(0)) or np.any(shape.max(0)<=lo):continue
                if _overlap(poly,shape):return {'obstacle':name,'part':item['part'],'body':item['name']}
            for color,position in (sliders or {}).items():
                center=np.asarray(position if isinstance(position,(list,tuple,np.ndarray)) else [position,155.],float)
                for shape in self.slider_shapes:
                    tolerance=(max(self.contact_tolerance,reference_tolerance) if color==target_color and shape['name']=='stem' and item.get('contact_face') else 0.)
                    if zhi<=shape['z_mm'][0] or shape['z_mm'][1]<=zlo:continue
                    if shape['shape']=='circle':
                        r=shape['radius_mm']
                        if np.any(hi<=center-r) or np.any(center+r<=lo):continue
                        hit=_circle_overlap(poly,center,r,tolerance)
                    else:
                        other=_box(center,shape['size_xy_mm'])
                        if np.any(hi<=other.min(0)) or np.any(other.max(0)<=lo):continue
                        hit=_overlap(poly,other,tolerance)
                    if hit:return {'obstacle':'slider','color':color,'primitive':shape['name'],'part':item['part'],'body':item['name']}
        return None

    def path_free(self,q_path,sliders=None,contact=None):
        return all(self.check(q,sliders,contact) is None for q in q_path)
