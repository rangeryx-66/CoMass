"""Metrics computed from measured integration states, never used to move objects."""
import numpy as np


def flight_metrics(result,rows,bucket,rim):
    rel=result.get('release')
    if not rel:return {}
    v=np.asarray(rel['linear_velocity']);p=np.asarray(rel['com_position']);g=9.81
    xy=np.linalg.norm(v[:2]);speed=np.linalg.norm(v);vz=float(v[2])
    end=result['com_crossing']['time'] if result.get('com_crossing') else rows[-1,0]
    if result.get('airborne_environment_contacts'):
        end=min(end,min(c['time'] for c in result['airborne_environment_contacts']))
    flight=rows[(rows[:,0]>=rel['time']-1e-8)&(rows[:,0]<=end+1e-8)]
    if not len(flight):return {}
    apex=flight[np.argmax(flight[:,6])];rise=float(apex[6]-p[2]);direction=np.asarray(bucket[:2])-np.asarray(rel['position'][:2])
    frac=float(np.dot(apex[4:6]-p[:2],direction)/max(np.dot(direction,direction),1e-12))
    disc=vz*vz+2*g*(p[2]-rim)
    predicted=None;error=None
    if disc>=0:
        t=(vz+np.sqrt(disc))/g
        if t>0:
            predicted=(p[:2]+v[:2]*t).tolist();error=float(np.linalg.norm(np.array(predicted)-bucket[:2]))
    return dict(release_speed=float(speed),release_angle=float(np.arctan2(vz,xy)),
                release_angle_deg=float(np.degrees(np.arctan2(vz,xy))),release_vertical_velocity=vz,
                horizontal_range=float(np.linalg.norm(direction)),
                apex_height_above_release=rise,apex_height_world=float(apex[6]),
                apex_geometry_height_world=float(np.max(flight[:,3])),apex_position=apex[4:7].tolist(),
                apex_time=float(apex[0]),time_to_apex=float(apex[0]-rel['time']),
                ballistic_rise_prediction=float(max(vz,0)**2/(2*g)),
                ballistic_target_error=error,ballistic_com_crossing_prediction=predicted,
                apex_between_release_and_bucket=bool(0<frac<1 and vz>0),
                apex_horizontal_fraction=frac,
                descending_entry_after_apex=bool(result.get('crossing') and result['crossing']['time']>apex[0]+.01
                                                 and np.any(flight[:,16]<-.05) and vz>0))
