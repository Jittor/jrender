import math
import jittor as jt

def get_points_from_angles(distance, elevation, azimuth, degrees=True):
    if isinstance(distance, float) or isinstance(distance, int):
        if degrees:
            elevation = math.radians(elevation)
            azimuth = math.radians(azimuth)
        return (
            distance * math.cos(elevation) * math.sin(azimuth),
            distance * math.sin(elevation),
            -distance * math.cos(elevation) * math.cos(azimuth))
    else:
        if degrees:
            elevation = math.pi/180. * elevation
            azimuth = math.pi/180. * azimuth
    #
        return jt.contrib.concat([
            (distance * jt.cos(elevation) * jt.sin(azimuth)).unsqueeze(0),
            (distance * jt.sin(elevation)).unsqueeze(0),
            (-distance * jt.cos(elevation) * jt.cos(azimuth)).unsqueeze(0)
            ], dim=0).transpose(1,0)