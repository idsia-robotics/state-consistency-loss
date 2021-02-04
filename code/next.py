import numpy as np
import pandas as pd
from scipy import stats
from utils import mktransf
from settings import robot_geometry


def get_map(rel_transform, sensor_readings, robot_geometry, coords, delta):
    '''Given a pose, constructs the occupancy map w.r.t. that pose.
    An occupancy map has 3 possible values:
    1 = object present;
    0 = object not present;
    -1 = unknown;

    Args:
            rel_transform:  the tranformation matrix from which to compute the occupancy map.
            sensor_readings:  a list of sensors' readings.
            robot_geometry: the transformations from robot frame to sensors' frames.
            coords: a list of relative coordinates of the form [(x1, y1), ...].
            delta: the maximum distance between a sensor reading and a coord to be matched.

    Returns:
            an occupancy map generated from the relative pose using coords and sensors' readings.
    '''
    # locate objects based on the distances read by the sensors
    sensor_readings_homo = np.array([[r, 0., 1.] for r in sensor_readings])

    # batch matrix multiplication
    rel_object_poses = np.einsum('ijk,ik->ij',
                                 np.matmul(rel_transform,
                                           robot_geometry),
                                 sensor_readings_homo)[:, :-1]

    # initialize occupancy map to -1
    occupancy_map = np.full((coords.shape[0],), -1, dtype=np.float)

    # compute distances between object poses and coords
    distances = np.linalg.norm(
        coords[:, None, :] - rel_object_poses[None, :, :],
        axis=-1)

    # find all coords with distance <= delta
    closer_than_delta = distances <= delta
    icoords, isens = np.where(closer_than_delta)
    # note: 0.11 is the maximum distance of a detected obstacle
    occupancy_map[icoords] = sensor_readings[isens] < 0.11

    return occupancy_map


def aggregate(maps):
    """Aggregates a list of occupancy maps into a single one.

    Args:
            maps: a list of occupancy maps.

    Returns:
            an aggregate occupancy map.
    """
    aggregated_map = np.full_like(maps[0], -1)

    if (maps == -1).all():
        return aggregated_map

    map_mask = (maps != -1).any(1)
    nonempty_maps = maps[map_mask]
    cell_mask = (nonempty_maps != -1).any(0)
    nonempty_cells = nonempty_maps[:, cell_mask]
    nonempty_cells[nonempty_cells == -1] = np.nan
    aggregated_map[cell_mask] = stats.mode(
        nonempty_cells, 0, nan_policy='omit')[0][0]

    return aggregated_map


def compute_occupancy_map(row, df, coords, target_columns, interval, delta):
    '''Given a pose, constructs the occupancy map w.r.t. that pose.
    An occupancy map has 3 possible values:
    1 = object present;
    0 = object not present;
    -1 = unknown;

    Args:
            row: a dataframe row containing the pose from which to compute the occupancy map.
            df: the dataframe from which to extract data.
            coords: a list of relative coordinates of the form [(x1, y1), ...].
            target_columns:  a list of columns containing sensors' readings.
            interval: a time interval, expressed as string, for limiting the computation time.
            delta: the maximum distance between a sensor reading and a coord to be matched.

    Returns:
            a series composed of {target_column1: occupancy_map1, ...}.
    '''
    idx = row.name

    # consider only poses within a time interval from the reference pose
    window = df.loc[idx - pd.Timedelta(interval):idx + pd.Timedelta(interval)]

    # compute relative transformation
    other_poses = window[['pos_x', 'pos_y', 'theta']].values
    other_transforms = np.stack([mktransf(pose) for pose in other_poses])
    inverse_frame = np.linalg.inv(
        mktransf(row[['pos_x', 'pos_y', 'theta']].values))
    rel_transforms = np.matmul(inverse_frame, other_transforms)

    sensor_readings = window[target_columns].values

    # compute occupancy maps (one map per relative transformation)
    maps = [get_map(trans, reading, robot_geometry, coords, delta)
            for trans, reading in zip(rel_transforms, sensor_readings)]

    # aggregate occupancy maps
    occupancy_map = aggregate(np.array(maps))

    return pd.Series({'target_map': occupancy_map})


def compute_occupancy_maps(df, coords, target_columns, interval='1s', delta=0.01):
    '''Creates occupancy map based on sensors' readings and odometry sored in a dataframe.

    Args:
            df: the dataframe in which to find the data to elaborate.
            coords: a list of relative coordinates of the form [(x1, y1), ...].
            target_columns: a list of columns containing sensors' readings.
            interval: a time interval, expressed as string, for limiting the computation time.
            delta: the maximum distance between a sensor reading and a coord to be matched.

    Returns:
            The new dataframe with the added columns corresponding to an occupancy map.
    '''
    next_df = df.apply(compute_occupancy_map, axis=1,
                       args=(df, coords, target_columns, interval, delta))

    output_cols = next_df.columns.values.tolist()
    df = pd.concat([df.drop(target_columns, axis=1), next_df], axis=1)

    return df, output_cols
