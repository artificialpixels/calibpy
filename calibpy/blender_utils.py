import os
import bpy
import pickle
import numpy as np
from glob import glob
from pathlib import Path
from mathutils import *

CAMERAS_DIR = "C:\\Users\\svenw\\OneDrive\\Desktop\\results"


def read_npys_from_dir(dirname: str) -> list:
    cameras = []
    filenames = []
    for f in glob(CAMERAS_DIR + os.sep + "*.npy"):
        filenames.append(f)
    filenames.sort()
    return filenames


def load_camera_props_from_file(filename: str):
    assert isinstance(filename, str)
    assert Path(filename).exists()
    with open(filename, 'rb') as file:
        return pickle.load(file)
    return None


def find_latest_object_by_name(context, name):
    bpy.ops.object.select_all(action='DESELECT')
    objs = [x for x in context.scene.objects.keys() if x.startswith(name)]
    objs.sort()
    if len(objs) <= 0:
        return None
    obj = context.scene.objects[objs[-1]]
    obj.select_set(True)
    return obj


def delete_object(context, obj):
    if isinstance(obj, str):
        if obj in context.scene.objects.keys():
            bpy.data.objects.remove(context.scene.objects[obj])
    elif isinstance(obj, bpy.types.Object):
        bpy.data.objects.remove(obj)


def matrix_from_numpy(array: np.ndarray) -> Matrix:
    return Matrix(array.tolist())


def create_camera(context: object, props: dict):
    cam_name = "CalibpyCam"
    if props["name"] is not None and props["name"] != "":
        cam_name = props["name"]
    if cam_name in bpy.data.objects.keys():
        delete_object(context, bpy.data.objects[cam_name])

    bpy.ops.object.camera_add(
        enter_editmode=False,
        align='VIEW',
        location=(0, 0, 0),
        rotation=(0, 0, 0),
        scale=(1, 1, 1))

    cam = find_latest_object_by_name(context, "Camera")
    cam.name = cam_name
    cam.data.sensor_height = props['sensor_size'][0]
    cam.data.sensor_width = props['sensor_size'][1]
    cam.data.lens_unit = 'MILLIMETERS'
    cam.data.lens = props['f_mm']
    cam.matrix_world = matrix_from_numpy(props['RTb'])


if __name__ == "__main__":
    context = bpy.context
    filenames = read_npys_from_dir(CAMERAS_DIR)
    props = load_camera_props_from_file(filenames[0])
    create_camera(context, props)
    props = load_camera_props_from_file(filenames[1])
    create_camera(context, props)
    props = load_camera_props_from_file(filenames[2])
    create_camera(context, props)
