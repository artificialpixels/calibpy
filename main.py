#!/usr/bin/env python

"""
:Author: Sven Wanner (artificial.pixels@gmail.com)
:Sponsor: SpexAI GmbH
"""


import argparse
from calibpy.Settings import Settings
from calibpy.single_cam_workflow import single_cam_workflow, show_pcl_set


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input',
                    type=str,
                    default="tests/data/demo_project_settings.yaml",
                    help="Project settings file")
parser.add_argument('-w', '--workflow',
                    type=str,
                    default='single_cam_workflow',
                    help="Workflow name ['single_cam_workflow']")


if __name__ == "__main__":
    args = parser.parse_args()

    ps = Settings()
    ps.from_config(args.input)

    if args.workflow == "single_cam_workflow":
        intr, extrs, pcls = single_cam_workflow(
            project_dir=ps.project_dir,
            project_name=ps.project_name,
            intrinsic_calibration_input_dir=ps.intrinsic_calibration_input_dir,
            calibration_config_file=ps.calibration_config_file,
            extrinsic_calibration_input=ps.extrinsic_calibration_input,
            depth_registration_input=ps.depth_registration_input,
            color_registration_input=ps.color_registration_input,
            blender_conform=ps.blender_conform,
            register_from_frame=ps.register_from_frame,
            register_to_frame=ps.register_to_frame,
            lazy_intrinsics=ps.lazy_intrinsics,
            visualize=ps.visualize)

        print("#"*30)
        print("#\tSINGLE CAM WORKFLOW")
        print("\nIntrinsics:")
        print(intr)
        print("\nExtrinsics:")
        for n, cam in enumerate(extrs):
            print("-"*20)
            print(f"frame: {n:06d}")
            print(cam)

        if ps.visualize:
            show_pcl_set(pcls)
