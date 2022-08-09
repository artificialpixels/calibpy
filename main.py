from pathlib import Path

from calibpy.Camera import Camera
from calibpy.Stream import Stream


if __name__ == "__main__":
    stream = Stream(Path.cwd() / "tests" / "data" /
                    "single_cam" / "undistorted")
    cam = Camera()
    cam.stream = stream
    print("Camera has Stream: ", cam.has_stream(), "of length:", stream.length)
