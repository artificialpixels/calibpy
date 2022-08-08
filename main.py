from calibpy.Camera import Camera


if __name__ == "__main__":
    cam = Camera()
    cam.quick_init()
    cam.serialize("C:\\Users\\svenw\\OneDrive\\Desktop\\test.npy")
    cam2 = Camera()
    cam2.load("C:\\Users\\svenw\\OneDrive\\Desktop\\test.npy")
    a = 0
