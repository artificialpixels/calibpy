import cv2
import unittest
import numpy as np
from pathlib import Path
from calibpy.Stream import FileStream


class TestStreamModule(unittest.TestCase):

    def setUp(self):
        self._root = Path.cwd() / "tests" / "data"

    def test_load_filename(self):
        filename = "D:\\Projects\\Python\\calibpy\\tests\\data\\single_cam\\undistorted\\0003.png"
        fs = FileStream()
        fs.initialize(filename=filename)
        img = fs.next(flag=cv2.IMREAD_GRAYSCALE)
        img_ref = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        self.assertAlmostEqual(np.sum(img-img_ref), 0)
        self.assertTrue(fs.next() is None)
        fs.reset()
        img = fs.get(0, flag=cv2.IMREAD_GRAYSCALE)
        img_ref = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        self.assertAlmostEqual(np.sum(img-img_ref), 0)
        fname = fs.current_filename()
        self.assertEqual(fname, filename)

    def test_load_from_list(self):
        fs = FileStream()
        filenames = [
            str(self._root / "single_cam" / "undistorted" / "0001.png"),
            str(self._root / "single_cam" / "undistorted" / "0002.png"),
            str(self._root / "single_cam" / "undistorted" / "0003.png")]
        fs.initialize(filenames=filenames)
        for i in range(3):
            img = fs.next(flag=cv2.IMREAD_GRAYSCALE)
            img_ref = cv2.imread(filenames[i], cv2.IMREAD_GRAYSCALE)
            self.assertAlmostEqual(np.sum(img-img_ref), 0)
            fname = fs.current_filename()
            self.assertEqual(fname, filenames[i])

    def test_load_from_dir(self):
        filenames = []
        for i in range(1, 25):
            filenames.append(
                str(self._root / "single_cam" / "undistorted" / f"{i:04d}.png"))
        fs = FileStream()
        fs.initialize(
            directory=str(self._root / "single_cam" / "undistorted"))

        for i in range(24):
            img = fs.next(flag=cv2.IMREAD_GRAYSCALE)
            img_ref = cv2.imread(filenames[i], cv2.IMREAD_GRAYSCALE)
            self.assertAlmostEqual(np.sum(img-img_ref), 0)
            fname = fs.current_filename()
            self.assertEqual(fname, filenames[i])

        for i in range(24):
            img = fs.get(i, flag=cv2.IMREAD_GRAYSCALE)
            img_ref = cv2.imread(filenames[i], cv2.IMREAD_GRAYSCALE)
            self.assertAlmostEqual(np.sum(img-img_ref), 0)

    def test_load_subsets_from_dir(self):
        filenames = []
        for i in range(10, 20):
            filenames.append(
                str(self._root / "single_cam" / "undistorted" / f"{i+1:04d}.png"))
        fs = FileStream()
        fs.initialize(
            directory=str(self._root / "single_cam" / "undistorted"),
            from_frame=10,
            to_frame=20)
        for i in range(10):
            img = fs.next(flag=cv2.IMREAD_GRAYSCALE)
            img_ref = cv2.imread(filenames[i], cv2.IMREAD_GRAYSCALE)
            self.assertAlmostEqual(np.sum(img-img_ref), 0)
            fname = fs.current_filename()
            self.assertEqual(fname, filenames[i])

    def test_load_pre_suffixes_from_dir(self):
        pass

    def test_looping(self):
        fs = FileStream(is_looping=False)
        self.assertFalse(fs.is_looping)
        filenames = [
            str(self._root / "single_cam" / "undistorted" / "0001.png"),
            str(self._root / "single_cam" / "undistorted" / "0002.png"),
            str(self._root / "single_cam" / "undistorted" / "0003.png")]
        fs.initialize(filenames=filenames)
        for n in range(3):
            _ = fs.next()
            self.assertEqual(fs.current_filename(), filenames[n])
        self.assertTrue(fs.next() is None)
        fs.reset()
        fs.is_looping = True
        self.assertTrue(fs.is_looping)
        for n in range(3):
            _ = fs.next()
            self.assertEqual(fs.current_filename(), filenames[n])
        img = fs.next()
        self.assertTrue(isinstance(img, np.ndarray))
        self.assertEqual(fs.current_filename(), filenames[0])

    # def test_loading_samepatterns(self):
    #     stream = Stream()
    #     stream.load(dir=str(self._root / "dummy_images" / "same_pattern"))
    #     self.assertEqual(stream.length, 3)
    #     img = stream.next()
    #     self.assertTrue(isinstance(img, np.ndarray))
    #     self.assertEqual(img.shape, (2, 2))
    #     fname = Path(stream.current_filename()).name
    #     self.assertEqual(fname, "0001.png")
    #     img = stream.next()
    #     self.assertTrue(isinstance(img, np.ndarray))
    #     fname = Path(stream.current_filename()).name
    #     self.assertEqual(fname, "0002.png")

    # def test_loading_prefixes(self):
    #     stream = Stream()
    #     stream.load(
    #         dir=str(self._root / "dummy_images" / "prefixes"), prefix="p1")
    #     self.assertEqual(stream.length, 2)
    #     img = stream.next()
    #     self.assertTrue(isinstance(img, np.ndarray))
    #     self.assertEqual(img.shape, (2, 2))
    #     fname = Path(stream.current_filename()).name
    #     self.assertEqual(fname, "p1_0001.png")
    #     img = stream.next()
    #     self.assertTrue(isinstance(img, np.ndarray))
    #     fname = Path(stream.current_filename()).name
    #     self.assertEqual(fname, "p1_0004.png")

    # def test_loading_suffixes(self):
    #     stream = Stream()
    #     stream.load(
    #         dir=str(self._root / "dummy_images" / "suffixes"), suffix="s1")
    #     self.assertEqual(stream.length, 2)
    #     img = stream.next()
    #     self.assertTrue(isinstance(img, np.ndarray))
    #     self.assertEqual(img.shape, (2, 2))
    #     fname = Path(stream.current_filename()).name
    #     self.assertEqual(fname, "0001_s1.png")
    #     img = stream.next()
    #     self.assertTrue(isinstance(img, np.ndarray))
    #     fname = Path(stream.current_filename()).name
    #     self.assertEqual(fname, "0004_s1.png")

    # def test_loading_pre_and_suffixes(self):
    #     stream = Stream()
    #     stream.load(
    #         dir=str(self._root / "dummy_images" / "pre_and_suffixes"),
    #         prefix="p1",
    #         suffix="s1")
    #     self.assertEqual(stream.length, 2)
    #     img = stream.next()
    #     self.assertTrue(isinstance(img, np.ndarray))
    #     self.assertEqual(img.shape, (2, 2))
    #     fname = Path(stream.current_filename()).name
    #     self.assertEqual(fname, "p1_0001_s1.png")
    #     img = stream.next()
    #     self.assertTrue(isinstance(img, np.ndarray))
    #     fname = Path(stream.current_filename()).name
    #     self.assertEqual(fname, "p1_0004_s1.png")
