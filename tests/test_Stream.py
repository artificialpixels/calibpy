import unittest
import numpy as np
from pathlib import Path
from calibpy.Stream import Stream


class TestStreamModule(unittest.TestCase):

    def setUp(self):
        self._root = Path.cwd() / "tests" / "data"

    def test_loading_samepatterns(self):
        stream = Stream()
        stream.load(dir=str(self._root / "dummy_images" / "same_pattern"))
        self.assertEqual(stream.length, 3)
        img = stream.next()
        self.assertTrue(isinstance(img, np.ndarray))
        self.assertEqual(img.shape, (2, 2))
        fname = Path(stream.current_filename()).name
        self.assertEqual(fname, "0001.png")
        img = stream.next()
        self.assertTrue(isinstance(img, np.ndarray))
        fname = Path(stream.current_filename()).name
        self.assertEqual(fname, "0002.png")

    def test_loading_prefixes(self):
        stream = Stream()
        stream.load(
            dir=str(self._root / "dummy_images" / "prefixes"), prefix="p1")
        self.assertEqual(stream.length, 2)
        img = stream.next()
        self.assertTrue(isinstance(img, np.ndarray))
        self.assertEqual(img.shape, (2, 2))
        fname = Path(stream.current_filename()).name
        self.assertEqual(fname, "p1_0001.png")
        img = stream.next()
        self.assertTrue(isinstance(img, np.ndarray))
        fname = Path(stream.current_filename()).name
        self.assertEqual(fname, "p1_0004.png")

    def test_loading_suffixes(self):
        stream = Stream()
        stream.load(
            dir=str(self._root / "dummy_images" / "suffixes"), suffix="s1")
        self.assertEqual(stream.length, 2)
        img = stream.next()
        self.assertTrue(isinstance(img, np.ndarray))
        self.assertEqual(img.shape, (2, 2))
        fname = Path(stream.current_filename()).name
        self.assertEqual(fname, "0001_s1.png")
        img = stream.next()
        self.assertTrue(isinstance(img, np.ndarray))
        fname = Path(stream.current_filename()).name
        self.assertEqual(fname, "0004_s1.png")

    def test_loading_pre_and_suffixes(self):
        stream = Stream()
        stream.load(
            dir=str(self._root / "dummy_images" / "pre_and_suffixes"),
            prefix="p1",
            suffix="s1")
        self.assertEqual(stream.length, 2)
        img = stream.next()
        self.assertTrue(isinstance(img, np.ndarray))
        self.assertEqual(img.shape, (2, 2))
        fname = Path(stream.current_filename()).name
        self.assertEqual(fname, "p1_0001_s1.png")
        img = stream.next()
        self.assertTrue(isinstance(img, np.ndarray))
        fname = Path(stream.current_filename()).name
        self.assertEqual(fname, "p1_0004_s1.png")
