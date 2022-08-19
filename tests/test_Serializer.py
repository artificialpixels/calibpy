import unittest
import numpy as np
from pathlib import Path
from calibpy.Serializer import Serializer


class Derivative(Serializer):
    def __init__(self):
        self._a = 0
        self._b = 1.0
        self._e = True
        self._f = "Hello World"
        self._g = [1, "Hello World"]
        self._h = np.array([[1, 2]])


class TestSerializerModule(unittest.TestCase):

    def setUp(self):
        print("start Serializer tests...")
        self._root = Path.cwd() / "tests" / "data"
        self._tmp = []
        self._gt = {
            "a": 0,
            "b": 1.0,
            "e": True,
            "f": "Hello World",
            "g": [1, "Hello World"],
            "h": np.array([[1, 2]])
        }

    def tearDown(self):
        import os
        for f in self._tmp:
            os.remove(f)

    def check_dicts(self, a, b):
        for ka, kb in zip(a.keys(), b.keys()):
            self.assertEqual(ka, kb)
            self.assertEqual(type(a[ka]), type(b[kb]))
            if type(a[ka]) == np.ndarray:
                np.testing.assert_array_equal(a[ka], b[kb])
            else:
                self.assertEqual(a[ka], b[kb])

    def test_functionality(self):
        obj = Derivative()
        data = obj.serialize()
        self.check_dicts(data, self._gt)

        obj1 = Derivative()
        obj2 = Derivative()
        obj3 = Derivative()
        obj4 = Derivative()

        self._tmp.append(self._root / "test_dump.npy")
        obj.serialize(filename=self._tmp[-1])
        obj1.load(filename=self._tmp[-1])
        self._tmp.append(self._root / "test_dump.yaml")
        obj.serialize(filename=self._tmp[-1])
        obj2.load(filename=self._tmp[-1])
        self._tmp.append(self._root / "test_dump.json")
        obj.serialize(filename=self._tmp[-1])
        obj3.load(filename=self._tmp[-1])

        self.check_dicts(obj1.serialize(), self._gt)
        self.check_dicts(obj2.serialize(), self._gt)
        self.check_dicts(obj3.serialize(), self._gt)

        obj4.from_dict(self._gt)
        self.check_dicts(obj4.serialize(), self._gt)
