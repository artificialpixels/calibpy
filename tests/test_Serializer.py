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


class TestSerializerModule(unittest.TestCase):

    def setUp(self):
        print("start Serializer tests...")
        self._root = Path.cwd() / "tests" / "data"
        self._tmp = []
        self._gt = {
            "a": 0,
            "b": 1.0,
            "e": True,
            "f": "Hello World"
        }

    def tearDown(self):
        import os
        for f in self._tmp:
            os.remove(f)

    def test_functionality(self):
        obj = Derivative()
        data = obj.serialize()
        self.assertDictEqual(data, self._gt)

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

        self.assertDictEqual(obj1.serialize(), self._gt)
        self.assertDictEqual(obj2.serialize(), self._gt)
        self.assertDictEqual(obj3.serialize(), self._gt)

        obj4.from_dict(self._gt)
        self.assertDictEqual(obj4.serialize(), self._gt)
