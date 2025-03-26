import unittest
from rag_simple import KVModel, Field


class TestKVModel(unittest.TestCase):
    def test_kv_model(self):
        class A(KVModel):
            a1: int = Field(default=3)
            a2: int = Field(default=4)

        class B(KVModel):
            a: A = A.as_field()
            b1 = Field(default="b1")
            b2 = Field(default="b2")

        self.assertIsNot(A.fields, B.fields)
        self.assertIsNot(A.fields, KVModel.fields)
        self.assertDictEqual(KVModel.fields, {})
        b = B()
        a = b.a
        self.assertDictEqual(
            b.dump(), {"a": {"a1": 3, "a2": 4}, "b1": "b1", "b2": "b2"}
        )
        a.load({"a2": 9})
        self.assertEqual(b.a.a2, 9)
        self.assertDictEqual(
            b.dump(), {"a": {"a1": 3, "a2": 9}, "b1": "b1", "b2": "b2"}
        )


if __name__ == "__main__":
    unittest.main()
