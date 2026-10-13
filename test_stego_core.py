import struct
import unittest
import zlib
import os
import tempfile
from PIL import Image
import numpy as np

import stego_core as sc


class StegoCoreTests(unittest.TestCase):
    def setUp(self):
        self.rgb = sc.synthetic_cover(64, seed=17)
        self.gray = sc.to_gray(self.rgb)
        self.payload = b"round-trip test payload"

    def test_all_embedding_methods_round_trip_for_rgb_and_gray(self):
        for cover in (self.rgb, self.gray):
            for method in ("seq", "match", "scatter", "lsb2", "edge", "dip"):
                with self.subTest(shape=cover.shape, method=method):
                    stego, _ = sc.embed(cover, self.payload, sc.KIND_TEXT, method, rng_seed=123)
                    results = sc.extract_all(stego)
                    self.assertTrue(any(r["crc_ok"] and r["data"] == self.payload for r in results))

    def test_dip_order_is_unchanged_by_embedding(self):
        order_before, _ = sc.dip_order(self.rgb)
        stego, _ = sc.embed(self.rgb, self.payload, sc.KIND_TEXT, "dip")
        order_after, _ = sc.dip_order(stego)
        np.testing.assert_array_equal(order_before, order_after)

    def test_each_dip_ablation_rebuilds_order_and_extracts(self):
        variants = (("variance",), ("sobel",), ("laplacian",), sc.DIP_FEATURES)
        for features in variants:
            with self.subTest(features=features):
                order_before, _ = sc.dip_order(self.rgb, features)
                stego, _ = sc.embed(self.rgb, self.payload, sc.KIND_TEXT, "dip", dip_components=features)
                order_after, _ = sc.dip_order(stego, features)
                np.testing.assert_array_equal(order_before, order_after)
                results = sc.extract_all(stego, dip_components=features)
                self.assertTrue(any(r["tech"] == "dip" and r["crc_ok"] and r["data"] == self.payload
                                    for r in results))

    def test_16bit_grayscale_is_rescaled_not_clipped(self):
        ramp = np.linspace(0, 65535, 64 * 64).reshape(64, 64).astype(np.uint16)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "ramp16.png")
            Image.fromarray(ramp).save(path)
            loaded = sc.load_image(path)
        self.assertEqual(loaded.dtype, np.uint8)
        self.assertEqual(loaded.min(), 0)
        self.assertEqual(loaded.max(), 255)
        self.assertGreater(len(np.unique(loaded)), 200)
        self.assertLess((loaded == 255).mean(), 0.05)

    def test_capacity_boundary(self):
        cover = np.zeros((16, 16), dtype=np.uint8)
        capacity = sc.capacity_bytes(cover, "seq")
        exact, _ = sc.embed(cover, b"x" * capacity, sc.KIND_TEXT, "seq")
        self.assertTrue(any(r["crc_ok"] and r["data"] == b"x" * capacity for r in sc.extract_all(exact)))
        with self.assertRaises(ValueError):
            sc.embed(cover, b"x" * (capacity + 1), sc.KIND_TEXT, "seq")

    def test_matching_is_reproducible_with_seed(self):
        first, _ = sc.embed(self.rgb, self.payload, sc.KIND_TEXT, "match", rng_seed=9)
        repeated, _ = sc.embed(self.rgb, self.payload, sc.KIND_TEXT, "match", rng_seed=9)
        other_seed, _ = sc.embed(self.rgb, self.payload, sc.KIND_TEXT, "match", rng_seed=10)
        np.testing.assert_array_equal(first, repeated)
        self.assertFalse(np.array_equal(first, other_seed))

    def test_detector_evaluation_is_reproducible_with_seed(self):
        cover = sc.synthetic_cover(256, seed=5)
        first, _ = sc.detector_accuracy(cover, ["match"], [0.1], size=64, max_crops=100, seed=42)
        repeated, _ = sc.detector_accuracy(cover, ["match"], [0.1], size=64, max_crops=100, seed=42)
        self.assertEqual(first, repeated)

    def test_detector_evaluates_named_dip_ablation_variants(self):
        cover = sc.synthetic_cover(256, seed=6)
        variants = (
            ("sequential", "seq", sc.DIP_FEATURES),
            ("variance", "dip", ("variance",)),
            ("combined", "dip", sc.DIP_FEATURES),
        )
        results, _ = sc.detector_accuracy(cover, variants, [0.05], size=64, max_crops=100, seed=13)
        self.assertIsNotNone(results)
        self.assertEqual(set(results), {"sequential", "variance", "combined"})
        self.assertTrue(all(0 <= scores[0] <= 1 for scores in results.values()))

    def test_password_payload_authenticates_and_rejects_wrong_password(self):
        stego, _ = sc.embed(self.rgb, self.payload, sc.KIND_TEXT, "seq", "correct horse")
        valid = sc.extract_all(stego, "correct horse")
        self.assertTrue(any(r["encrypted"] and r["crc_ok"] and r["data"] == self.payload for r in valid))
        wrong = sc.extract_all(stego, "wrong password")
        self.assertTrue(any(r["encrypted"] and not r["crc_ok"] for r in wrong))

    def test_legacy_xor_payload_remains_readable(self):
        password = "legacy password"
        crc = zlib.crc32(self.payload) & 0xFFFFFFFF
        header = sc.MAGIC + bytes([sc.KIND_TEXT, sc.ENC_LEGACY_XOR])
        header += struct.pack(">II", len(self.payload), crc)
        payload = header + sc.xor_crypt(self.payload, password)
        bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
        flat = self.rgb.flatten().copy()
        flat[:len(bits)] = (flat[:len(bits)] & 0xFE) | bits
        legacy_stego = flat.reshape(self.rgb.shape)
        results = sc.extract_all(legacy_stego, password)
        self.assertTrue(any(r["crc_ok"] and r["data"] == self.payload for r in results))


if __name__ == "__main__":
    unittest.main()
