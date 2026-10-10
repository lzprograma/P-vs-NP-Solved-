import random
import unittest

from egd_qec import (
    accumulated_potential_bound,
    count_survivors,
    exact_phi,
    qec_causal_halving,
    qec_protect_estimate,
    repetition_decode,
    repetition_encode_bits,
    required_repetition_length,
    theorem_condition,
    transmit_bsc,
)


class TestEGDQEC(unittest.TestCase):
    def test_repetition_corrects_up_to_t_bit_flips(self):
        q = 5
        original = [1, 0, 1]
        blocks = repetition_encode_bits(original, q)
        # Corrupt no more than floor((q-1)/2)=2 bits in each block.
        received = [
            [0, 1, 1, 1, 1],  # originally 1, one flip
            [1, 0, 0, 0, 0],  # originally 0, one flip
            [1, 1, 1, 0, 0],  # originally 1, two flips
        ]
        decoded, _ = repetition_decode(received)
        self.assertEqual(decoded, original)

    def test_roundtrip_without_noise(self):
        decoded, syndrome_count, failed = qec_protect_estimate(
            0.375, scale=1000, width=10, q=5, p=0.0,
            rng=random.Random(4))
        self.assertAlmostEqual(decoded, 0.375)
        self.assertEqual(syndrome_count, 0)
        self.assertFalse(failed)

    def test_finite_halving_survives_with_high_precision(self):
        family = ["0000", "1111", "0011", "0101"]
        prefix, states = qec_causal_halving(
            family, 3, scale=10000, q=5, p=0.0, seed=7)
        self.assertEqual(count_survivors(family, prefix), 0)
        self.assertLess(states[0].phi_exact + 3 * (1 / 20000), 1)

    def test_potential_identity(self):
        family = ["0000", "1111", "0011", "0101"]
        u = "0"
        lhs = exact_phi(family, u, 3)
        rhs = (exact_phi(family, u + "0", 3) + exact_phi(family, u + "1", 3)) / 2
        self.assertAlmostEqual(lhs, rhs)

    def test_error_budget_condition(self):
        self.assertTrue(theorem_condition(0.25, [0.1, 0.1, 0.1]))
        self.assertFalse(theorem_condition(0.25, [0.4, 0.4]))
        self.assertAlmostEqual(accumulated_potential_bound(0.25, [0.1, 0.1]), 0.45)

    def test_repetition_length(self):
        q = required_repetition_length(100, 10, 0.01, 1e-6)
        self.assertGreaterEqual(q, 1)
        self.assertEqual(q % 2, 1)

    def test_random_channel_is_reproducible_with_seed(self):
        source1 = random.Random(99)
        source2 = random.Random(99)
        blocks = repetition_encode_bits([0, 1, 1, 0], 7)
        self.assertEqual(transmit_bsc(blocks, 0.08, source1),
                         transmit_bsc(blocks, 0.08, source2))

    def test_bad_noise_parameter_rejected(self):
        with self.assertRaises(ValueError):
            required_repetition_length(10, 3, 0.5, 1e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
