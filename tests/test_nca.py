import unittest

import numpy as np

from livingpixels.evaluation import damage_sweep, is_recovered
from livingpixels.nca import NCA, NCAConfig, mse_rgba
from livingpixels.targets import living_pixel_target
from livingpixels.training import PoolConfig, fit_pool, train_step
from livingpixels.web import model_payload


class NCATest(unittest.TestCase):
    def setUp(self):
        self.model = NCA(NCAConfig(hidden_channels=8), seed=1)

    def test_shapes_and_identity_initialization(self):
        state = self.model.seed_state(3, 16)
        self.assertEqual(self.model.perceive(state).shape, (3, 48, 16, 16))
        np.testing.assert_allclose(self.model.step(state, training=False), state)

    def test_damage_and_rollout_are_finite(self):
        state = self.model.rollout(self.model.seed_state(size=16), 5, training=True)
        self.assertTrue(np.isfinite(self.model.damage(state, 0.25)).all())

    def test_training_and_pool_smoke(self):
        target = living_pixel_target(16)
        loss, gradients = train_step(self.model, target, steps=3, batch_size=2)
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(all(np.isfinite(value).all() for value in gradients.values()))
        history = fit_pool(self.model, target, iterations=2, steps=2, pool_config=PoolConfig(pool_size=4, batch_size=2), log_every=0)
        self.assertEqual(len(history), 2)

    def test_evaluation_smoke(self):
        target = living_pixel_target(16)
        self.assertFalse(is_recovered(self.model.seed_state(size=16), target))
        results = damage_sweep(self.model, target, fractions=(0.25,), modes=("circle",), trials=2, growth_steps=2, repair_steps=2)
        self.assertEqual(len(results), 1)
        self.assertTrue(0 <= results[0].success_rate <= 1)
        self.assertGreaterEqual(mse_rgba(self.model.seed_state(size=16), target), 0)

    def test_browser_payload_contains_serialized_weights(self):
        payload = model_payload(self.model)
        self.assertEqual(payload["channels"], 16)
        self.assertEqual(len(payload["w1"]), 48 * 8)
        self.assertEqual(len(payload["w2"]), 8 * 16)


if __name__ == "__main__":
    unittest.main()
