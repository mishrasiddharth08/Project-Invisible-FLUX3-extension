import unittest
from unittest.mock import patch

import torch

from flux3_forge.spectrum import SpectrumConfig, SpectrumRejected, SpectrumState


class SpectrumTests(unittest.TestCase):
    def test_default_is_off(self):
        state = SpectrumState(SpectrumConfig.from_request(None))
        self.assertFalse(state.active)
        self.assertTrue(state.should_compute(4, 16, 0.5))

    def test_chebyshev_forecasts_feature_residual_at_descending_target_time(self):
        config = SpectrumConfig(enabled=True, order=2, history_points=4, ridge=1e-6, blend=1.0)
        state = SpectrumState(config)
        scale = torch.tensor([1.0, 2.0, -0.5])
        formula = lambda t: scale * (2.0 + 3.0 * t + 4.0 * t * t)
        for time in (1.0, 0.8, 0.6, 0.4):
            state.observe(time, formula(time))
        actual = state.predict(0.2).float()
        torch.testing.assert_close(actual, formula(0.2), rtol=0.03, atol=0.03)

    def test_local_taylor_is_signed_for_descending_flow_map_times(self):
        config = SpectrumConfig(enabled=True, order=1, history_points=3, ridge=0.1, blend=0.0)
        state = SpectrumState(config)
        for time in (1.0, 0.7, 0.4):
            state.observe(time, torch.tensor([5.0 + 2.0 * time]))
        torch.testing.assert_close(state.predict(0.1).float(), torch.tensor([5.2]), rtol=0.02, atol=0.02)

    def test_scheduler_allows_only_one_consecutive_forecast_and_exact_final(self):
        state = SpectrumState(SpectrumConfig(enabled=True, order=1, history_points=3))
        for time in (1.0, 0.8, 0.6):
            state.observe(time, torch.ones(2))
        self.assertFalse(state.should_compute(4, 10, 0.4))
        self.assertTrue(state.should_compute(5, 10, 0.2))
        state.observe(0.2, torch.ones(2))
        self.assertTrue(state.should_compute(9, 10, 0.0))

    def test_large_feature_disables_cleanly_before_unbounded_history(self):
        state = SpectrumState(SpectrumConfig(enabled=True, order=1, history_points=3, max_cache_mb=16))
        state.observe(1.0, torch.zeros(5_000_000, dtype=torch.bfloat16))
        self.assertFalse(state.active)
        self.assertIn("minimum history", state.disabled_reason)
        self.assertEqual(state.cache_bytes, 0)

    def test_release_resets_request_state(self):
        state = SpectrumState(SpectrumConfig(enabled=True, order=1, history_points=3))
        state.observe(1.0, torch.ones(4))
        state.release()
        self.assertEqual(state.times, [])
        self.assertEqual(state.features, [])
        self.assertEqual(state.cache_bytes, 0)

    def test_rejects_bad_numeric_configuration(self):
        bad = (
            {"ridge": -1.0}, {"ridge": float("nan")}, {"ridge": "0.1"},
            {"max_horizon_ratio": 0.0}, {"taylor_order": 2}, {"warmup_steps": 1.5},
        )
        for options in bad:
            with self.subTest(options=options), self.assertRaises(ValueError):
                SpectrumConfig.from_request({"enabled": True, **options})

    def test_spectral_system_is_solved_on_cpu(self):
        state = SpectrumState(SpectrumConfig(enabled=True, order=1, history_points=3))
        for time in (1.0, 0.8, 0.6):
            state.observe(time, torch.ones(2))
        original = torch.linalg.solve
        seen = []

        def capture(a, b):
            seen.append((a.device.type, b.device.type))
            return original(a, b)

        with patch("torch.linalg.solve", side_effect=capture):
            state._spectral_weights(0.4, torch.device("cpu"))
        self.assertEqual(seen, [("cpu", "cpu")])

    def test_budget_is_checked_before_residual_clone(self):
        class Oversized:
            shape = (10_000_000,)

            def numel(self):
                return 10_000_000

            def detach(self):
                raise AssertionError("must not allocate")

        state = SpectrumState(SpectrumConfig(enabled=True, order=1, history_points=3, max_cache_mb=16))
        state.observe(1.0, Oversized())
        self.assertFalse(state.active)

    def test_unstable_forecast_disables_and_requests_real_stack(self):
        state = SpectrumState(SpectrumConfig(enabled=True, order=1, history_points=3, blend=0.0))
        for time, value in ((1.0, 100.0), (0.8, 100.0), (0.6, 0.0)):
            state.observe(time, torch.tensor([value]))
        with self.assertRaises(SpectrumRejected):
            state.predict(0.4)
        self.assertFalse(state.active)
        self.assertIn("relative feature change", state.disabled_reason)


if __name__ == "__main__":
    unittest.main()
