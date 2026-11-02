"""CPU-only guards for the standalone 2DGS differential harness.

Run without importing PyTorch:
    python -m unittest discover -s tests -p test_2dgs_backward_regression.py
"""

import contextlib
import copy
import io
import unittest

from benchmarks.blog_eval import gsplat_2dgs_backward_regression as regression


class ReferenceProtocolTests(unittest.TestCase):
    def setUp(self):
        self.configs = regression.select_cases(["occlusion_packed0"])
        self.identity = regression.reference_identity(self.configs, 2026)
        self.reference = dict(
            identity=self.identity,
            tensors={config["name"]: {} for config in self.configs},
        )

    def test_matching_reference(self):
        regression.validate_reference(self.reference, self.identity)

    def test_rejects_changed_workload_seed_or_source(self):
        for field in ("seed", "configs", "fixture_sha256", "protocol"):
            reference = copy.deepcopy(self.reference)
            reference["identity"][field] = None
            with self.subTest(field=field), self.assertRaises(ValueError):
                regression.validate_reference(reference, self.identity)

    def test_rejects_missing_or_extra_cases(self):
        for tensors in ({}, {"unknown": {}}):
            reference = dict(self.reference, tensors=tensors)
            with self.subTest(tensors=tensors), self.assertRaises(ValueError):
                regression.validate_reference(reference, self.identity)

    def test_rejects_unknown_case_instead_of_empty_success(self):
        with self.assertRaises(ValueError):
            regression.select_cases(["does_not_exist"])

    def test_filter_keeps_suite_order_and_configs(self):
        configs = regression.cases()
        names = [configs[-1]["name"], configs[0]["name"]]
        self.assertEqual(regression.select_cases(names), [configs[0], configs[-1]])

    def test_densification_failure_is_not_ignored(self):
        record = dict(tensors={
            "grad/ray_transforms": dict(passed=True),
            "grad/densify": dict(passed=False),
        })
        self.assertFalse(regression.report_passed({"case": record}))

    def test_empty_results_cannot_pass(self):
        self.assertFalse(regression.report_passed({}))
        self.assertFalse(regression.report_passed({"case": dict(tensors={})}))

    def test_all_outputs_must_pass(self):
        record = dict(tensors={"image": dict(passed=True), "grad/normals": dict(passed=True)})
        self.assertTrue(regression.report_passed({"case": record}))
        record["tensors"]["image"]["passed"] = False
        self.assertFalse(regression.report_passed({"case": record}))


class CommandLineTests(unittest.TestCase):
    def assert_invalid(self, *arguments):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            regression.parse_args(["--extension", "binary.so", *arguments])
        self.assertEqual(error.exception.code, 2)

    def test_save_and_compare_are_mutually_exclusive(self):
        self.assert_invalid("--save", "baseline.pt", "--compare", "baseline.pt")

    def test_report_cannot_replace_reference_or_binary(self):
        for arguments in (
            ("--save", "baseline.pt", "--output", "baseline.pt"),
            ("--compare", "baseline.pt", "--output", "baseline.pt"),
            ("--save", "baseline.pt", "--output", "binary.so"),
            ("--save", "binary.so"),
        ):
            with self.subTest(arguments=arguments):
                self.assert_invalid(*arguments)

    def test_invalid_timing_counts_are_rejected(self):
        self.assert_invalid("--save", "baseline.pt", "--warmup", "-1")
        self.assert_invalid("--save", "baseline.pt", "--repeats", "-1")


if __name__ == "__main__":
    unittest.main()
