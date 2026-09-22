#!/usr/bin/env python3
"""Self-check for layer-build. Run it with:

    python3 layer-build/tests/test-layer-build.py

It exercises the pure logic (parsing, version comparison, classification) and the one effectful
function that writes into a user's file. Everything that needs singularity is out of reach here and
is not faked: a fake would test the fake.
"""


import contextlib
import importlib.machinery
import importlib.util
import io
import json
import pathlib
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
TOOL = HERE.parent / "layer-build"
REPO = HERE.parent.parent

loader = importlib.machinery.SourceFileLoader("layer_build", str(TOOL))
spec = importlib.util.spec_from_loader("layer_build", loader)
lb = importlib.util.module_from_spec(spec)
loader.exec_module(lb)


class TestNames(unittest.TestCase):
    def test_normalize_name(self):
        for raw, want in [
            ("Torch", "torch"),
            ("scikit_learn", "scikit-learn"),
            ("scikit.learn", "scikit-learn"),
            ("zope.interface", "zope-interface"),
            ("A__B--C", "a-b-c"),
        ]:
            self.assertEqual(lb.normalize_name(raw), want, raw)


class TestVersions(unittest.TestCase):
    def test_release_ordering(self):
        self.assertLess(lb.version_key("2.0rc1"), lb.version_key("2.0"))
        self.assertLess(lb.version_key("2.0"), lb.version_key("2.0.1"))
        self.assertLess(lb.version_key("2.9.0"), lb.version_key("2.10.0"))
        self.assertLess(lb.version_key("1.0.dev1"), lb.version_key("1.0a1"))
        self.assertLess(lb.version_key("1.0"), lb.version_key("1.0.post1"))

    def test_local_label_is_compared_only_when_the_specifier_names_one(self):
        # A specifier without a local label matches the image's ROCm build (`2.10.0+rocm7.0`),
        # which is what a LUMI user means; one that names a label must match it.
        self.assertEqual(
            lb.compare_versions(
                lb.version_key("2.10.0+rocm7.0"), lb.version_key("2.10.0")
            ),
            0,
        )
        self.assertTrue(lb.satisfies("2.10.0+rocm7.0", "==2.10.0"))
        self.assertFalse(lb.satisfies("1.0+rocm", "==1.0+cpu"))

    def test_satisfies(self):
        cases = [
            ("2.10.0+rocm7.0", "==2.10.0", True),
            ("2.3.5", ">=2,<3", True),
            ("1.9.0", ">=2,<3", False),
            ("0.13.2", ">=0.13", True),
            ("1.8.0", "==1.8.0", True),
            ("1.8.1", "==1.8.0", False),
            ("1.8.1", "!=1.8.0", True),
            ("2.3.5", "2.*", True),
            ("3.0.0", "2.*", False),
            ("1.4.2", "~=1.4", True),
            ("2.0.0", "~=1.4", False),
            ("anything", "", True),
            ("anything", "*", True),
            ("1.0", "==1.0.0", True),  # trailing zeros are the same release
            ("1.0rc1", "==1.0rc2", False),  # the pre-release serial is compared
            ("1.2.3", "!=1.2.*", False),
            ("1.3.0", "!=1.2.*", True),
            ("1.4.5", "~=1.4.0", True),  # compatible release keeps the requested precision
            ("1.9.0", "~=1.4.0", False),
            ("1.9.0", "~=1.4", True),
            ("1.0+rocm", "==1.0+cpu", False),
        ]
        for version, spec, want in cases:
            self.assertEqual(lb.satisfies(version, spec), want, f"{version} {spec}")

    def test_conda_spec_to_pip(self):
        for rest, want in [
            ("=1.8.0", "==1.8.0"),
            (" 1.8.0", "==1.8.0"),
            ("=1.8.0=py312_0", "==1.8.0"),
            ("=1.8", ">=1.8,<1.9"),
            ("=3", ">=3,<4"),
            (">=1.0", ">=1.0"),
            ("", ""),
        ]:
            self.assertEqual(lb.conda_spec_to_pip(rest), want, rest)
        with self.assertRaises(ValueError):
            lb.conda_spec_to_pip("nonsense")


class TestParsing(unittest.TestCase):
    def test_requirement_lines(self):
        self.assertEqual(lb.parse_requirement_line("seaborn", "pip").spec, "")
        self.assertEqual(lb.parse_requirement_line("seaborn>=0.13,<0.14", "pip").spec, ">=0.13,<0.14")
        self.assertEqual(lb.parse_requirement_line("torchinfo==1.8.0", "pip").name, "torchinfo")
        self.assertEqual(lb.parse_requirement_line("numpy<2", "conda").spec, "<2")
        self.assertEqual(lb.parse_requirement_line("python=3.12", "conda").spec, ">=3.12,<3.13")
        self.assertIsNone(lb.parse_requirement_line("# comment", "pip"))
        with self.assertRaises(ValueError):  # a marker is a condition we would otherwise drop
            lb.parse_requirement_line("requests>=2; python_version<'3.9'", "pip")

    def test_multi_clause_conda_constraint_is_not_truncated(self):
        # The fail-open case: `numpy>=1,<2` must not silently become `>=1`.
        self.assertEqual(lb.parse_requirement_line("numpy>=1,<2", "conda").spec, ">=1,<2")
        plan = lb.classify([lb.Requirement("numpy", ">=1,<2")], {"numpy": "2.3.5"}, [])
        self.assertEqual([c.kind for c in plan.conflicts], ["version-drift"])

    def test_json_shape_is_checked(self):
        for text in [
            '{"name": "x", "dependencies": "torch"}',
            '{"name": "x", "dependencies": [{"pip": "torch"}]}',
            '{"name": "x", "dependencies": [{"pip": [1]}]}',  # an entry must be a string
            '{"name": "x", "dependencies": [{"pip": ["ok", 2]}]}',
        ]:
            with self.assertRaises(ValueError):
                lb.parse_environment(text, "bad.json")

    def test_resolved_and_channel_qualified_conda_lines(self):
        # `conda list` prints `numpy 1.26.4 py312_0`; `conda env export` can print a channel.
        self.assertEqual(lb.parse_requirement_line("numpy 1.26.4 py312_0", "conda").spec, "==1.26.4")
        qualified = lb.parse_requirement_line("conda-forge::numpy=1.26", "conda")
        self.assertEqual((qualified.name, qualified.spec), ("numpy", ">=1.26,<1.27"))

    def test_malformed_operators_are_refused_not_reinterpreted(self):
        # `><1` used to be read as a bare version, i.e. silently widened.
        for text in ["><1", "!==1", "~~1"]:
            with self.assertRaises(ValueError):
                lb.conda_spec_to_pip(text)

    def test_exact_pin_is_not_read_as_a_build_string(self):
        self.assertEqual(lb.conda_spec_to_pip("==1.8"), "==1.8")
        self.assertEqual(lb.conda_spec_to_pip("=1.8.0=py312_0"), "==1.8.0")

    def test_environment_yml(self):
        name, reqs = lb.parse_environment(
            "name: demo\n"
            "channels:\n"
            "  - conda-forge\n"
            "dependencies:\n"
            "  - python=3.12\n"
            "  - pip\n"
            "  - pip:\n"
            "      - seaborn==0.13.2\n"
            "      - torchinfo\n"
            "variables:\n"
            "  - OMP_NUM_THREADS: 8\n",
            "demo.yml",
        )
        self.assertEqual(name, "demo")
        self.assertEqual(
            [(r.name, r.spec, r.source) for r in reqs],
            [
                ("python", ">=3.12,<3.13", "conda"),
                ("pip", "", "conda"),
                ("seaborn", "==0.13.2", "pip"),
                ("torchinfo", "", "pip"),
            ],
        )

    def test_environment_yml_pip_only(self):
        name, reqs = lb.parse_environment(
            "name: extras\ndependencies:\n  - pip:\n      - seaborn\n", "extras.yml"
        )
        self.assertEqual(name, "extras")
        self.assertEqual([(r.name, r.source) for r in reqs], [("seaborn", "pip")])

    def test_the_repository_environment_parses(self):
        path = REPO / "environment.yml"
        if not path.is_file():
            self.skipTest("no environment.yml in this checkout")
        name, reqs = lb.parse_environment(path.read_text(), str(path))
        self.assertTrue(name, "the environment has a name")
        keys = {r.key for r in reqs}
        self.assertIn("torch", keys)
        self.assertIn("seaborn", keys)

    def test_conda_env_export_json(self):
        name, reqs = lb.parse_environment(
            json.dumps(
                {
                    "name": "demo",
                    "dependencies": ["python=3.12", {"pip": ["seaborn==0.13.2"]}],
                }
            ),
            "export.json",
        )
        self.assertEqual(name, "demo")
        self.assertEqual([r.name for r in reqs], ["python", "seaborn"])

    def test_conda_list_json(self):
        name, reqs = lb.parse_environment(
            json.dumps([{"name": "seaborn", "version": "0.13.2", "channel": "pypi"}]), "list.json"
        )
        self.assertEqual(name, "")
        self.assertEqual([(r.name, r.spec) for r in reqs], [("seaborn", "==0.13.2")])

    def test_bad_input_stops(self):
        for text in [
            "name: x\ndependencies: [seaborn]\n",  # flow style is not this subset
            "name: x\n",  # no packages at all
            "name: x\ndependencies:\n  - {name: seaborn}\n",  # an inline mapping
            "name: x\ndependencies:\n  - just a bare sentence\n",  # not a package name
        ]:
            with self.assertRaises(ValueError):
                lb.parse_environment(text, "bad.yml")
        with self.assertRaises(ValueError):
            lb.parse_environment("{not json", "bad.json")


class TestClassification(unittest.TestCase):
    IMAGE = {"seaborn": "0.13.2", "numpy": "2.3.5", "torch": "2.10.0+rocm7.0", "pip": "24.0"}

    def plan(self, *lines, allow=()):
        reqs = [lb.parse_requirement_line(line, "pip") for line in lines]
        return lb.classify([r for r in reqs if r], self.IMAGE, list(allow))

    def test_satisfied_is_left_alone(self):
        plan = self.plan("seaborn>=0.13", "torch==2.10.0")
        self.assertEqual(plan.installable(), [])
        self.assertEqual(sorted(r.name for r, _ in plan.satisfied), ["seaborn", "torch"])
        self.assertEqual(plan.conflicts, [])

    def test_missing_is_installed(self):
        plan = self.plan("torchinfo>=1.7")
        self.assertEqual(plan.installable(), ["torchinfo>=1.7"])
        self.assertEqual(plan.conflicts, [])

    def test_version_drift_is_a_conflict(self):
        plan = self.plan("numpy<2")
        self.assertEqual(len(plan.conflicts), 1)
        self.assertEqual(plan.conflicts[0].kind, "version-drift")
        self.assertEqual(plan.conflicts[0].image_version, "2.3.5")
        self.assertEqual(plan.installable(), [])

    def test_underscore_package_is_refused(self):
        # `_libgcc_mutex` normalizes to `-libgcc-mutex`, so the check must read the raw name.
        plan = self.plan("_libgcc_mutex=0.1")
        self.assertEqual([c.kind for c in plan.conflicts], ["non-pip"])
        # conda's compiler-runtime packages are the same category, and are never on PyPI
        for name in ("libgcc-ng", "libstdcxx-ng"):
            self.assertEqual(
                [c.kind for c in lb.classify([lb.Requirement(name)], {}, []).conflicts], ["non-pip"]
            )

    def test_allow_shadow_overrides_a_missing_stack_package(self):
        plan = self.plan("torchvision==0.20.0", allow=["torchvision"])
        self.assertEqual(plan.conflicts, [])
        self.assertEqual(plan.installable(), ["torchvision==0.20.0"])
        self.assertEqual(plan.allowed_shadow, ["torchvision"])

    def test_allow_shadow_overrides_deliberately(self):
        plan = self.plan("numpy<2", allow=["numpy"])
        self.assertEqual(plan.conflicts, [])
        self.assertEqual(plan.installable(), ["numpy<2"])
        self.assertEqual(plan.allowed_shadow, ["numpy"])

    def test_stack_package_the_image_lacks_is_refused(self):
        plan = self.plan("torchvision==0.20.0")
        self.assertEqual([c.kind for c in plan.conflicts], ["shadow-stack"])

    def test_runtime_is_refused(self):
        plan = self.plan("python==3.12.3")
        self.assertEqual([c.kind for c in plan.conflicts], ["non-pip"])

    def test_case_and_separator_insensitive(self):
        plan = self.plan("Seaborn>=0.13")
        self.assertEqual(plan.installable(), [])

    def test_every_conflict_says_why(self):
        plan = self.plan("numpy<2", "python", "torchvision")
        self.assertEqual(len(plan.conflicts), 3)
        for conflict in plan.conflicts:
            self.assertTrue(conflict.why, conflict)
            self.assertIn("kind", conflict.as_dict())


class TestEffects(unittest.TestCase):
    def test_pip_check_status_policy(self):
        # `pip check` exits 1 when it FINDS conflicts: the image has pre-existing ones, so that
        # status is data to compare, not a failure.
        self.assertFalse(lb.pip_check_failed(0))
        self.assertFalse(lb.pip_check_failed(1))
        self.assertTrue(lb.pip_check_failed(2))
        self.assertTrue(lb.pip_check_failed(127))

    def test_checksum_must_describe_this_base_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = pathlib.Path(tmp) / "base.sif"
            other = pathlib.Path(tmp) / "other.sif"
            base.write_text("the image we asked for\n")
            other.write_text("a different file\n")
            env = pathlib.Path(tmp) / "e.yml"
            env.write_text("name: demo\ndependencies:\n  - pip:\n      - seaborn\n")
            checksum = pathlib.Path(tmp) / "base.path.sha256"
            checksum.write_text(lb.run(["sha256sum", str(other)]))  # names the OTHER file
            options = lb.Options(
                base=str(base), env_file=str(env), layer_id="demo-1", out=tmp, python="python",
                allow_shadow=[], dry_run=True, as_json=False, export_to=None,
                base_sha256=str(checksum), timeout=60, keep_stage=False,
            )
            with self.assertRaises(lb.ToolError) as caught:
                lb.build(options)
            self.assertEqual(caught.exception.code, lb.EXIT_USAGE)
            self.assertIn("not the pinned one", str(caught.exception))

    def test_record_export_refuses_what_the_job_could_not_source_cleanly(self):
        # train.sbatch sources env.sh under `set -euo pipefail`, so the verification mirrors that.
        cases = {
            "export PROJECT_ID=x\n": True,
            "export MODEL_LAYER=/old\nset +e\nfalse\n": False,
            "export MODEL_LAYER=/old\nreturn 7\n": False,
            "false\nexport MODEL_LAYER=/old\n": False,
            "export PROJECT_ID=x\nexit 0\n": False,
        }
        for text, expected in cases.items():
            with tempfile.TemporaryDirectory() as tmp:
                path = pathlib.Path(tmp) / "env.sh"
                path.write_text(text)
                ok, _detail = lb.record_export(path, "/layer.sqsh")
                self.assertEqual(ok, expected, text)

    def test_publication_commits_records_before_the_layer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            venv = root / "venv"
            venv.mkdir()
            (venv / "marker").write_text("pretend a virtualenv\n")
            partial = root / "demo.sqsh.partial"
            partial.write_text("pretend a squashfs\n")
            sqsh = root / "demo.sqsh"
            artifacts = lb.publish(partial, sqsh, "seaborn==0.13.2\n", "layer=demo\n")
            for name in ("layer", "freeze", "manifest", "sha256"):
                self.assertTrue(pathlib.Path(artifacts[name]).is_file(), name)
            self.assertEqual(sorted(p.name for p in root.glob("*.partial")), [])
            lb.run(["sha256sum", "-c", artifacts["sha256"]])  # the record must verify

    def test_a_failed_publication_leaves_no_final_layer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            venv = root / "venv"
            venv.mkdir()
            partial = root / "demo.sqsh.partial"
            partial.write_text("pretend a squashfs\n")
            sqsh = root / "demo.sqsh"

            def boom(*_args, **_kwargs):
                raise lb.ToolError("the disk filled up")

            original, lb.run = lb.run, boom
            try:
                with self.assertRaises(lb.ToolError):
                    lb.publish(partial, sqsh, "freeze\n", "manifest\n")
            finally:
                lb.run = original
            # the layer is the commit marker: it must not exist while its records do not
            self.assertFalse(sqsh.exists())
            self.assertTrue(partial.exists())


    def test_regression_gate(self):
        noise = "vllm 0.22.1 requires compressed-tensors==0.15.0.1, which is not installed.\n"
        lb.check_regressions(noise, noise)  # the image's own conflict is not ours
        lb.check_regressions("", "")
        with self.assertRaises(lb.ToolError) as caught:
            lb.check_regressions(noise, noise + "seaborn 0.13.2 requires pandas\n")
        self.assertEqual(caught.exception.code, lb.EXIT_CONFLICT)

    def test_missing_checksum_file_stops_before_singularity_is_needed(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = pathlib.Path(tmp) / "image.sif"
            base.write_text("not really a container\n")
            env = pathlib.Path(tmp) / "e.yml"
            env.write_text("name: demo\ndependencies:\n  - pip:\n      - seaborn\n")
            options = lb.Options(
                base=str(base), env_file=str(env), layer_id="demo-1", out=tmp, python="python",
                allow_shadow=[], dry_run=True, as_json=False, export_to=None,
                base_sha256=str(pathlib.Path(tmp) / "missing.sha256"),
                timeout=60, keep_stage=False,
            )
            with self.assertRaises(lb.ToolError) as caught:
                lb.build(options)
            self.assertEqual(caught.exception.code, lb.EXIT_USAGE)
            self.assertIn("no checksum file", str(caught.exception))

    def test_import_probe_compiles_and_checks_metadata(self):
        program = lb.import_probe(["seaborn", "torchinfo"])
        compile(program, "<probe>", "exec")  # a program that does not parse is a build failure
        self.assertIn("packages_distributions", program)
        self.assertIn("torchinfo", program)

    def test_record_export_round_trips_and_keeps_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "env.sh"
            path.write_text("export PROJECT_ID=x\nexport MODEL_LAYER=/old.sqsh\n")
            path.chmod(0o664)
            ok, detail = lb.record_export(path, "/scratch/p/u/software/venvs/new.sqsh")
            self.assertTrue(ok, detail)
            text = path.read_text()
            self.assertEqual(text.count("export MODEL_LAYER="), 1)
            self.assertIn("export PROJECT_ID=x", text)
            self.assertIn("/scratch/p/u/software/venvs/new.sqsh", text)
            self.assertEqual(path.stat().st_mode & 0o777, 0o664)

    def test_record_export_appends_when_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "env.sh"
            path.write_text("export PROJECT_ID=x\n")
            self.assertTrue(lb.record_export(path, "/layer.sqsh")[0])
            self.assertEqual(path.read_text().count("export MODEL_LAYER="), 1)

    def test_record_export_refuses_a_file_that_exits_early(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "env.sh"
            original = "export PROJECT_ID=x\nexit 0\n"
            path.write_text(original)
            ok, detail = lb.record_export(path, "/layer.sqsh")
            self.assertFalse(ok)
            self.assertIn("round-trip", detail)
            self.assertEqual(path.read_text(), original)

    def test_record_export_refuses_a_value_the_shell_cannot_use(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "env.sh"
            path.write_text("export PROJECT_ID=x\n")
            ok, _ = lb.record_export(path, '/layer"; touch INJECTED; #.sqsh')
            if not ok:  # refused, and nothing was created
                self.assertFalse((pathlib.Path(tmp) / "INJECTED").exists())
            else:  # or written in a form the shell reads back exactly
                self.assertIn("MODEL_LAYER", path.read_text())

    def test_record_export_refuses_a_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            real = pathlib.Path(tmp) / "real.sh"
            real.write_text("export PROJECT_ID=x\n")
            link = pathlib.Path(tmp) / "env.sh"
            link.symlink_to(real)
            ok, detail = lb.record_export(link, "/layer.sqsh")
            self.assertFalse(ok)
            self.assertIn("regular env file", detail)
            self.assertEqual(real.read_text(), "export PROJECT_ID=x\n")


class TestCli(unittest.TestCase):
    def test_bad_id_is_refused(self):
        for bad in ["../evil", "a/b", "a b", "a;b", ".", ".."]:
            with self.assertRaises(lb.ToolError):
                lb.parse_args(["--base", "x.sif", "--env", "e.yml", "--id", bad])

    def test_report_json_is_what_an_agent_reads(self):
        plan = lb.classify(
            [lb.parse_requirement_line("numpy<2", "pip"), lb.parse_requirement_line("torchinfo", "pip")],
            {"numpy": "2.3.5"},
            [],
        )
        report = {
            "tool": lb.TOOL, "version": lb.VERSION, "base_image": "b.sif", "env_file": "e.yml",
            "environment": "demo", "layer_id": "demo-20260921", "python": "python",
            "image_packages": 1, "satisfied": [], "missing": ["torchinfo"],
            "allowed_shadow": [], "conflicts": [c.as_dict() for c in plan.conflicts],
        }
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            lb.print_report(report, True, plan)
        parsed = json.loads(buffer.getvalue())  # --json must be parseable, not merely printed
        self.assertEqual(parsed["conflicts"][0]["kind"], "version-drift")
        self.assertEqual(parsed["missing"], ["torchinfo"])

    def test_missing_env_file_is_a_usage_error_not_a_traceback(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(
                lb.main(["--base", "x.sif", "--env", "/no/such/environment.yml"]),
                lb.EXIT_USAGE,
            )

    def test_bad_environment_is_a_usage_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = pathlib.Path(tmp) / "bad.yml"
            env.write_text("name: x\ndependencies: [seaborn]\n")
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(
                    lb.main(["--base", "x.sif", "--env", str(env)]),  # no --id: it reads the file
                    lb.EXIT_USAGE,
                )

    def test_help_and_missing_args(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(lb.main(["--help"]), 0)
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(lb.main([]), lb.EXIT_USAGE)

    def test_refuses_inside_a_container(self):
        import os

        os.environ["SINGULARITY_CONTAINER"] = "/path/to/container"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                env = pathlib.Path(tmp) / "e.yml"
                env.write_text("name: demo\ndependencies:\n  - pip:\n      - seaborn\n")
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(
                        lb.main(["--base", "x.sif", "--env", str(env), "--dry-run"]),
                        lb.EXIT_ENVIRONMENT,
                    )
        finally:
            del os.environ["SINGULARITY_CONTAINER"]

    def test_derives_a_layer_id_from_the_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "env.yml"
            path.write_text("name: demo\ndependencies:\n  - pip:\n      - seaborn\n")
            options = lb.parse_args(["--base", "x.sif", "--env", str(path)])
            self.assertTrue(options.layer_id.startswith("demo-"), options.layer_id)


if __name__ == "__main__":
    unittest.main(verbosity=2)
