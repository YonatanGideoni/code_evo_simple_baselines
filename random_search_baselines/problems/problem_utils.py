import base64
import importlib
import inspect
import json
import os
import pickle
import subprocess
import sys
import tempfile
import textwrap
import time
import types
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any, Optional, Iterator

import cloudpickle as _cp
import numpy as np

PROBLEMS_DIR = Path(__file__).parent
CONFIG_FILE = 'config.json'


@dataclass
class EvaluationResult:
    """Standard evaluation result format."""
    success: bool
    execution_time: float
    error_message: str = ""
    metrics: dict[str, Any] = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}


class ProblemConfig:
    """Configuration for a specific problem."""

    def __init__(self, config_path: Path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

    def get(self, key: str, default=None):
        return self.config.get(key, default)

    def __getitem__(self, key: str):
        return self.config[key]


class BaseProblem(ABC):
    """Base class for defining optimization problems."""

    def __init__(self, config: ProblemConfig):
        self.config = config
        self.params = config['problem_parameters']

    def generate_instruction(self) -> str:
        # Get the file path of the actual class (not the base class)
        class_file = inspect.getfile(self.__class__)
        prompt_path = Path(class_file).parent / 'prompt.txt'

        instruction_template = prompt_path.read_text(encoding='utf-8')
        template = Template(instruction_template)
        return template.safe_substitute(max_execution_time=self.get_max_execution_time())

    @abstractmethod
    def get_function_name(self) -> str:
        """Return the name of the function that should be implemented."""
        pass

    @abstractmethod
    def get_namespace(self) -> dict[str, Any]:
        """Return the namespace to be used when executing the algorithm."""
        pass

    def get_max_execution_time(self) -> float:
        return self.params['max_execution_time']


def helper(fn):
    """Mark a function to be injected into the algorithm subprocess so it can be used e.g. for evalling a solution."""
    fn.__expose_to_algorithm__ = True
    return fn


class BaseEvaluator(ABC):
    """
    Base evaluator, handles:
      - subprocess execution with timeout
      - problem namespace reconstruction (modules, from-imports, literals)
      - helper code injection for the algorithm namespace
      - timing, error handling, and cleanup
      - a template method for parsing/validation/metrics

    Subclasses only need to implement a few hooks:
      - parse_output(raw, problem)
      - validate_output(parsed, problem)
      - compute_metrics(parsed, problem)
      - default_failure_metrics(problem)
      - (optionally) get_extra_namespace(problem)
      - (optionally) get_helper_code(problem)
    """

    # -------------------- public entrypoint (template method) --------------------

    def evaluate_algorithm(self, algorithm_code: str, problem: Any) -> EvaluationResult:
        start = time.time()
        try:
            raw = self._run_algorithm_with_timeout(algorithm_code, problem)
            parsed = self.parse_output(raw, problem)
            self.validate_output(parsed, problem)
            metrics = self.compute_metrics(parsed, problem)
            return EvaluationResult(
                success=True,
                execution_time=time.time() - start,
                metrics=metrics
            )
        except Exception as e:
            return EvaluationResult(
                success=False,
                execution_time=time.time() - start,
                error_message=str(e),
                metrics=self.default_failure_metrics(problem)
            )

    # -------------------- subclass hooks --------------------

    @abstractmethod
    def parse_output(self, raw_output: Any, problem: Any) -> Any:
        """Convert raw return value from the user function into the shape/type you expect."""
        ...

    @abstractmethod
    def validate_output(self, parsed_output: Any, problem: Any) -> None:
        """Raise if the parsed output is invalid."""
        ...

    @abstractmethod
    def compute_metrics(self, parsed_output: Any, problem: Any) -> dict[str, Any]:
        """Return a metrics dict for EvaluationResult."""
        ...

    @abstractmethod
    def default_failure_metrics(self, problem: Any) -> dict[str, Any]:
        """Return fallback metrics when evaluation fails."""
        ...

    # Optional extension points

    def get_extra_namespace(self, problem: Any) -> dict[str, Any]:
        """Let subclasses inject additional names (modules, constants, etc.) into the algorithm namespace."""
        return {}

    def get_helper_prelude(self, problem: Any) -> str:
        return ""  # e.g. "import itertools\nfrom scipy.special import hermite"

    @staticmethod
    def get_multiproc_prelude() -> str:
        # needed as some subprocesses call methods that use multiprocessing (e.g. scipy.optimize.differential_evolution)
        return r"""
try:
    import os as _os
    import cloudpickle as _cp
    import multiprocessing.reduction as _mpr
    # Make Pool/task pickling serialize via cloudpickle
    _mpr.ForkingPickler.dumps = _cp.dumps  # type: ignore[attr-defined]
except Exception:
    # If anything fails, leave default behavior in place.
    pass
            """

    def get_helper_code(self, problem: Any) -> str:
        cls = self.__class__
        parts: list[str] = []
        parts.append(self.get_multiproc_prelude())
        prelude = self.get_helper_prelude(problem).strip()
        if prelude:
            parts.append(prelude)

        # preserve definition order (Python 3.7+ keeps class dict insertion order)
        for name, attr in cls.__dict__.items():
            # only export static methods that are explicitly marked
            if isinstance(attr, staticmethod):
                fn = attr.__func__
                if getattr(fn, "__expose_to_algorithm__", False):
                    src = inspect.getsource(fn)
                    src = textwrap.dedent(src)
                    # strip any decorators (@staticmethod, @helper, etc.) so it becomes a plain top-level def
                    lines = [ln for ln in src.splitlines() if not ln.lstrip().startswith('@')]
                    parts.append("\n".join(lines))

        return "\n\n".join(p for p in parts if p)

    # -------------------- shared implementation --------------------

    def _get_algrun_script_prelude(
            self,
            module_imports: list[tuple[str, str]],
            from_imports: list[tuple[str, str, str]],
            literals_b64: str,
            helper_code: str
    ) -> str:
        """
        Return the script prelude that:
          - imports stdlibs and numpy
          - reconstructs the namespace from module/from imports and pickled literals
          - injects helper code into `namespace`
        """
        return f"""
import sys
import pickle
import base64
import numpy as np

# Reconstruct namespace
_module_imports = {repr(module_imports)}
_from_imports = {repr(from_imports)}
_literals = pickle.loads(base64.b64decode({literals_b64!r}))

namespace = {{}}

# Import full modules with alias (alias -> module)
for _alias, _modname in _module_imports:
    namespace[_alias] = __import__(_modname)

# Import named symbols with alias (from mod import sym as alias)
for _modname, _symname, _alias in _from_imports:
    _mod = __import__(_modname, fromlist=[_symname])
    namespace[_alias] = getattr(_mod, _symname)

# Add literal values
namespace.update(_literals)

# Inject helper code
_helper_code = {helper_code!r}
if _helper_code:
    exec(_helper_code, namespace)
""".lstrip()

    def _get_algrun_wrapper_script(
            self,
            module_imports: list[tuple[str, str]],
            from_imports: list[tuple[str, str, str]],
            literals_b64: str,
            helper_code: str,
            algorithm_code: str,
            problem: BaseProblem,
            result_path: str,
    ) -> str:
        prelude = self._get_algrun_script_prelude(module_imports, from_imports, literals_b64, helper_code)

        run_func_script = f"""
# Execute user's algorithm
algorithm_code = {algorithm_code!r}
exec(algorithm_code, namespace)

function_name = {getattr(problem, "get_function_name")()!r}
if function_name not in namespace:
    raise RuntimeError(f"Function {{function_name}} not found")

result = namespace[function_name]()

with open({result_path!r}, 'wb') as f:
    pickle.dump(result, f)
""".lstrip()

        return prelude + run_func_script

    def _unlink_paths(self, paths: list[str]) -> None:
        for p in paths:
            try:
                if os.path.exists(p):
                    os.unlink(p)
            except OSError:
                pass

    def _run_script(self, script_path: str, problem: BaseProblem, result_path: str) -> None:
        lax_timeout = problem.params.get('lax_timeout', False)
        timeout_mul = 1 if not lax_timeout else 3

        proc = subprocess.run(
            [sys.executable, script_path],
            timeout=getattr(problem, "get_max_execution_time")() * timeout_mul,
            capture_output=True,
            check=False
        )
        if proc.returncode != 0:
            err = proc.stderr.decode('utf-8', errors='replace') if proc.stderr else "Unknown subprocess error"
            raise RuntimeError(f"Subprocess failed: {err}")

        if not os.path.exists(result_path):
            raise RuntimeError("No results file created by subprocess")

    def _run_and_return_from_alg(self, script_path: str, result_path: str, problem: BaseProblem):
        try:
            self._run_script(script_path, problem, result_path)

            try:
                with open(result_path, 'rb') as f:
                    output = pickle.load(f)
            except Exception as e:
                raise RuntimeError(f"Failed to read subprocess results: {e}")

            return output

        except subprocess.TimeoutExpired:
            raise TimeoutError(
                f"Execution exceeded time limit of {getattr(problem, 'get_max_execution_time')():.1f} seconds")

        finally:
            self._unlink_paths([script_path, result_path])

    def _run_algorithm_with_timeout(self, algorithm_code: str, problem: Any):
        # Result file for IPC
        result_file = tempfile.NamedTemporaryFile(mode='w+b', delete=False)
        result_path = result_file.name
        result_file.close()

        # Merge problem namespace with any extras
        original_namespace = dict(getattr(problem, "get_namespace")())
        original_namespace.update(self.get_extra_namespace(problem))

        module_imports, from_imports, literals = self._decompose_namespace(original_namespace)

        helper_code = inspect.cleandoc(self.get_helper_code(problem))

        literals_b64 = base64.b64encode(pickle.dumps(literals, protocol=pickle.HIGHEST_PROTOCOL)).decode("ascii")
        wrapper_script = self._get_algrun_wrapper_script(module_imports, from_imports, literals_b64, helper_code,
                                                         algorithm_code, problem, result_path)
        wrapper_script = inspect.cleandoc(wrapper_script)

        # Write wrapper file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(wrapper_script)
            script_path = f.name

        return self._run_and_return_from_alg(script_path, result_path, problem)

    @staticmethod
    def _decompose_namespace(ns: dict[str, Any]) -> tuple[list, list, dict[str, Any]]:
        """
        Split a namespace dict into:
          - module_imports: [(alias, module_name)]
          - from_imports  : [(module_name, symbol_name, alias)]
          - literals      : {name: value}
        """
        module_imports = []
        from_imports = []
        literals = {}

        for name, value in ns.items():
            if isinstance(value, types.ModuleType):
                module_imports.append((name, value.__name__))
            elif hasattr(value, "__module__") and hasattr(value, "__name__") and value.__module__ not in ("builtins",):
                from_imports.append((value.__module__, value.__name__, name))
            else:
                # basic data / arrays / numbers / strings / dicts, etc.
                literals[name] = value
        return module_imports, from_imports, literals

    # -------------------- shared small helpers (optional to use) --------------------

    @staticmethod
    def assert_shape(arr: np.ndarray, expected: tuple[int, ...], name: str = "array") -> None:
        if tuple(arr.shape) != tuple(expected):
            raise ValueError(f"Shape mismatch for {name}: expected {expected}, got {arr.shape}")

    @staticmethod
    def assert_all_finite(arr: np.ndarray, name: str = "array") -> None:
        if np.any(~np.isfinite(arr)):
            raise ValueError(f"{name} contains NaN or infinite values")


class BaseClassProblem(BaseProblem, ABC):
    def get_function_name(self) -> str:
        raise NotImplementedError('Use get_class_name() for class-based problems')

    @abstractmethod
    def get_class_name(self) -> str:
        """Return the name of the class that should be implemented."""
        pass


class BaseClassEvaluator(BaseEvaluator, ABC):
    """
    Variant of BaseEvaluator where the subprocess returns a class rather than calling a function and returning its output.
    """

    def _get_algrun_wrapper_script(
            self,
            module_imports: list[tuple[str, str]],
            from_imports: list[tuple[str, str, str]],
            literals_b64: str,
            helper_code: str,
            algorithm_code: str,
            problem: BaseClassProblem,
            result_path: str,
    ) -> str:
        prelude = self._get_algrun_script_prelude(module_imports, from_imports, literals_b64, helper_code)

        class_name = getattr(problem, "get_class_name")()

        run_cls_script = f"""
# Execute user's algorithm
algorithm_code = {algorithm_code!r}
exec(algorithm_code, namespace)

class_name = {class_name!r}
if class_name not in namespace:
    raise RuntimeError(f"Class {{class_name}} not found")

cls_obj = namespace[class_name]
if not isinstance(cls_obj, type):
    raise TypeError(f"{{class_name}} exists but is not a class (got: {{type(cls_obj)}})")

# Serialize class via cloudpickle
try:
    import cloudpickle as _cp
except Exception as _e:
    raise RuntimeError(f"cloudpickle is required to return classes safely: {{_e}}")

with open({result_path!r}, 'wb') as f:
    _cp.dump(cls_obj, f)
""".lstrip()

        return prelude + run_cls_script

    def _run_and_return_from_alg(self, script_path: str, result_path: str, problem: BaseProblem):
        try:
            self._run_script(script_path, problem, result_path)

            try:
                with open(result_path, 'rb') as f:
                    output = _cp.load(f)
            except Exception as e:
                raise RuntimeError(f"Failed to read subprocess results: {e}")

            return output

        except subprocess.TimeoutExpired:
            raise TimeoutError(
                f"Execution exceeded time limit of {getattr(problem, 'get_max_execution_time')():.1f} seconds"
            )

        finally:
            self._unlink_paths([script_path, result_path])


class ProblemLoader:
    """
    PROBLEMS_DIR/
      geometry/triangle_area/{__init__.py,config.json,problem.py,evaluator.py}
      combinatorics/permutations/{...}
    Also supports uncategorized: PROBLEMS_DIR/<name>/{...}
    """

    @staticmethod
    def load_problem(
            problem_name: str,
            problem_type: Optional[str] = None,
    ) -> tuple[BaseProblem, BaseEvaluator, ProblemConfig]:
        category, name, problem_dir = ProblemLoader._locate_problem_dir(problem_name, problem_type)

        config_path = problem_dir / CONFIG_FILE
        if not config_path.exists():
            raise ValueError(f"Config file not found: {config_path}")

        config = ProblemConfig(config_path)

        # make parent of PROBLEMS_DIR importable so "problems.*" resolves
        pkg_root = PROBLEMS_DIR.parent
        if str(pkg_root) not in sys.path:
            sys.path.insert(0, str(pkg_root))

        base = ProblemLoader._fqmod_base(category, name)
        problem_module = importlib.import_module(f"{base}.problem")
        evaluator_module = importlib.import_module(f"{base}.evaluator")

        problem_class = getattr(problem_module, config["problem_class"])
        evaluator_class = getattr(evaluator_module, config["evaluator_class"])
        return problem_class(config), evaluator_class(config), config

    @staticmethod
    def list_types() -> list[str]:
        if not PROBLEMS_DIR.exists():
            return []

        # only dirs that actually contain at least one problem subdir
        cats = {cat for cat, _, _ in ProblemLoader._iter_problem_dirs()}
        return sorted(cats)

    @staticmethod
    def list_problems(problem_type: Optional[str] = None) -> list[str]:
        if not PROBLEMS_DIR.exists():
            return []

        if problem_type:
            return sorted(
                name for cat, name, _ in ProblemLoader._iter_problem_dirs(problem_type) if cat == problem_type)

        # qualified (cat/name) + uncategorized (name)
        out = []
        for cat, name, _ in ProblemLoader._iter_problem_dirs():
            out.append(f"{cat}/{name}" if cat else name)
        return sorted(set(out))

    # -------------------- helpers --------------------

    @staticmethod
    def _iter_problem_dirs(
            type_filter: Optional[str] = None,
    ) -> Iterator[tuple[str, str, Path]]:
        """Yield (category, problem_name, path) for all problems."""
        root = PROBLEMS_DIR
        if not root.exists():
            return

        for config in root.rglob(CONFIG_FILE):
            p = config.parent

            rel = p.relative_to(root)
            *category_parts, name = rel.parts

            if type_filter and (not category_parts or category_parts[0] != type_filter):
                continue

            category = ".".join(category_parts) if category_parts else None
            yield category, name, p

    @staticmethod
    def _locate_problem_dir(
            problem_name: str,
            problem_type: Optional[str],
    ) -> tuple[Optional[str], str, Path]:
        parts = Path(problem_name).parts

        if len(parts) >= 2 and not problem_type:
            problem_type, name = parts[-2], parts[-1]

        # explicit category arg
        if problem_type:
            name = Path(problem_name).name

            matches = ProblemLoader.get_category_name_matches(problem_type, name)

            if not matches:
                raise ValueError(f"Problem '{name}' in category '{problem_type}' not found under {PROBLEMS_DIR}")

            if len(matches) > 1:
                raise ValueError(f"Ambiguous problem '{name}' in category '{problem_type}': {matches}")

            p = matches[0]
            return problem_type, name, p

        # search all categories + uncategorized
        name = problem_name
        candidates = [(cat, n, path) for cat, n, path in ProblemLoader._iter_problem_dirs() if n == name]

        if not candidates:
            raise ValueError(f"Problem '{name}' not found anywhere under {PROBLEMS_DIR}")

        if len(candidates) > 1:
            cats = [c[0] or "<uncategorized>" for c in candidates]
            raise ValueError(f"Ambiguous problem name '{name}' in categories: {cats}. "
                             f"Use 'type/{name}' or problem_type=...")

        return candidates[0]

    @staticmethod
    def get_category_name_matches(category: Optional[str], name: str) -> list[Path]:
        if '.' in category:
            category = category.replace('.', '/')

        if category:
            matches = list(PROBLEMS_DIR.glob(f"**/{category}/{name}"))
        else:
            matches = list(PROBLEMS_DIR.glob(f"**/{name}"))

        return [p for p in matches if p.is_dir() and p.name == name]

    @staticmethod
    def _fqmod_base(category: Optional[str], name: str) -> str:
        matches = ProblemLoader.get_category_name_matches(category, name)

        assert len(matches) == 1, f"Expected exactly one match for problem path, category={category}, name={name}"

        p = matches[0].relative_to(PROBLEMS_DIR.parent)
        return ".".join(p.parts)
