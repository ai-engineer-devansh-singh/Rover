"""Code quality analysis across 14 dimensions.

Paxel Step 14: "Analyzing code quality: 117 files, 14 dimensions"

Uses a hybrid approach:
- Static analysis (AST, regex) for measurable dimensions
- LLM judge for qualitative dimensions

Dimensions:
1. Cyclomatic Complexity     (static)
2. Function Length           (static)
3. Test Coverage             (static)
4. Documentation             (static)
5. Type Safety               (static)
6. Error Handling            (LLM)
7. Naming Quality            (LLM)
8. Consistency               (static + LLM)
9. Modularity                (static)
10. Security                  (LLM)
11. Performance               (LLM)
12. Idiomaticness             (LLM)
13. Maintainability          (composite)
14. Overall Quality           (LLM judge)
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

from rover.llm.client import LLMClient
from rover.models import Session


class CodeQualityAnalyzer:
    """Analyze code quality across 14 dimensions for files touched in sessions."""

    # Dimensions that need LLM analysis
    LLM_DIMENSIONS = {
        "error_handling",
        "naming_quality",
        "consistency",
        "security",
        "performance",
        "idiomaticness",
        "overall_quality",
    }

    # Dimensions measurable via static analysis
    STATIC_DIMENSIONS = {
        "complexity",
        "function_length",
        "test_coverage",
        "documentation",
        "type_safety",
        "modularity",
    }

    DIMENSION_DESCRIPTIONS = {
        "complexity": "Cyclomatic complexity and cognitive load",
        "function_length": "Average function/method length",
        "test_coverage": "Presence and quality of tests",
        "documentation": "Docstrings, comments, and README quality",
        "type_safety": "Type hints, TypeScript strictness, typed signatures",
        "error_handling": "Try/catch, Result types, error propagation",
        "naming_quality": "Semantic meaningfulness of names",
        "consistency": "Style adherence within and across files",
        "modularity": "Coupling, cohesion, import graph health",
        "security": "SQL injection, XSS, secret leakage patterns",
        "performance": "Algorithmic complexity, N+1 queries, lazy loading",
        "idiomaticness": "Framework-specific best practices",
        "maintainability": "Composite score from complexity + docs + modularity",
        "overall_quality": "Holistic assessment by LLM judge",
    }

    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client or LLMClient()

    def analyze_session(self, session: Session, repo_root: Path | None = None) -> dict[str, Any]:
        """Analyze code quality for files touched in a session.

        Returns a dict mapping file paths to 14-dimension quality scores.
        """
        if not session.file_paths:
            return {}

        results: dict[str, dict[str, Any]] = {}
        for file_path in session.file_paths:
            # Skip non-code files
            if not self._is_code_file(file_path):
                continue

            full_path = repo_root / file_path if repo_root else Path(file_path)
            if not full_path.exists():
                continue

            try:
                static = self._static_analysis(full_path)
                llm = {}
                if self.client.is_available():
                    llm = self._llm_analysis(full_path, static)

                # Compute composite maintainability
                maintainability = self._compute_maintainability(static)

                results[file_path] = {
                    **static,
                    **llm,
                    "maintainability": maintainability,
                }
            except Exception:
                continue

        return results

    @staticmethod
    def _is_code_file(path: str) -> bool:
        """Check if path is a code file we can analyze."""
        code_extensions = {
            ".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java", ".kt",
            ".swift", ".rb", ".php", ".cs", ".cpp", ".c", ".h", ".scala", ".clj",
        }
        return any(path.lower().endswith(ext) for ext in code_extensions)

    def _static_analysis(self, file_path: Path) -> dict[str, Any]:
        """Run static analysis on a file."""
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        lines = content.split("\n")
        total_lines = len(lines)

        result: dict[str, Any] = {
            "lines": total_lines,
            "complexity": 0.0,
            "function_length": 0.0,
            "test_coverage": 0.0,
            "documentation": 0.0,
            "type_safety": 0.0,
            "modularity": 0.0,
        }

        # Python analysis
        if file_path.suffix == ".py":
            result.update(self._analyze_python(content, lines))
        # TypeScript/JavaScript analysis
        elif file_path.suffix in (".ts", ".tsx", ".js", ".jsx"):
            result.update(self._analyze_typescript(content, lines))
        # Generic fallback
        else:
            result.update(self._analyze_generic(content, lines))

        return result

    def _analyze_python(self, content: str, lines: list[str]) -> dict[str, Any]:
        """Python-specific static analysis."""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return self._analyze_generic(content, lines)

        complexities: list[float] = []
        func_lengths: list[int] = []
        docstrings = 0
        total_funcs = 0
        type_annotations = 0
        total_params = 0
        imports: set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                total_funcs += 1
                func_lines = node.end_lineno - node.lineno if node.end_lineno else 10
                func_lengths.append(func_lines)

                # Cyclomatic complexity approximation
                complexity = 1
                for child in ast.walk(node):
                    if isinstance(child, (ast.If, ast.While, ast.For, ast.With, ast.Try)):
                        complexity += 1
                    elif isinstance(child, ast.BoolOp):
                        complexity += len(child.values) - 1
                complexities.append(complexity)

                # Docstring check
                if ast.get_docstring(node):
                    docstrings += 1

                # Type annotations
                if node.returns:
                    type_annotations += 1
                for arg in node.args.args:
                    total_params += 1
                    if arg.annotation:
                        type_annotations += 1

            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    imports.add(alias.name.split(".")[0])

        # Compute scores (0-100)
        avg_complexity = sum(complexities) / len(complexities) if complexities else 0
        complexity_score = max(0, 100 - (avg_complexity - 3) * 10) if avg_complexity > 3 else 100

        avg_func_length = sum(func_lengths) / len(func_lengths) if func_lengths else 0
        func_length_score = max(0, 100 - (avg_func_length - 20) * 2) if avg_func_length > 20 else 100

        doc_score = (docstrings / total_funcs * 100) if total_funcs > 0 else 0
        type_score = (type_annotations / max(total_params, 1) * 100) if total_params > 0 else 0

        # Modularity: unique imports vs total imports
        modularity = min(100, len(imports) * 10) if imports else 50

        # Test coverage heuristic: does file have tests?
        test_score = 0.0
        if "test" in content.lower() or "pytest" in content.lower() or "unittest" in content.lower():
            test_score = 60.0
        # Check for companion test file
        test_file = Path(str(file_path).replace(".py", "_test.py")).exists() or \
                    Path(str(file_path).replace(".py", ".test.py")).exists() or \
                    (file_path.parent / f"test_{file_path.name}").exists()
        if test_file:
            test_score = 85.0

        return {
            "complexity": round(complexity_score, 1),
            "function_length": round(func_length_score, 1),
            "test_coverage": round(test_score, 1),
            "documentation": round(doc_score, 1),
            "type_safety": round(type_score, 1),
            "modularity": round(modularity, 1),
        }

    def _analyze_typescript(self, content: str, lines: list[str]) -> dict[str, Any]:
        """TypeScript/JavaScript static analysis."""
        total_lines = len(lines)

        # Function length approximation
        func_pattern = re.compile(r"(function\s+\w+|const\s+\w+\s*=\s*(async\s+)?\(|\w+\s*\(.*\)\s*=>)")
        func_starts = [i for i, line in enumerate(lines) if func_pattern.search(line)]
        func_lengths = []
        for start in func_starts:
            # Rough estimate: count lines until next function or end
            end = start + 1
            brace_count = lines[start].count("{") - lines[start].count("}")
            while end < len(lines) and brace_count > 0:
                brace_count += lines[end].count("{") - lines[end].count("}")
                end += 1
            func_lengths.append(end - start)

        avg_func_length = sum(func_lengths) / len(func_lengths) if func_lengths else 0
        func_length_score = max(0, 100 - (avg_func_length - 20) * 2) if avg_func_length > 20 else 100

        # Complexity: count if/else/for/while/try
        control_flow = len(re.findall(r"\b(if|else|for|while|switch|try|catch)\b", content))
        complexity_score = max(0, 100 - control_flow * 2)

        # Type safety: TypeScript-specific
        type_score = 50.0
        if file_path_str := getattr(self, "_current_file", ""):
            if ".ts" in file_path_str and ".d.ts" not in file_path_str:
                # Count type annotations
                type_patterns = len(re.findall(r":\s*(string|number|boolean|any|unknown|void|Promise<|Record<|Map<|Array<)", content))
                interface_count = len(re.findall(r"\binterface\s+\w+", content))
                type_score = min(100, (type_patterns + interface_count * 5) * 3)

        # Documentation
        jsdoc_count = len(re.findall(r"/\*\*", content))
        comment_lines = sum(1 for line in lines if line.strip().startswith("//") or line.strip().startswith("*"))
        doc_score = min(100, (jsdoc_count * 10 + comment_lines / max(total_lines, 1) * 50))

        # Test coverage heuristic
        test_score = 0.0
        if ".test." in content or ".spec." in content or "describe(" in content or "it(" in content:
            test_score = 70.0
        # Check for companion test file
        fp = getattr(self, "_current_path", None)
        if fp:
            base = str(fp).replace(".ts", "").replace(".tsx", "").replace(".js", "").replace(".jsx", "")
            test_files = [f"{base}.test.ts", f"{base}.spec.ts", f"{base}.test.tsx", f"{base}.spec.tsx"]
            if any(Path(tf).exists() for tf in test_files):
                test_score = 85.0

        # Modularity
        import_count = len(re.findall(r"\bimport\s+.*\bfrom\b", content))
        modularity = min(100, import_count * 5 + 20)

        return {
            "complexity": round(complexity_score, 1),
            "function_length": round(func_length_score, 1),
            "test_coverage": round(test_score, 1),
            "documentation": round(doc_score, 1),
            "type_safety": round(type_score, 1),
            "modularity": round(modularity, 1),
        }

    def _analyze_generic(self, content: str, lines: list[str]) -> dict[str, Any]:
        """Generic fallback analysis for any code file."""
        total_lines = len(lines)
        comment_lines = sum(1 for line in lines if line.strip().startswith(("#", "//", "*", "--")))
        doc_score = min(100, comment_lines / max(total_lines, 1) * 100)

        # Rough complexity: control flow keywords
        control_flow = len(re.findall(r"\b(if|else|for|while|switch|try|catch)\b", content.lower()))
        complexity_score = max(0, 100 - control_flow * 2)

        return {
            "complexity": round(complexity_score, 1),
            "function_length": 50.0,  # unknown
            "test_coverage": 0.0,  # unknown
            "documentation": round(doc_score, 1),
            "type_safety": 0.0,  # unknown
            "modularity": 50.0,  # unknown
        }

    def _llm_analysis(self, file_path: Path, static: dict[str, Any]) -> dict[str, Any]:
        """Use LLM to analyze qualitative dimensions."""
        # Read file with redaction
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            # Truncate and redact
            content = self._redact_content(content)
            content = content[:3000]  # Cap for LLM context
        except Exception:
            return {}

        prompt = f"""Analyze this code file across qualitative quality dimensions.

FILE: {file_path.name}
LANGUAGE: {file_path.suffix}
LINES: {static.get('lines', 0)}

CODE (first ~100 lines, redacted):
```
{content[:2000]}
```

STATIC SCORES:
- Complexity: {static.get('complexity', 0)}/100
- Documentation: {static.get('documentation', 0)}/100
- Type Safety: {static.get('type_safety', 0)}/100

Score these dimensions (0-100) with brief reasoning:
1. error_handling: Try/catch coverage, graceful failures, error propagation
2. naming_quality: Clarity and semantic meaning of names
3. consistency: Style adherence within file and with conventions
4. security: SQL injection, XSS, secret leakage, input validation
5. performance: Algorithmic efficiency, resource usage patterns
6. idiomaticness: Language/framework best practices
7. overall_quality: Holistic judgment

Return JSON:
{{
  "error_handling": {{"score": 0, "reasoning": "..."}},
  "naming_quality": {{"score": 0, "reasoning": "..."}},
  "consistency": {{"score": 0, "reasoning": "..."}},
  "security": {{"score": 0, "reasoning": "..."}},
  "performance": {{"score": 0, "reasoning": "..."}},
  "idiomaticness": {{"score": 0, "reasoning": "..."}},
  "overall_quality": {{"score": 0, "reasoning": "..."}}
}}
"""

        system = (
            "You are a senior code reviewer. Score code quality objectively. "
            "Respond in valid JSON only."
        )

        try:
            raw = self.client.generate(prompt, system=system, json_mode=True, temperature=0.3)
            result = self._extract_json(raw)
            if result:
                return {
                    k: v.get("score", 0) if isinstance(v, dict) else v
                    for k, v in result.items()
                }
        except Exception:
            pass

        return {}

    @staticmethod
    def _redact_content(content: str) -> str:
        """Redact sensitive content before sending to LLM."""
        # Redact API keys, tokens, connection strings
        patterns = [
            (r'sk-[a-zA-Z0-9]{48}', 'REDACTED_API_KEY'),
            (r'gh[pousr]_[A-Za-z0-9_]{36,}', 'REDACTED_GH_TOKEN'),
            (r'AKIA[0-9A-Z]{16}', 'REDACTED_AWS_KEY'),
            (r'sk_live_[a-zA-Z0-9]{24,}', 'REDACTED_STRIPE'),
            (r'eyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*', 'REDACTED_JWT'),
            (r'[a-f0-9]{32,64}', 'REDACTED_HASH'),
        ]
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content)
        return content

    @staticmethod
    def _compute_maintainability(static: dict[str, Any]) -> float:
        """Compute composite maintainability score."""
        scores = [
            static.get("complexity", 50),
            static.get("documentation", 50),
            static.get("modularity", 50),
            static.get("function_length", 50),
        ]
        return round(sum(scores) / len(scores), 1)

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
        text = text.strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return None


class SessionQualityReport:
    """Aggregate code quality results into a session-level report."""

    @staticmethod
    def aggregate(file_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Aggregate per-file quality into session-level metrics."""
        if not file_results:
            return {
                "files_analyzed": 0,
                "overall_quality": 0.0,
                "dimension_averages": {},
                "top_issues": [],
            }

        dimension_totals: dict[str, list[float]] = {}
        for file_path, dims in file_results.items():
            for dim, score in dims.items():
                if dim in ("lines", "source"):
                    continue
                if isinstance(score, (int, float)):
                    dimension_totals.setdefault(dim, []).append(float(score))

        dimension_averages = {
            dim: round(sum(scores) / len(scores), 1)
            for dim, scores in dimension_totals.items()
        }

        overall = sum(dimension_averages.values()) / len(dimension_averages) if dimension_averages else 0

        # Find weakest dimensions
        sorted_dims = sorted(dimension_averages.items(), key=lambda x: x[1])
        top_issues = [
            {"dimension": dim, "score": score, "description": CodeQualityAnalyzer.DIMENSION_DESCRIPTIONS.get(dim, "")}
            for dim, score in sorted_dims[:3]
        ]

        return {
            "files_analyzed": len(file_results),
            "overall_quality": round(overall, 1),
            "dimension_averages": dimension_averages,
            "top_issues": top_issues,
        }
