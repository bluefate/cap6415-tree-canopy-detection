#!/usr/bin/env python
"""
Script to systematically add or update docstrings across the codebase.
This helps ensure consistent documentation for all functions, methods, and classes.
"""

import ast
import re
from pathlib import Path
from typing import List, Dict, Any


def check_docstring(node) -> bool:
    """Check if a node has a docstring."""
    return (
        ast.get_docstring(node) is not None
        and len(ast.get_docstring(node).strip()) > 0
    )


def analyze_file(filepath: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Analyze a Python file to identify functions, methods, and classes.
    
    Args:
        filepath (Path): Path to the Python file to analyze.
    
    Returns:
        Dict: Contains lists of classes, functions, and their docstring status.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
    except Exception as e:
        return {"error": str(e)}

    result = {
        "classes": [],
        "functions": [],
        "methods_missing_docs": [],
        "functions_missing_docs": [],
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            has_doc = check_docstring(node)
            result["classes"].append({
                "name": node.name,
                "line": node.lineno,
                "has_docstring": has_doc,
                "methods": []
            })
            
            # Check class methods
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    has_method_doc = check_docstring(item)
                    if not has_method_doc:
                        result["methods_missing_docs"].append({
                            "class": node.name,
                            "method": item.name,
                            "line": item.lineno
                        })

        elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:
            has_doc = check_docstring(node)
            if not has_doc:
                result["functions_missing_docs"].append({
                    "name": node.name,
                    "line": node.lineno
                })

    return result


def analyze_codebase(src_path: Path) -> Dict[str, Any]:
    """
    Analyze entire codebase to identify documentation needs.
    
    Args:
        src_path (Path): Path to the src directory.
    
    Returns:
        Dict: Summary of all files and missing documentation.
    """
    results = {}
    
    for py_file in src_path.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        
        analysis = analyze_file(py_file)
        if "error" not in analysis:
            total_missing = (
                len(analysis["functions_missing_docs"]) +
                len(analysis["methods_missing_docs"])
            )
            if total_missing > 0:
                results[str(py_file)] = analysis

    return results


def print_summary(results: Dict[str, Any]):
    """Print a summary of files needing documentation."""
    total_files = len(results)
    total_items_missing = 0

    print(f"\n{'='*80}")
    print(f"Documentation Audit Summary")
    print(f"{'='*80}\n")
    print(f"Files needing documentation updates: {total_files}\n")

    for filepath, analysis in sorted(results.items()):
        functions_missing = len(analysis["functions_missing_docs"])
        methods_missing = len(analysis["methods_missing_docs"])
        total_missing = functions_missing + methods_missing

        if total_missing > 0:
            total_items_missing += total_missing
            print(f"File: {filepath}")
            if functions_missing > 0:
                print(f"  Functions missing docstrings: {functions_missing}")
                for func in analysis["functions_missing_docs"]:
                    print(f"    - {func['name']} (line {func['line']})")
            if methods_missing > 0:
                print(f"  Methods missing docstrings: {methods_missing}")
                for method in analysis["methods_missing_docs"]:
                    print(f"    - {method['class']}.{method['method']} (line {method['line']})")
            print()

    print(f"{'='*80}")
    print(f"Total items missing documentation: {total_items_missing}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    src_path = Path("c:/github/Tree-Canopy-Detection/src")
    
    if not src_path.exists():
        print(f"Error: src path not found: {src_path}")
    else:
        results = analyze_codebase(src_path)
        print_summary(results)
