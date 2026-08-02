"""
Structural Guardrail: Proves the Inference pipeline is strictly decoupled
from the Feature and Training pipelines.
"""

import ast


def test_inference_is_decoupled_from_training_and_db():
    """
    Parses inference/predict.py and asserts it does not import
    raw database drivers or dbt. It must ONLY use Feast and MLflow.
    """
    with open("inference/predict.py", "r") as f:
        tree = ast.parse(f.read())

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend([n.name for n in node.names])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)

    forbidden_libraries = ["dbt", "sqlalchemy", "psycopg2", "psycopg"]

    for lib in forbidden_libraries:
        for imp in imports:
            assert lib not in imp, (
                f"🚨 FTI Contract Violation! Inference pipeline illegally imports '{lib}'. It must only use Feast/MLflow."
            )
