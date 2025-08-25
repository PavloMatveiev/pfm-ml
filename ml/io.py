from __future__ import annotations
import pickle
from typing import Sequence


def save_artifacts(pipeline, labels: Sequence[str], model_path: str = "model.pkl") -> None:
    """Persist the fitted pipeline to a single pickle, embedding the label list.

    The file contents are a dict: {"pipeline": fitted_pipeline, "labels": [str, ...]}.
    """
    with open(model_path, "wb") as f:
        pickle.dump({"pipeline": pipeline, "labels": list(labels)}, f)
