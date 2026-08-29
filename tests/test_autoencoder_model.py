import numpy as np
import pandas as pd
import pytest


pytest.importorskip("torch")
autoencoder_model = pytest.importorskip("covertlens.models.autoencoder_model")


def test_ood_points_have_higher_reconstruction_error() -> None:
    generator = np.random.default_rng(42)
    normal = pd.DataFrame(generator.normal(0.0, 0.15, size=(128, 4)))
    outliers = pd.DataFrame(np.full((8, 4), 6.0))

    model = autoencoder_model.train_autoencoder(normal, epochs=60, batch_size=32)
    normal_error = autoencoder_model.reconstruction_error(model, normal)
    outlier_error = autoencoder_model.reconstruction_error(model, outliers)

    assert outlier_error.mean() > normal_error.mean() * 10
    assert outlier_error.min() > normal_error.median()
