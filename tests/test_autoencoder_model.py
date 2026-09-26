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


def test_small_dataset_clamps_batch_size(caplog, monkeypatch) -> None:
    normal = pd.DataFrame([[0.0, 0.1], [0.1, -0.1], [-0.1, 0.0]], index=[10, 20, 30])
    loader_class = autoencoder_model.DataLoader
    batch_sizes = []

    def capture_loader(*args, **kwargs):
        batch_sizes.append(kwargs["batch_size"])
        return loader_class(*args, **kwargs)

    monkeypatch.setattr(autoencoder_model, "DataLoader", capture_loader)
    model = autoencoder_model.train_autoencoder(normal, epochs=10)
    errors = autoencoder_model.reconstruction_error(model, normal)

    assert batch_sizes == [len(normal)]
    assert "batch_size 32 exceeds 3 flows; clamping to 3" in caplog.text
    assert errors.index.equals(normal.index)
    assert np.isfinite(errors).all()
    assert (errors >= 0).all()
