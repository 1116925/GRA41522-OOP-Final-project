import pytest
from UI import Input

parser = Input(True)  # Initialize Input in test mode

def test_valid_mnist_bw():
    parser.args = ["train_vae.py", "mnist_bw", "10", "--visualize_latent"]
    assert parser.args.dset == "mnist_bw"
    assert parser.args.epochs == 10
    assert parser.args.visualize_latent is True

def test_valid_mnist_color_with_custom_params():
    parser.args = ["train_vae.py", "mnist_color", "20", "--custom_params", '{"lr":0.001}']
    assert parser.args.dset == "mnist_color"
    assert parser.args.epochs == 20
    assert parser.args.custom_params == '{"lr":0.001}'

def test_invalid_dataset():
    with pytest.raises(SystemExit):
        parser.args = ["train_vae.py", "cifar10", "10"]

def test_negative_epochs():
    with pytest.raises(SystemExit):
        parser.args = ["train_vae.py", "mnist_bw", "-5"]

def test_non_integer_epochs():
    with pytest.raises(SystemExit):
        parser.args = ["train_vae.py", "mnist_bw", "abc"]

def test_large_epochs_warning(capfd):
    parser.args = ["train_vae.py", "mnist_bw", "1000"]
    assert parser.args.epochs == 1000
    # Capture stderr for warning
    captured = capfd.readouterr()
    assert "Warning" in captured.err