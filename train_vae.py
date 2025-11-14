'''
This module is the main script for a program that generates an image using a VAE model.

The following documentation is also provided when running the script from the terminal.

---
Train a Variational Autoencoder (VAE) with configurable options.

positional arguments:
  {mnist_bw,mnist_color}
                        Dataset to train the VAE on. Choices: "mnist_bw" (grayscale MNIST), "mnist_color" (colored MNIST).
  epochs                Number of training epochs (positive integer).

options:
  -h, --help            show this help message and exit
  --visualize_latent    Visualize the latent space after training.
  --generate_form_prior
                        Generate a new image from the prior distribution p(z).
  --generate_form_posterior
                        Generate a new image from the posterior distribution q(z|x).
  --clean_log           Overwrite the current log file. If no log file exists, prints a message.
  --custom_params CUSTOM_PARAMS
                        Custom parameters for the VAE model IN DEVELOPMENT

Examples of usage:
    python train_vae.py mnist_bw 50 --visualize_latent
    python train_vae.py mnist_color 200 --generate_form_prior --custom_params '{"lr":0.001,"beta":4}'
    python train_vae.py mnist_bw 10 --clean_log
'''

## --- Imports --- ##
import UI as user
from datetime import datetime

## --- Constants --- ##
LOGFILE = "log.txt"

## --- Functions --- ##
def log(func):
    '''
    Decorator to log time of function calls in external file.
    '''
    def wrapper(*args, **kwargs):
        with open(LOGFILE, "a") as log_file:
            log_file.write(f"{datetime.now()}: Entering '{func.__name__}' ::: ")
            result = func(*args, **kwargs)
            log_file.write(f"Exiting '{func.__name__}': {datetime.now()}\n")
        return result
    return wrapper

@log
def dummy_logtest():
    print("This is a dummy function to test logging.")

## --- Main --- ##
def main():
    parsed = user.Input(LOGFILE)
    dummy_logtest()

    for i in range(parsed.args.epochs):
        dummy_logtest()

if __name__ == "__main__":
    main()