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
  --color_version {1,2,3,4,5}
                        chosen color dataset variant (1 to 5). Only relevant if "mnist_color" is chosen as dataset.
  --custom_params CUSTOM_PARAMS
                        Custom parameters for the VAE model IN DEVELOPMENT

Examples of usage:
    python train_vae.py mnist_bw 50 --visualize_latent
    python train_vae.py mnist_color 200 --generate_form_prior --custom_params '{"lr":0.001,"beta":4}'
    python train_vae.py mnist_bw 10 --clean_log
'''

## --- Imports --- ##
import UI as user
import Data_handler
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
def get_data(choice:str, color_version = 1):
    '''
    Function to get data based on user input.

    @param choice: dataset choice from user input.
    @return: DataLoader object with the selected dataset.
    '''
    if choice == "mnist_bw":
        dataloader = Data_handler.BlackWhite()
    else: # No other options due to argparse choices
        dataloader = Data_handler.Color(color_version)
    return dataloader

## --- Main --- ##
def main():
    parsed = user.Input(LOGFILE)
    print("Valide argument, inicializing program...")
    data = get_data(parsed.args.dset)
    first_batch = next(iter(data.train))
    print(first_batch.shape[-1])

if __name__ == "__main__":
    main()