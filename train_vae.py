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
import numpy as np
import tensorflow as tf
import UI as user
import Data_handler
import GeneratorMachine
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
    elif choice == "mnist_color": 
        dataloader = Data_handler.Color(color_version)
    elif choice == "DUMMY":
        dataloader = Data_handler.Dummytestdata()
    return dataloader

@log
def get_vae_model(choice:str):
    '''
    Function to get VAE model based on user input.

    @param choice: dataset choice from user input.
    @return: VAE model object.
    '''
    return GeneratorMachine.VAE(choice)

@log
def visualize_latent_space(model):
    model.visualize_latent_space()
    
@log
def generate_from_posterior(model):
    model.generate_from_posterior()

@log
def generate_from_prior(model):
    model.generate_from_prior()

@log
def train_vae(model, data, epochs:int, optimizer):
    for e in range(epochs):
        for i, tr_batch in enumerate(data.train):
            loss = model.train(tr_batch , optimizer)
    return model

## --- Main --- ##
def main():
    parsed = user.Input(LOGFILE)
    print("Valide argument, inicializing program...")
    data = get_data(parsed.args.dset)
    model = get_vae_model(parsed.args.dset)
    # Adam optimizer is my default choice 
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4) 
    train_vae(model, data, parsed.args.epochs, optimizer)

    if parsed.args.visualize_latent:
        model.visualize_latent_space()
    if parsed.args.generate_form_posterior:
        model.generate_from_posterior()
    if parsed.args.generate_form_prior:
        model.generate_from_prior()

if __name__ == "__main__":
    main()  