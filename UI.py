'''
    A module that holds the input class used for parsing command line inputs.
'''

## --- Imports --- ##
import argparse
import os
import sys
from unittest.mock import patch

class Input:
    '''
    Interfaces the command line inputs and the training machine.

    ---
    Main use is to extract the *args* property which holds the parsed arguments.
    '''
    def __init__(self, is_test=False):
        self._is_test = is_test
        self._parser = self._get_parser()
        if not is_test:
            self._args = self._parser.parse_args()
            self._clean_log_warning()
        else: # In test mode, args will be set manually via the setter
            self._args = None

    @property
    def args(self): # Getter for parsed arguments
        return self._args   
    @args.setter
    def args(self, new_args): # EXCLUSIVE FOR TESTING Setter for forced parsed arguments. Used for testing with mocked sys.argv.
        if not self._is_test:
            raise AttributeError("Cannot set args when not in test mode.")
        with patch.object(sys, 'argv', new_args):
            self._args = self._parser.parse_args()
            self._clean_log_warning()
        
    @property
    def parser(self): # Getter for the argument parser
        return self._parser

    def _positive_int(self, value):
        '''
        Does a sanity check on the "epochs" input to ensure it's a positive \n
        integer. raise ArgumentTypeError: if input is invalid

        @param value: input value from command line
        @return: sanitized positive integer input
        '''
        try:
            ivalue = int(value) # Catch non-integer inputs
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid int value: '{value}'")
        if ivalue <= 0:    # Catch negative and zero 
            raise argparse.ArgumentTypeError("Epochs must be a positive integer.")
        if ivalue > 500:   # threshold for 'large'
            print(f"Warning: {ivalue} epochs is quite large, training may take a long time.", file=sys.stderr)
        return ivalue

    def _get_parser(self):
        parser = argparse.ArgumentParser(
            description="Train a Variational Autoencoder (VAE) with configurable options.",
            epilog='''Examples of usage:
    python train_vae.py mnist_bw 50 --visualize_latent
    python train_vae.py mnist_color 200 --generate_form_prior --custom_params '{"lr":0.001,"beta":4}'
    python train_vae.py mnist_bw 10 --clean_log
            ''',
            formatter_class=argparse.RawTextHelpFormatter
        )

        # 1. Mandatory dataset choice
        parser.add_argument(
            "dset",
            choices=["mnist_bw", "mnist_color"],
            help='Dataset to train the VAE on. Choices: "mnist_bw" (grayscale MNIST), "mnist_color" (colored MNIST).'
        )

        # 2. Mandatory epochs
        parser.add_argument(
            "epochs",
            type= self._positive_int,
            help="Number of training epochs (positive integer)."
        )

        # 3. Optional visualize_latent
        parser.add_argument(
            "--visualize_latent",
            action="store_true",
            help="Visualize the latent space after training."
        )

        # 4. Optional generate_form_prior
        parser.add_argument(
            "--generate_form_prior",
            action="store_true",
            help="Generate a new image from the prior distribution p(z)."
        )

        # 5. Optional generate_form_posterior
        parser.add_argument(
            "--generate_form_posterior",
            action="store_true",
            help="Generate a new image from the posterior distribution q(z|x)."
        )

        # 6. Optional clean_log
        parser.add_argument(
            "--clean_log",
            action="store_true",
            help="Overwrite the current log file. If no log file exists, prints a message."
        )

        # 7. Optional custom_params
        parser.add_argument(
            "--custom_params", # IN DEVELOPMENT
            type=str,
            help="Custom parameters for the VAE model IN DEVELOPMENT"
        )

        return parser
    
    def _clean_log_warning(self):
        # Handle clean_log logic
        if self._args.clean_log:
            if not os.path.exists("training.log"):
                print("No file to clean, a new file will be created.")
            else:
                print("Log file will be overwritten.")