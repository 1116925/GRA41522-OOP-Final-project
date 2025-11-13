'''
This module is the main script for a program that generates an image using a VAE model.

The following documentation is also provided when running the script from the terminal.

---
'''

## --- Imports --- ##
import UI as user

## --- Functions --- ##

## --- Main --- ##
def main():
    parsed = user.Input()
    print("Parsed arguments:", parsed.args)

if __name__ == "__main__":
    main()