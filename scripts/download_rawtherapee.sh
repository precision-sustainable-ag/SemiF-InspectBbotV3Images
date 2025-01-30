#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Print commands and their arguments as they are executed
set -x

wget https://rawtherapee.com/shared/builds/linux/RawTherapee_5.8.AppImage
chmod +x RawTherapee_5.8.AppImage