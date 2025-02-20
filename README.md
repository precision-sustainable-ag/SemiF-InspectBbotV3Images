# SemiF-InspectBbotV3Images

This Python script processes batches of RAW images, copying files, demosaicing, and applying color corrections.


## Installation and Setup

### Installing Conda
To manage the project's dependencies efficiently, we use Conda, a powerful package manager and environment manager. Follow these steps to install Conda if you haven't already:

1. Download the appropriate version of Miniconda for your operating system from the official [Miniconda website](https://docs.anaconda.com/free/miniconda/).
2. Follow the installation instructions provided on the website for your OS. This typically involves running the installer from the command line and following the on-screen prompts.
3. Once installed, open a new terminal window and type `conda list` to ensure Conda was installed correctly. You should see a list of installed packages.


### Setting Up Your Environment Using an Environment File
After installing Conda, you can set up an environment for this project using an environment file, which specifies all necessary dependencies. Here's how:

1. Clone this repository to your local machine.
2. Navigate to the repository directory in your terminal.
3. Locate the `environment.yaml` file in the repository. This file contains the list of packages needed for the project.
4. Create a new Conda environment by running the following command:
   ```bash
   conda env create -f environment.yaml
   ```
   This command reads the `environment.yaml` file and creates an environment with the name and dependencies specified within it.

5. Once the environment is created, activate it with:
   ```bash
   conda activate <env_name>
   ```
   Replace `<env_name>` with the name of the environment specified in the `environment.yaml` file.
6. (Optional): To update your environment after making changes to the environment.yaml file, run: 
    ```bash
    conda env update --file environment.yaml
    ```


## Scripts:

> **Info:** Run these scripts from the main repo directory.

1. **inspect_bbotv31_unpreprocessed.py**

   This script automates sampling, color demosaicing, downscaling, and file removal for RAW images:

   - **Sampling:** Selects a subset of RAW images from a source directory using different strategies (random, first, last, middle).
   - **Color Demosaicing:** Converts RAW Bayer-pattern images to RGB using bilinear interpolation and applies a transformation matrix for color correction.
   - **Downscaling:** Resizes processed images by a specified factor while preserving aspect ratio.
   - **File Removal:** Deletes original RAW and/or processed images if configured.

   The script leverages parallel processing.

   **Usage**

   ```bash
   python main.py tasks=[inspect_v31_unpreprocessed] batch_id=NC_2025-01-29
   ```
   Note: Replace the batch_id with your desired batch_id


2. **report.py**
   
   This script generates an **image collection report** by processing metadata and visualizing upload statistics:

   - **Image Sampling & Metadata Extraction:** Extracts filename details, timestamps, and file sizes from RAW images.
   - **Line Plots:** Generates time-based plots for **capture time**, **upload time**, and **upload delay** using Matplotlib.
   - **CSV & PDF Reports:** Saves metadata to CSV and creates a summary **PDF report** with key statistics and embedded plots.
   - **Partial Upload Detection:** Identifies incomplete uploads by comparing file sizes.

   **Usage**
   
   ```bash
   python main.py tasks=[report] batch_id=NC_2025-02-03
   ```
   Note: Replace the batch_id with your desired batch_id

3. **inspect_developed.py**

   This script provides a **sample of "developed" images** that have already been **preprocessed** and performs the following tasks:

   - **Sampling:** Selects a subset of **developed images (JPGs)**.
   - **Copying:** Transfers the sampled images to a specified output directory using parallel processing.
   - **Downscaling:** Optionally resizes the images by a configured factor while maintaining aspect ratio.
   - **File Removal:** Deletes original images from the output directory if downscaling is applied.

   **Usage**
   
   ```bash
   python main.py tasks=[inspect_developed] batch_id=NC_2025-01-29
   ```
   Note: Replace the batch_id with your desired batch_id

4. **download_rawtherapee.sh**

   This **bash script** automates the **download and setup** of RawTherapee 5.8 for Linux:

   - **Downloads** the RawTherapee 5.8 AppImage from the official source.  
   - **Grants execution permissions** to the downloaded file, allowing it to run as an application.

   **Usage**

   1. To install RawTherapee, run the installation script:
   
      ```bash
      bash scripts/download_rawtherapee.sh
      ```

   2. To run RawTherapee:

         a. open MobaExterm

         b. start local terminal

         c. script: 

         ```bash 
         ssh -X -C username@SUNNY.ece.ncsu.edu
         ```
         d. enter password

         e. move into SemiF-InspectBbotV3Images 

         ```bash
         cd SemiF-InspectBbotV3Images
         ```
         f. In the Mobaxterm terminal, and after changing directories to the repo, run:
          ```bash
         ./RawTherapee_5.8.AppImage
         ```
   
   3. To view images in RawTherapee:

      Once RawTherapee opens, move to and select the folder that contain the recently processed images.
   4. 
      From here, you can select the downscaled images and report to inspect for non-target weeds use the report to analyse image upload speed as well as progress
      
   


