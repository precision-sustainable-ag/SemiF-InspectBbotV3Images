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

### Running the Script
With the environment set up and activated, you can run the scripts provided in the repository to begin data exploration and analysis:

1. Ensure your Conda environment is activated:
   ```
   conda activate field
   ```
2. To run a script, use the following command syntax:
   ```bash
   sh run_volume_assessment.sh
   ```
3. [NOTE] Setup the pipeline in the main [config](conf/config.yaml#L11). To run a script, use the following command syntax:
   ```bash
   python FIELD_REPORT.py
   ```

## Script Workflow

1. **Setup Paths**:
   - Defines source, intermediate, and output directories based on the batch ID specified in the configuration.

2. **File Copying**:
   - Selects RAW files from the source directory using a sampling strategy.
   - Copies the selected files to a local "raw" directory in parallel.

3. **Load Transformation Matrix**:
   - Reads a precomputed 9x9 color correction matrix from a `.npz` file.

4. **Image Processing**:
   - Reads RAW files, performs demosaicing, applies the transformation matrix, and saves the results as JPEG images.

5. **Graceful Termination**:
   - Handles `SIGINT` (Ctrl+C) and other termination signals to ensure the script stops cleanly, avoiding hanging processes.

## Installation

1. Clone the repository.
2. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt #TODO change to `environment.yaml` and include instructions for miniconda
   ```

3. Ensure the `Hydra` configuration files (`config.yaml`) are set up with the appropriate paths and parameters.


## Usage

Run the script with Hydra to manage configurations:

```bash
python main.py mode=inspect_v31_unpreprocessed
```

### Example YAML Configuration (`config.yaml`):
```yaml
paths:
  root_dir: ${hydra:runtime.cwd}
  primary_storage_uploads: /mnt/research-projects/s/screberg/longterm_images2/semifield-upload
  local_upload: ${paths.root_dir}/data/semifield-upload
  color_matrix: ${paths.root_dir}/data/semifield-utils/image_development/color_matrix/transformation_matrix.npz

inspect_v31:
  batch_id: NC_2024-01-01
  sample_size: 10
  sample_strategy: "random"
  image_height: 9528
  image_width: 13376
  bit_depth: 8
  concurrent_workers: 8
```

## Key Configuration Parameters

- **Paths**:
  - `primary_storage_uploads`: Source directory for RAW files - lts lockers.
  - `local_upload`: Local directory for processing.
  - `color_matrix`: Path to the color correction matrix file.

- **Batch Processing**:
  - `batch_id`: Identifier for the batch to process.
  - `sample_size`: Number of RAW files to process (optional).
  - `sample_strategy`: Strategy for selecting files (`random`, `first`, `last`, `middle`).

- **Image Properties**:
  - `image_height`, `image_width`: Dimensions of the RAW images.
  - `bit_depth`: Output image bit depth (`8` or `16`).

- **Parallelism**:
  - `concurrent_workers`: Number of parallel workers for processing.

## Logging and Debugging

The script uses Python's `logging` module for detailed logs:
- File operations (e.g., skipped or copied files).
- Image processing status.
- Errors encountered during execution.

