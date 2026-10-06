# Preparing the virtual environment on Windows

To set up the virtual environment for the DLF project on Windows, follow these steps:

1. **Install Python**: Make sure you have Python 3.8 or higher installed on your system (I'll be using Python 3.9.13 in this guide).

2. **Choose a directory**: Open the Command Prompt (cmd) and use the cd command to navigate to the directory where you want your DLF project to be located (I've chosen `C:\Code\DLF` for this guide and for my scripts).
   ```bash
   cd C:\Code\DLF
   ```

3. **Clone the repository**: Clone the DLF project repository into the chosen directory and navigate into it.
   ```bash
   git clone https://github.com/dlfcodec/Dual-generative-Latent-Fusion.git
   cd Dual-generative-Latent-Fusion
   ```

4. **Create and activate the virtual environment**: This will be important to ensure that all dependencies are installed in an isolated environment, preventing conflicts with other Python projects on your system.
   ```bash
   python -m venv dlf_env
   dlf_env\Scripts\activate
   ```

5. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install ninja
   pip install cmake
   ```

6. **Upgrade PyTorch packages**: The torch, torchvision and torchaudio packages will come in the wrong versions when installed via the requirements.txt file. You should then upgrade them using the following commands:

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```

This specific version comes with CUDA 12.4 support (If your system has another CUDA version configured, adjust the index URL accordingly), which is necessary for leveraging GPU acceleration with PyTorch on compatible hardware.

# Setting up Nvidia CUDA Toolkit

To ensure that your system can leverage GPU acceleration with PyTorch, you need to have the appropriate Nvidia CUDA Toolkit installed. You can download it from the official Nvidia website:

1. Visit the [Nvidia CUDA Toolkit 12.4 download page](https://developer.nvidia.com/cuda-12-4-0-download-archive).
2. Select your operating system, architecture, version and choose the exe(local) installer.
3. Follow the installation instructions.

After installing the CUDA Toolkit, make sure to verify the installation by running:
```bash
nvcc --version
```
# Downloading and Installing cuDNN

cuDNN is a GPU-accelerated library for deep neural networks, which is required for optimal performance with PyTorch. To download and install cuDNN:

1. Visit the [Nvidia cuDNN download page](https://developer.nvidia.com/rdp/cudnn-archive).
2. Download the cuDNN version 8.9.7 for CUDA 12.X library for Windows (Zip).
3. Extract the contents and copy them to the corresponding CUDA Toolkit directories (usually `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4`).

For example, you should copy the cuDNN files from `cudnn-windows-x86_64-8.9.7.29_cuda12-archive\bin`, to the respective `bin` folder inside of the CUDA Toolkit directories (usually `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin`), and similarly for the `include` and `lib` folders.

# Compiling C++ Extensions

Firstly, you'll need to install the necessary build tools for compiling C++ extensions. You can do that by installing the Visual Studio Build Tools **with the desktop development with C++ workload**, which include the required C++ compiler and libraries.

1. Visit the [Visual Studio Build Tools download page](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
2. Download and run the installer.
3. In the installer, **select the "Desktop development with C++" workload** (for me it was the first option) and install it.

After that, we'll build the C++ extensions required for the project.

1. **Make sure the venv is still activated**: You can check this by looking at your command prompt; it should show the name of the virtual environment (e.g., `(dlf_env)`) at the beginning of the line. If it is not activated, you can activate it by using:

   ```bash
   C:/Code/DLF/Dual-generative-Latent-Fusion/dlf_env/Scripts/activate
   ```
   If you have created the exact folder structure as mentioned above, the command will correctly activate the virtual environment.

2. **Prepare the compilation directory**: Navigate to the root directory of the project where the C++ extensions need to be compiled. And create a build folder for cmake:
   ```bash
   cd C:/Code/DLF/Dual-generative-Latent-Fusion/src
   mkdir build
   cd build
   ```

3. **Run CMake to configure the build**: From within the `build` directory, run the following command to configure the build system:
   ```bash
   cmake ../cpp -DCMAKE_POLICY_VERSION_MINIMUM=3.5
   ```

4. **Build the C++ extensions**: After configuring the build system with CMake, you can compile the extensions by running:
   ```bash
   cmake --build . --config Release
   ```

5. **Verify the build**: After the build completes, you should see the compiled C++ extension files in the `build` directory. You can also check for any errors during the build process to ensure that everything was compiled correctly.

# Deploying the Scripts

Inside of the Dual-generative-Latent-Fusion directory, you will need to create some folders:

```bash
cd C:/Code/DLF/Dual-generative-Latent-Fusion
mkdir Images
mkdir Models
mkdir Scripts
cd Images
mkdir Originals
mkdir Compressed
mkdir Decompressed
```

Inside the `Models` directory, you will need to download the zip files for the models required for the project. They are avaiable [here](https://onedrive.live.com/?id=%2Fpersonal%2Faf332a47fcf136b4%2FDocuments%2F%E5%85%B1%E4%BA%AB%2FDLF%2FDLF_models&listurl=%2Fpersonal%2Faf332a47fcf136b4%2FDocuments&ithint=folder&migratedtospo=true&redeem=aHR0cHM6Ly8xZHJ2Lm1zL2YvYy9hZjMzMmE0N2ZjZjEzNmI0L0VzeXQ1c2NXeFRGR2k4THU3aVNwVU9FQkFFUVVRVm9CRjZaY2JmSU9SdmRUY2c&ga=1). After downloading, extract the contents of the zip files into the `Models` directory.

You can download the `compressor.py` and `decompressor.py` scripts from this repository and place them inside the `Scripts` directory.

And inside the `Images` directory, you should place your original images in the `Originals` folder. After running the compression and decompression scripts, the resulting files will be stored in the `Compressed` and `Decompressed` folders, respectively.

***For the scripts to run correctly you'll need to add the Microsoft Visual C++ build tools to your system PATH*** - Your file paths may vary depending on your installation. After adding the build tools to your PATH, restart your terminal or VSCode (or your whole system) to apply the changes.
```bash
# Add Microsoft Visual C++ build tools to PATH
set PATH=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64;%PATH%
```

It is common for VSCode to show false errors withing the `compression.py` script. You can usually ignore these warnings as long as the scripts run correctly.