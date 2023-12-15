# PiX3L - Image Enhancer App

PiX3L is an intuitive web application designed for enhancing the quality of digital images using advanced AI models. It combines elements from StyleGAN, WGAN, and CycleGAN to provide powerful image enhancement capabilities.



![Architecture](https://esrgan.readthedocs.io/en/latest/_images/architecture.png)



## Features

- **Image Enhancement:** Utilize a hybrid AI model to enhance the quality of images.
- **Multiple AI Models:** Choose between ESRGAN and PSNR models for image super resolution.
- **Comparison:** Compare the original and enhanced images side by side.
- **Gallery:** Access an enhanced image gallery showcasing the output.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/itzsamr/pi3xl.git
    cd pix3l
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the application:

    ```bash
    streamlit run app.py
    ```

## Usage

- Launch the application by running the command specified in the installation steps.
- Login with the provided credentials or use your own authentication system.
- Upload images and choose the AI model for enhancement.
- Explore the different options available in the sidebar for image comparison and gallery viewing.

## Contribution

Contributions are welcome! Feel free to open issues for suggestions or bugs. Pull requests are also encouraged.
