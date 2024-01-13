# Image Processing and Nanoparticle Quantification for Effective Biomarker Detection in Sepsis Diagnostics

![NEOSENS Logo](<NEOSENS_Logo.png>)

## Description

This image processing software is designed for nanoparticle identification and quantification, developed in the context of NEOSENS, a startup focused on Neonatal Sepsis diagnostics. Neonatal Sepsis is a life-threatening condition that demands early diagnosis and treatment. Emerging optical biosensing technologies hold promise for accurate biomarker concentration detection, requiring robust image analysis software for single nanoparticle detection.

In this thesis project, various image processing methods for nanoparticle processing and detection have been tested and compared to find the optimal pipeline. The comparison utilized analytical metrics and visual inspection to assess the efficacy of these techniques.

The results demonstrate that processing using Gaussian filters and K-space filters significantly improves image quality. When combined with the right segmentation tool, such as a global fixed threshold, it can achieve a low limit of detection (< 0.05ng/mL) with a short computation time, making it suitable for clinical testing.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Utility Scripts (Optional)](#utility-scripts-optional)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Prerequisites

Before you begin, ensure you have met the following requirements:

- opencv-python
- numpy
- scikit-image
- pandas

You can install these libraries by running `pip install -r requirements.txt`.

### Optional Dependencies for Utility Scripts

- matplotlib
- moviepy

Uncomment these dependencies in the `requirements.txt` file if needed.

## Installation

Follow these steps to install and set up the project:

1. Clone the repository:
   ```bash
   git clone <repository_url>
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
3. [Optional] Install additional dependencies for utility scripts:
   ```bash
   pip install matplotlib moviepy

## Usage

This project includes a main script and an optimized pipeline for nanoparticle detection. You can use the provided functions to test various parameters and study their behavior to find the optimal settings.

For detailed usage instructions, refer to the project's documentation and code comments.

## Utility Scripts (Optional)

Several utility scripts are included in this project to perform specific tasks. These scripts may have optional dependencies. You can find the list of utility scripts in the project directory. Uncomment the required dependencies in the `requirements.txt` file if you plan to use these scripts.

## Contributing

Contributions to this project are welcome! To contribute, follow these steps:

1. Fork the project.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them.
4. Create a pull request to submit your contribution.

## License

This project is not licensed.

## Contact

If you have any questions or need assistance, feel free to reach out to Mateo Hamel at mateo.hamel@neosens-dx.com.

