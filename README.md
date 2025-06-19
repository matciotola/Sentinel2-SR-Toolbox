# A Comprehensive Benchmarking Framework for Sentinel-2 Sharpening: Methods, Dataset, and Evaluation Metrics



[A Comprehensive Benchmarking Framework for Sentinel-2 Sharpening: Methods, Dataset, and Evaluation Metrics](https://www.mdpi.com/2072-4292/17/12/1983): The advancement of super-resolution and sharpening algorithms for satellite images has significantly expanded the potential applications of remote sensing data. In the case of Sentinel-2, despite significant progress, the lack of standardized datasets and evaluation protocols has made it difficult to fairly compare existing methods and advance the state of the art. This work introduces a comprehensive benchmarking framework for Sentinel-2 sharpening, designed to address these challenges and foster future research.
It analyzes several state-of-the-art sharpening algorithms, selecting representative methods ranging from traditional pansharpening to ad hoc model-based optimization and deep learning approaches. All selected methods have been re-implemented within a consistent Python-based framework and evaluated on a suitably designed, large-scale Sentinel-2 dataset. This dataset features diverse geographical regions, land cover types, and acquisition conditions, ensuring robust training and testing scenarios. The performance of the sharpening methods is assessed using both reference-based and no-reference quality indexes, highlighting strengths, limitations, and open challenges of current state-of-the-art algorithms.
The proposed framework, dataset, and evaluation protocols are openly shared with the research community to promote collaboration and reproducibility
## Cite Sentinel 2 Sharpening Toolbox
If you use this toolbox in your research, please use the following BibTeX entry.

    @Article{rs17121983,
        AUTHOR = {Ciotola, Matteo and Guarino, Giuseppe and Mazza, Antonio and Poggi, Giovanni and Scarpa, Giuseppe},
        TITLE = {A Comprehensive Benchmarking Framework for Sentinel-2 Sharpening: Methods, Dataset, and Evaluation Metrics},
        JOURNAL = {Remote Sensing},
        VOLUME = {17},
        YEAR = {2025},
        NUMBER = {12},
        ARTICLE-NUMBER = {1983},
        URL = {https://www.mdpi.com/2072-4292/17/12/1983},
        ISSN = {2072-4292},
        DOI = {10.3390/rs17121983}
        }



## License

Copyright (c) 2025 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
All rights reserved.
This software should be used, reproduced and modified only for informational and nonprofit purposes.

By downloading and/or using any of these files, you implicitly agree to all the
terms of the license, as specified in the document [`LICENSE`](https://github.com/matciotola/Sentinel2-SR-Toolbox/LICENSE.md)
(included in this package)

## Dataset

The dataset used is downable from the [GRIP-UNINA website](https://www.grip.unina.it/download/prog/S2SRToolbox/).
However, we provide extensive instructions on how to download and elaborate the images correctly in the [`Dataset`](https://github.com/matciotola/hyperspectral_pansharpening_toolbox/tree/main/Dataset) folder of this repository.
For any problem or question, please contact me at If you have any problems or questions, please contact me by email ([matteo.ciotola@unina.it](mailto:matteo.ciotola@unina.it)).


## Prerequisites

All the functions and scripts were tested on Windows and Ubuntu O.S., with these constrains:

*   Python 3.10.10
*   PyTorch >= 2.0.0
*   Cuda  11.8 (For GPU acceleration).

the operation is not guaranteed with other configurations.

## Installation

*   Install [Anaconda](https://www.anaconda.com/products/individual) and [git](https://git-scm.com/downloads)
*   Create a folder in which save the toolbox
*   Download the toolbox and unzip it into the folder or, alternatively, from CLI:

<!---->

    git clone https://github.com/matciotola/Sentinel2-SR-Toolbox

*   Create the virtual environment with the `s2_sharp_toolbox_env.yml`

<!---->

    conda env create -n s2_sharp_toolbox_env -f s2_sharp_toolbox_env.yml

*   Activate the Conda Environment

<!---->

    conda activate s2_sharp_toolbox_env

* Edit the 'preamble.yaml' file with the correct paths and the desired algorithms to run

*   Test it

<!---->

    python main_20.py
    python main_60.py



