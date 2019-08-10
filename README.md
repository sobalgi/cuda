# CUDA : Contradistinguisher for Unsupervised Domain Adaptation
**Paper Accepted in 19th IEEE International Conference on Data Mining (ICDM),  2019.**

Bibtex :

``
Bibtex here
``

Paper URL :

``
Paper URL here
``

The original code base for the experiments and results for Image datasets.

## Installation

You will need:

- Python 3.6 (Anaconda Python recommended)
- PyTorch 
- torchvision
- nltk
- pandas
- scipy
- pandas
- tqdm
- scikit-image 
- scikit-learn  
- tensorboardX
- tensorflow==1.13.1 (for tensorboard visualizations)

## Installation Instructions

On Linux:

```> conda install pytorch torchvision cudatoolkit=10.0 -c pytorch```

install relevant cuda if GPUs are available. 
Use of GPUs is very much recommended and inevitable because of the size of the model and datasets.

### The rest of the dependencies

Use requirements.txt in the respective sub-folders with pip as below:

```> pip install -r requirements.txt```


# Visual Domain Adaptation Experiments ([visual_README.md](visual/README.md))
Full details on visual domain adaptation including codes in [`visual`](visual) subdirectory.
**Coming soon!!!**

```> cd visual```


# Language Domain Adaptation Experiments ([language_README.md](language/README.md))
Full details on language domain adaptation including codes in [`language`](language) subdirectory.
**Coming soon!!!**

```> cd language```

# Acknowledgements
Special thanks to <a href="http://sml.csa.iisc.ac.in/index.html">Statistics and Machine Learning Group</a>, <a href="https://www.csa.iisc.ac.in/">Department of Computer Science and Automation</a>, <a href="https://www.iisc.ac.in/">Indian Institute of Science</a>, Bengaluru, India for proving the necessary computational resources for the experiments.

# Author Details
Sourabh Balgi
sourabhbalgi[at]gmail[dot]com
