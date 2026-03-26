# mc-hg-psf-ecd

A unified ML–TDDFT workflow for ECD spectrum prediction, combining MC-HG time-domain surrogate dynamics, PSF-ECD static-structure modeling, and cross-scale mechanism analysis.

---

## Overview

This repository presents a unified research-oriented workflow for **electronic circular dichroism (ECD) spectrum prediction**, aiming to connect machine learning with time-dependent electronic-structure modeling in a physically meaningful manner.

The project integrates three core components:

- **MC-HG** for time-domain surrogate dynamics
- **PSF-ECD** for static-structure-based spectrum modeling
- **Cross-scale mechanism analysis** for interpreting the relationships among structure, dynamics, and spectral response

Our goal is to explore how machine learning can assist ECD prediction while still preserving physical interpretability and scientific insight. This repository contains code, datasets, representative case-study folders, and related materials developed during this ongoing research.

---

## Motivation

Predicting ECD spectra accurately and efficiently remains a challenging problem in computational chemistry and spectroscopy, especially for systems with complex structures, nontrivial excited-state behavior, or strong sensitivity to geometric and electronic details.

This project is motivated by the following questions:

- Can machine learning help accelerate ECD spectrum prediction?
- Can time-domain dynamical information and static structural descriptors be integrated into a unified framework?
- Can such a framework go beyond prediction and provide useful clues for mechanism interpretation?

The present repository is an attempt toward such a unified ML–TDDFT-oriented workflow.

---

## Main Features

- Machine-learning-assisted ECD spectrum prediction
- Combination of **time-domain surrogate dynamics** and **static-structure modeling**
- Support for representative molecular or cluster case studies
- Preliminary basis for **cross-scale mechanism analysis**
- Python-based research code for further extension and reproducibility

---

## Repository Structure

At the current stage, the repository includes several development-stage folders and case-study materials. The current public structure is organized as follows:

```text
mc-hg-psf-ecd/
├── project2/
│   ├── dataset/
│   ├── ecd_58_2/
│   ├── tabpfn-extensions/
│   ├── xyz/
│   └── cdspecX.py
├── project2_CD/
│   └── projec1/
│       ├── Ag4/
│       ├── Ag4_TCM/
│       ├── Ag78/
│       ├── GG-Ag2/
│       └── xyz/
├── LICENSE
└── README.md
```

### Brief Description

- `project2/`  
  Main working directory containing code, datasets, structure files, and related modules.

- `project2/dataset/`  
  Dataset-related files used in model development or analysis.

- `project2/ecd_58_2/`  
  ECD-related project materials or intermediate working files.

- `project2/tabpfn-extensions/`  
  Extension modules or experimental components related to model development.

- `project2/xyz/`  
  Molecular structure files in XYZ format.

- `project2/cdspecX.py`  
  A spectrum-related Python script in the current workflow.

- `project2_CD/projec1/`  
  Case-study directory containing representative systems and related analysis materials.

- `project2_CD/projec1/Ag4/`, `Ag4_TCM/`, `Ag78/`, `GG-Ag2/`  
  Representative case folders for molecular or cluster systems.

- `project2_CD/projec1/xyz/`  
  Additional XYZ structure files for case studies.

> **Note**  
> The current folder names largely follow the internal development history of the project. They may be reorganized in future updates to provide a cleaner and more reproducible layout.

---

## Requirements

This project is primarily developed in **Python**.

Recommended environment:

- Python 3.9 or above
- NumPy
- SciPy
- pandas
- matplotlib
- scikit-learn
- Jupyter Notebook (optional)

Depending on the specific submodule or future updates, additional dependencies may be required.

---

## Installation

Clone the repository:

```bash
git clone git@github.com:ZXLab2005/mc-hg-psf-ecd.git
cd mc-hg-psf-ecd
```

Create and activate a recommended conda environment:

```bash
conda create -n mc_hg_psf_ecd python=3.10
conda activate mc_hg_psf_ecd
```

Install common dependencies:

```bash
pip install numpy scipy pandas matplotlib scikit-learn jupyter
```

If some modules inside the repository require additional packages, please install them according to your local workflow.

---

## Quick Start

At the current stage, this repository is still a research-development codebase rather than a fully packaged software release. A typical workflow may include:

1. Prepare molecular or cluster structure files
2. Load the corresponding dataset or descriptors
3. Run the relevant spectrum-related script(s)
4. Analyze predicted ECD spectra and mechanism-related outputs

A possible starting point is:

```text
project2/cdspecX.py
```

You may also explore representative case-study folders under:

```text
project2_CD/projec1/
```

Since this project is still being organized, some scripts may contain development-stage assumptions such as local paths or experiment-specific settings.

---

## Current Status

This repository is under active development.

The current public version mainly serves as an **initial research codebase and project archive**, which means it may still contain:

- development-stage folder naming
- experiment-specific scripts
- hard-coded paths
- incomplete documentation
- exploratory modules under continuous revision

We are gradually improving the structure, documentation, and reproducibility of the repository.

---

## Planned Improvements

We hope to improve the repository in future updates by adding:

- a cleaner folder organization
- `requirements.txt` or `environment.yml`
- minimal reproducible examples
- more detailed usage instructions
- figure examples and expected outputs
- model/workflow diagrams
- citation information for related papers or preprints

---

## Research Attitude

We would like to emphasize that this repository is shared in a spirit of **openness, learning, and academic exchange**.

Although we have tried to organize the current materials as carefully as possible, we are fully aware that the project is still evolving and that there may remain many aspects that can be improved, including code structure, documentation clarity, naming conventions, and reproducibility.

We therefore sincerely welcome:

- constructive suggestions
- questions and discussions
- corrections of possible mistakes
- ideas for collaboration or extension

We remain humble about the current stage of this work and are very open to feedback from researchers, students, and developers with related interests.

---

## Contact

If you have any questions, suggestions, or would like to discuss related ideas, please feel free to contact:

**207436834@qq.com**

We sincerely welcome communication, feedback, and academic discussion.

---

## Citation

If you find this repository helpful in your research, please consider citing the related paper or preprint when available.

Citation information will be added in future updates.

---

## License

This project is released under the **MIT License**.

---

## Acknowledgment

Thank you for your interest in this project.

We hope this repository can serve as a useful starting point for further exploration at the intersection of **machine learning**, **spectroscopy**, and **electronic-structure-based scientific modeling**.
