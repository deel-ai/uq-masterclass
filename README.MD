# 🚀 Welcome to the **Uncertainty Quantification DEEL Master Class** !

Predictive uncertainty quantification in machine learning aims to assess the predictive error of models as precisely as possible, and in some cases, to calibrate this error according to specific statistical targets. Among various approaches, Conformal Prediction stands out due to its strong theoretical properties and ease of implementation and integration in production environments. 

This repository gathers materials from DEEL master class to acquire theoretical and practical knowledge about uncertainty quantification, and more specifically conformal prediction. Through these resources, you'll understand the key concepts, learn to apply them using PUNCC library across various examples, and develop the skills to interpret results and select suitable algorithms for real-world applications.

## 👨‍🎓 Tutorial Notebooks

This repository contains the tutorial notebooks for the following topics:

- [Part 1: Conformal Regression Tutorial](tutorial_notebooks/part1_regression_tuto.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/deel-ai/uq-masterclass/blob/main/tutorial_notebooks/part1_regression_tuto.ipynb) 


- [Part 2: Conformal Classification Tutorial](tutorial_notebooks/part2_classification_tuto.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/deel-ai/uq-masterclass/blob/main/tutorial_notebooks/part2_classification_tuto.ipynb) 


- [Part 3: Conformal Object Detection Tutorial](tutorial_notebooks/part3_objectdetection_tuto.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/deel-ai/uq-masterclass/blob/main/tutorial_notebooks/part3_objectdetection_tuto.ipynb) 

You can either work locally by cloning the project or open the notebooks following the colab links.

## 🐾 Installation

To get started with this tutorial, you have two options:


#### Option 1: Using Google Colab (Recommended)

If you have access to Google Colab, you can run this project directly in your web browser without needing to install anything locally. 

1. **Sign in to Google Colab**: Ensure you have a Google account to sign in.
2. **Open the Colab Notebook**: Open the Colab notebook links (see below).
3. **Install Dependencies in Colab**: The notebook will install all required packages in Colab’s environment.

This option is ideal if Colab is accessible to you.


#### Option 2: Local Installation (If Colab Access is Unavailable)

If access to Google Colab is restricted or unavailable, follow these steps to install the project locally:

1. **Clone the repository:**
```bash
git clone https://github.com/deel-ai/uq-masterclass.git
cd uq-masterclass
```
2. **Create a virtual environment:**
```bash
python -m virtualenv venv-masterclass
```
3. **Activate the virtual environment:**
    * On windows:
    ```bash
    venv-masterclass\Scripts\activate
    ```
    * On macOS/Linux:
    ```bash
    source venv-masterclass/bin/activate
    ```

4. **Install the required packages:**

```bash
pip install -r requirements.txt
```
5. **Add the virtual environment to Jupyter as a new kernel:**

```bash
python -m ipykernel install --user --name=venv-masterclass"
```

6. **Launch Jupyter Notebook:**

```bash
jupyter notebook
```

In Jupyter, you can select the newly created kernel Python (`venv-masterclass`) from the kernel options.


## 📖 Completed Notebooks

If you want to check your work, you can find all solutions in the **"completed_notebooks"** folder. Each file in this folder corresponds to the respective tutorial notebook. Feel free to review these solutions as you progress through the exercises.

- [Completed Conformal Regression Tutorial](completed_notebooks/part1_regression_solution.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/deel-ai/uq-masterclass/blob/main/completed_notebooks/part1_regression_solution.ipynb) 


- [Completed Conformal Classification Tutorial](completed_notebooks/part2_classification_solution.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/deel-ai/uq-masterclass/blob/main/completed_notebooks/part2_classification_solution.ipynb) 


- [Completed Conformal Object Detection Tutorial](completed_notebooks/part3_objectdetection_solution.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/deel-ai/uq-masterclass/blob/main/completed_notebooks/part3_objectdetection_solution.ipynb) 


## 🔗 Links
- [<img src="https://github.githubassets.com/images/icons/emoji/octocat.png" width=20> Puncc Github](https://github.com/deel-ai/puncc)
- [📘 Puncc Documentation](https://deel-ai.github.io/puncc/index.html)

## 🙏 Acknowledgments

<img align="right" src="https://www.irt-saintexupery.com/wp-content/uploads/logo-DEEL.png" width="25%">
This project received funding from the French ”Investing for the Future – PIA3” program within the Artificial and Natural Intelligence Toulouse Institute (ANITI). The authors gratefully acknowledge the support of the <a href="https://www.irt-saintexupery.com/deel/"> DEEL </a> project.
