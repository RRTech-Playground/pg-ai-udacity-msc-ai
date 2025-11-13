# Udacity MSc AI

## Setup

### Python Environment
We use [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

You can install it with brew
```bash
brew install miniconda
```
You should find the init snipped in the caveats. Should be someting like
```bash
conda init "$(basename "${SHELL}")"
```

#### Working with Conda

```bash
conda info     # Shows all info and pathes to conda
```
```bash
conda list    # Shows what's installed
```

Conda is using shared environments, you can creat new ones per project or reuse existing envs. We use
```bash
conda env list      # Shows available environments
```
```bash
conda create -n udacity-ai-env     # To create
```         
```bash
conda activate udacity-ai-env     # To activate an environment
``` 
```bash
conda deactivate    # To deactivate an environment
``` 
```bash
conda env remove -n udacity-ai-env    # To remove an environment
```

To save an environment to a file and load from a file
```bash
conda env export --file environment.yml    # Exports environment to a yaml file
``` 
```bash
conda env create --file environment.yml -n q-env    # Create an environment from a yaml file
```

### Install dependencies
Now we need to install the needed dependencies







---
Old


You don't need to install python, PyEnv, pip, venv and all the tooling.
Python itself comes with Miniconda and conda does a great job combining whats needed.
Following this post to learn why https://codesolid.com/conda-vs-pip

However, it's ok if you have python and all the other toos installed.





#### Jupyter
```bash
conda install jupyterlab
#conda install notebook
```
```bash
jupyter --version    # Shows what's installed of jupyter
```

#### AI
```bash
pip install gym  # OpenAI Gym 
```

#### Udacity
To successfully submit a project you need to have `.udacity-pa` folder.
```bash
pip install udacity-pa  # Udacity Project Assistant CLI
udacity submit  
```

#### Tools
```bash
pip install ipywidgets    
pip install rise pylatexenc    
```

#### Qiskit
```bash
pip install 'qiskit[visualization]'    # To use visualization functionality or Jupyter notebooks
pip install qiskit-ibm-runtime    
pip install qiskit-aer    
```

#### Create Qiskit kernel
We create a special qiskit kernel
```bash
python -m ipykernel install --user --name qiskit --display-name "Qiskit"   
```

#### Kotlin kernel
```bash
pip install kotlin-jupyter-kernel
```

#### SOS Multi-Kernel
To work with multiple kernels in the same Notebook we use SOS
```bash
conda install jupyter_contrib_nbextensions -c conda-forge
conda install sos -c conda-forge
conda install sos-pbs -c conda-forge
conda install jupyterlab-sos -c conda-forge
conda install sos-notebook -c conda-forge
conda install sos-papermill -c conda-forge
#conda install sos-r sos-python -c conda-forge  # SOS Kernels
jupyter kernelspec list  # To check the kernels we have
#jupyter kernelspec uninstall unwanted-kernel
python -m sos_notebook.install # Link the SOS magic into the notebook based on the existing kernels
```
Start a notebook to see if the dropdowns of the kernels are reflected.
Then continue with the [docs](https://vatlab.github.io/sos-docs/doc/user_guide/exchange_variable.html)
and find more languages [here](https://vatlab.github.io/sos-docs/running.html#Supported-Languages).

You can find more information about SOS [SoS Home](https://vatlab.github.io/sos-docs/) and [Documentation](https://vatlab.github.io/sos-docs/doc/user_guide/multi_kernel_notebook.html).
 
### Start Notebook
```bash
jupyter lab   # Start
#jupyter notebook   # Start
```

### Day2 Maintenance
tbc

