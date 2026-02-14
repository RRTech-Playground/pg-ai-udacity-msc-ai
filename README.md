# Udacity MSc AI

## Setup

### Python Environment
We use [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

You can install it with brew. However, somtimes a local venv is enough.
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
conda env create --file environment.yml -n udacity-ai-env    # Create an environment from a yaml file
```

```bash
# Day two
conda upgrade conda # However, this does not work, miniconda?
conda update -n base -c defaults conda # This does not work either

conda upgrade --all # Does work and updates installed packages, but doesn't update conda
```

Sometimes we want a local venv [venv](https://towardsdatascience.com/virtual-environments-104c62d48c54).

#### Working with venv

Create the virtual environment
```bash
python3 -m venv venv/       # Creates an environment called venv/
```

Activate the virtual environment
```bash
source venv/bin/activate
```

To clean up and go back to the system-wide Python just deactivate the current environment
```bash
deactivate
```

To write an environment to a file and install it from a file
```bash
pip freeze > requirements.txt
```

```bash
pip install -r requirements.txt
```

#### Working with uv

And if we use uv, here is a short how to.

```bash
brew install uv
```

##### Key Features

- **Performance**: It is 10–100x faster than pip and other traditional managers. It uses a global cache and parallel downloads to resolve and install dependencies in milliseconds.
- **Unified Tooling**: It replaces pip, pip-tools, pipx, poetry, pyenv, and virtualenv with a single binary.
- **Python Version Management**: It can install and switch between different Python versions (e.g., uv python install 3.12) without needing separate tools like pyenv.
- **Reproducibility**: It uses a universal uv.lock file to ensure the exact same environment is recreated across different machines.
- **Script Support**: You can run single-file scripts with metadata-defined dependencies using uv run script.py.

##### Common Commands

```bash
uv init  # Initialize project	
uv add <package_name>  # Add a package	
uv run <file.py>  # Run a script	
uv sync # Sync dependencies	 
uvx <tool_name> (similar to pipx)  # Install a tool	

# Python Version Management
uv python install <version>  # Install Python
uv python pin <version>  # Switch to a specific Python version
uv python list  # List installed Python versions
uv python uninstall <version>  # Uninstall Python
```

### Install dependencies (with conda)
Now we need to install the needed dependencies

With conda you don't need to install python, PyEnv, pip, venv and all the tooling.
Python itself comes with Miniconda and conda does a great job combining whats needed.
Following this post to learn why https://codesolid.com/conda-vs-pip
However, it's ok if you have python and all the other toos installed.

#### Python
```bash
conda install python=3.13    # To install Python
python --version
```

#### Jupyter
```bash
conda install jupyterlab
#conda install notebook

conda install nb_conda  # Only needed to manage conda environments in Jupyter
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
conda install jupyterlab-sos -c conda-forge
pip install jupyterlab-sos

conda install sos-notebook -c conda-forge 
pip install sos-notebook

# Workflows
conda install jupyter_contrib_nbextensions -c conda-forge
conda install sos -c conda-forge
conda install sos-pbs -c conda-forge
conda install jupyterlab-sos -c conda-forge
conda install sos-notebook -c conda-forge
conda install sos-papermill -c conda-forge
conda install sos-r sos-python -c conda-forge  # SOS Kernels for R and Python

# List kernels
jupyter kernelspec list  # To check the kernels we have
jupyter kernelspec uninstall <unwanted-kernel>

# Link the SOS magic into the notebook based on the existing kernels
python -m sos_notebook.install 
```

#### virtualenv or pipenv
If you are using virtualenv or pipenv, you might need to remove the sos kernel installed globally to install sos for the particular python interpreter of the virtual env.

```bash
jupyter kernelspec remove sos
python -m sos_notebook.install
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
Or just use a kotlin notebook in intellij.


### Markdown

https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet

### Math in LaTeX
To use Mathematical formulas with LaTeX in Markdown with the double $$ signs

```markdown
$$ y = \frac{a}{b+c} $$
```
A great [tutorial](https://latex-tutorial.com/tutorials/amsmath/) on using LaTeX to create math expressions.


### Magic Keywords

Timer
```markdown
%timeit  [comment]: for execution on one line
%%timeit [comment]: for execution on multiple lines
```

Inline mathplotlib
```markdown
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
```

Use `pdb` to debug code. http://ipython.readthedocs.io/en/stable/interactive/magics.html
```markdown
%pdb [comment]: Turn it on in the cell before or the exectuing cell
```
More info https://docs.python.org/3/library/pdb.html 
and about magic functions https://ipython.readthedocs.io/en/stable/interactive/magics.html

### Convert Notebooks
A notebook is just json, its easy to convert it into HTML, LaTeX, PDF, WebPDF, Reveal.js HTML slideshow, Markdown, Ascii, reStructuredText, executable script, notebook etc.
```bash
pip install nbconvert 
jupyter nbconvert --to html mynotebook.ipynb
```

Learn more about nbconvert from the [documentation](https://nbconvert.readthedocs.io/en/latest/usage.html).

### Running Slideshows
You need to designate which cells are slides and the type of slide the cell will be. 
In the menu bar, click View > Cell Toolbar > Slideshow to bring up the slide cell menu on each cell.

To convert a notebook into a slideshow
```bash
jupyter nbconvert notebook.ipynb --to slides
```

Run the slideshow from a html server, in Jupyter directly
```bash
jupyter nbconvert notebook.ipynb --to slides --post serve
```

### Day2 Maintenance

To update pip
```bash
python3 -m pip install --upgrade pip
```
