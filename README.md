## Reproducibility
  
### Download this git repository and run local
The firts step is to clone this repository
 
~~~
git clone https://github.com/dhcryan/ecg_denoise
~~~

If you are using a Unix-like system such as Linux, MacOS, FreeBSD, etc open a console in the path of DeepFilter code, 
then execute the download_data.sh bash file. 

~~~
bash ./download_data.sh
~~~

The next step is to create the Conda environment using the provided environment.yml file. For this step you need the 
conda python package manage installed. In case you don't have it installed we recommend installing 
[Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) to avoid installing unnecessary Python 
packages. 

To create the conda environment run the following command:
~~~
conda env create -f environment.yaml
~~~

Then activate the Python environment you just created with the following command:

~~~
conda activate DeepFilter
~~~

Finally start the training and the experiments by running the command:

~~~
python main.py
~~~

This python script will train all the models, execute the experiments calculate the metrics and plot the result table 
and some figures.

~~~
python eval_final.py >> result.txt
~~~


If you have a Nvidia CUDA capable device for GPU acceleration this code will automatically use it (faster). Otherwise the 
training will be done in CPU (slower).   

