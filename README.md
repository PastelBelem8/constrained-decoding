# Constrained Decoding


## Setup

In this section, we assume you have successfully [installed Anaconda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) and can promptly use it to create a virtual environment in your machine. Run the command below to create the environment `constrained-dec` in your machine. This command will install all the dependencies under this virtual environment, including Python 3.9, Pytorch, and cudatoolkit version 11.3 (note: you might need to update the CUDA TOOLKIT version to work with your hardware).

```bash
$ conda env create -f requirements/env.yml
```

After running this command successfully (i.e., no errors) you can activate the environment using:

```bash
$ conda activate constrained-dec
```

To verify whether your installation was successful and that your Pytorch+CudaToolkit is working properly, you can run the following command:

```bash
$ python -c "import torch; assert torch.cuda.is_available(); torch.tensor([1, 2]).to('cuda'); print('Success!')"
```

## Using this project

TBD