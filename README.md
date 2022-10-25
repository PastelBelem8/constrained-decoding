# Constrained Decoding

What is constrained decoding? HuggingFace put together a good [blogpost](https://huggingface.co/blog/constrained-beam-search).

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


## Resources

- []()
- [generation_utils](https://github.com/huggingface/transformers/blob/927f654427833dbf1da03f0cc036eed66f1d2533/src/transformers/generation_utils.py#L2679)
- [generation_beam_search](https://github.com/huggingface/transformers/blob/main/src/transformers/generation_beam_search.py)
- Papers relevant for the HuggingFace implementation [Post and Vilar 2018](https://arxiv.org/abs/1804.06609), [Hu et al 2019](https://aclanthology.org/N19-1090/), [Li et al. 2021](https://arxiv.org/pdf/2107.09846.pdf).