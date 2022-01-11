This is a basic code repository for Assignment 3.

The repository contains a basic model and a basic training and testing
procedure. It will work on the testing-platform (but it will not
perform well against adversarial examples). The goal of the project is
to train a new model that is as robust as possible.

# Basic usage

Install python dependencies with pip: 

    $ pip install -r requirements.txt

Test the basic model:

    $ ./model.py
    Testing with model from 'models/default_model.pth'. 
    Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ./data/cifar-10-python.tar.gz
    100.0%
    Extracting ./data/cifar-10-python.tar.gz to ./data/
    Model natural accuracy (test): 53.07

(Re)train the basic model:

Note: default training procedure does nothing, so you will have to implement it yourself in order to observe good results. 

    $ ./model.py --force-train
    Training model
    models/default_model.pth
    Files already downloaded and verified
    Starting training
    [1,   500] loss: 0.576
    [1,  1000] loss: 0.575
    ...

Train/test the basic model and store the weights to a different file:

    $ ./model.py --model-file models/mymodel.pth
    ...

Load the module project and test it as close as it will be tested on the testing plateform:

    $ ./test_project.py

Even safer: do it from a different directory:

    $ mkdir tmp
    $ cd /tmp
    $ ../test_project.py ../

# Modifying the project

You can modify anything inside this git repository, it will work as long as:

- it contains a `model.py` file in the root directory
- the `model.py` file contains a class called `Net` derived from `torch.nn.Module`
- the `Net` class has a function call `load_for_testing()` that initializes the model for testing (typically by setting the weights properly).  The default load_for_testing() loads and store weights from a model file, you will also need to make sure the repos contains a model file that can be loaded into the `Net` architecture using Net.load(model_file).
- You may modify this `README.md` file. 

# Before pushing

When you have made improvements your version of the git repository:

1. Add and commit every new/modified file to the git repository, including your model files in models/.(Check with `git status`) *DO NOT CHECK THE DATA IN PLEASE!!!!*
2. Run `test_project.py` and verify the default model file used by load_for_testing() is the model file that you actually want to use for testing on the platform. 
3. Push your last change

Note: If you want to avoid any problems, it is a good idea to make a local copy of your repos (with `git clone <repos> <repos-copy>`) and to test the project inside this local copy.

Good luck!
