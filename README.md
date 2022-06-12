# **Basis Of Learning**
>    SnT Project -2022 

## **Getting Started**
### ***Installing Anaconda***
Getting Started with Python Anaconda Distribution 

Download Python Anaconda Distribution
> [Download Here](https://www.anaconda.com/download/#windows)

 Find the Anaconda Navigator application. 
 This is the go-to application for using all of the capabilities of Python Anaconda distribution.


### Managing Python Packages

> — Conda Installation
>>Select the Environments tab located on the left side of the application.
On the base environment, click on the play icon. Then, use the drop down menu to select Open Terminal.
If you need to install other packages, you can repeat these steps.
### ***Creating Virtual Envinorments***

Like many other languages Python requires a different version for different kind of applications. The application needs to run on a specific version of the language because it requires a certain dependency that is present in older versions but changes in newer versions. Virtual environments makes it easy to ideally separate different applications and avoid problems with different dependencies. Using virtual environment we can switch between both applications easily and get them running.

You can an environment using virtualenv, venv and conda. 

Conda command is preferred interface for managing installations and virtual environments with the Anaconda Python distribution.

Step 1: Check if conda is installed in your path.

Open up the anaconda command prompt and type

    conda -V

If the conda is successfully installed 
you can see the version of conda.


Step 2: Update the conda environment 

Enter the following in the anaconda prompt.

    conda update conda

Step 3: Set up the virtual environment

Type conda search “^python$”  to see the list of available python versions.

Now replace the envname with the name you want to give to your virtual environment and replace x.x with the python version you want to use.

    conda create -n envname python=x.x anaconda

Step 4: Activating the virtual environment

To see the list of all the available environments use command 

    conda info -e

To activate the virtual environment, enter the given command and replace your given environment name with envname.

    conda activate envname


 

 When conda environment is activated it modifies the PATH and shell variables points specifically to the isolated Python set-up you created.

 

Step 5: Installation of required packages to the virtual environment


Type the following command to install the additional packages to the environment and replace envname with the name of your environment.

    conda install -n yourenvname packagename

Step 6: Deactivating the virtual environment

To come out of the particular environment type the following command. The settings of the environment will remain as it is.

    conda deactivate


Step 7: Deletion of virtual environment

If you no longer require a virtual environment. Delete it using the following command and replace your environment name with envname

    conda remove -n envname -all 

### **Using Jupyter Notebooks**
**Launching the Jupyter Notebook:**

Jupyter Notebook is a web-based tool, so it requires your web browser to work.

 To open Jupyter Notebook, you can search for the application on your computer and select it. Alternatively, you can open it up from your terminal or command line with the following command:

    jupyter notebook


***Understanding the Jupyter Notebook Interface:***



How to navigate the Jupyter Notebook interface.

***The Notebook Name:***

 The name of your notebook is next to the Jupyter Notebook logo in the top left corner of your screen:

![image](https://miro.medium.com/max/1400/0%2ADoCDP8q8teCrKmb_)

By default, the notebook name is set to Untitled. You can change this name by clicking it.


Next to the name of your notebook, you’ll notice a checkpoint. Jupyter Notebook automatically saves when you edit the notebook cells. The checkpoint simply shows the last time the notebook was saved.

***The Menu Bar:***

![image](https://miro.medium.com/max/1400/0%2Aj5uNXY3arianSJi5)

The Menu Bar contains different menus that you’ll be using when working within the notebook.

**File:** can access file options. 

**Edit:**  To edit existing cells. 

**View:**  To personalize the appearance of your notebook

**Insert:** This is used to insert new cells

**Cell:**  used to run a cell after you have entered Python code. You can choose to run a specific cell, the cell below it, or the cell above it.

**Kernel:** The engine of the notebook.

options in the Kernel menu:

**Interrupt:** This is used to stop a cell that is currently running. 

**Restart:** This is used to restart the Kernel. 

***Restart & clear output:*** This restarts the Kernel and clears the output from the notebook. 

**Restart & run all:** This option will restart the Kernel and run all the cells in one go.

**Reconnect:** Sometimes a kernel will die for no particular reason. When this happens you can reconnect using this option.

**Shut down:** This is used to shut down the kernel and stop all processes

**Change kernel:** When you have more than one anaconda virtual environment installed, you can switch between different kernels.

**Widget:** Widgets are used to build interactive GUIs on your Notebook

**Help:** This is used to find resources when working with Jupyter Notebook, or the key libraries that come with it

The Command Icons

![command icon](https://miro.medium.com/max/1400/0%2ABOy_aTKrPw1a5t8F)

This is another critical area when working within a Jupyter Notebook. These icons allow you to perform certain functions on cells or on a single cell.

***Cells:***

![cell](https://miro.medium.com/max/1400/0%2AFm0RReHc5O4PkO_i)

 this is where you run Python code. .

Running Python Code

 you run it by clicking the run button from the command palette above.

There is a shortcut, use ‘*Shift+Enter*’ to run a cell and add a new cell below.

***Markdown:***

 Used to write plain text. 

You can Manuplate size of text using "*#*"

To bold text surround your text double asterixis (**).

For italics an underscore (_) before and after the words.

***Raw NB Convert:***

Used to convert the notebook file to another format.


## **Assignment-1**

>[Python Basics](Python.ipynb)

>[Pandas Basics](Pandas.ipynb)

>[Numpy Basics](Numpy.ipynb)

>[Matplotlib Basics](Matplotlib.ipynb)

Stoati c gradient
Bias
Run dataset
Normalize and non normalise
Joshua beningo DL


## Perceptron algorithm

The perceptron is a simple supervised machine learning algorithm and one of the earliest **neural network** architectures. It was introduced by Rosenblatt in the late 1950s. A perceptron represents a **binary linear classifier** that maps a set of training examples (of $d$ dimensional input vectors) onto binary output values using a $d-1$ dimensional hyperplane.

The perceptron as follows.

**Given:** 
- dataset $\{(\boldsymbol{x}^{(1)}, y^{(1)}), ..., (\boldsymbol{x}^{(m)}, y^{(m)})\}$
- with $\boldsymbol{x}^{(i)}$ being a d-$dimensional vector $
        
     $\boldsymbol{x}^i = (x^{(i)}_1, ..., x^{(i)}_d)$
- $y^{(i)}$ being a binary target variable, $y^{(i)} \in \{0,1\}$

The perceptron is a very simple neural network:
- it has a real-valued weight vector $\boldsymbol{w}= (w^{(1)}, ..., w^{(d)})$
- it has a real-valued bias $b$
- it uses the Heaviside step function as its activation function

* * *
A perceptron is trained using **gradient descent**. The training algorithm has different steps.

 In the beginning  the model parameters are initialized. 
 
 The other steps  are repeated for a specified number of training iterations or until the parameters have converged.
* * *
**Step 0:** Initialize the weight vector and bias with zeros (or small random values).


**Step 1:** Compute a linear combination of the input features and weights. This can be done in one step for all training examples, using vectorization and broadcasting:
$\boldsymbol{a} = \boldsymbol{X} \cdot \boldsymbol{w} + b$

where $\boldsymbol{X}$ is a matrix of shape $(n_{samples}, n_{features})$ that holds all training examples, and $\cdot$ denotes the dot product.


**Step 2:** Apply the Heaviside function, which returns binary values:

$\hat{y}^{(i)} = 1 \, if \, a^{(i)} \geq 0, \, else \, 0$


**Step 3:** Compute the weight updates using the perceptron learning rule

$$ \begin{equation}
\Delta \boldsymbol{w} = \eta \, \boldsymbol{X}^T \cdot \big(\boldsymbol{\hat{y}} - \boldsymbol{y} \big)
\end{equation} $$
$$ \Delta b = \eta \, \big(\boldsymbol{\hat{y}} - \boldsymbol{y} \big) $$

where $\eta$ is the learning rate.


**Step 4:** Update the weights and bias

$$\begin{equation}
\boldsymbol{w} = \boldsymbol{w} + \Delta \boldsymbol{w}
\end{equation}$$

$$
b = b  + \Delta b
$$
* * *

Here I made  an example  perceptron

# Literature Review on Stocastic Gradient Descent


<blockquote style="margin-right:auto; margin-left:auto; padding: 1em; margin:24px;">
    <strong>What's In a Name?</strong><br>
The <strong>gradient</strong> is a vector that tells us in what direction the weights need to go. More precisely, it tells us how to change the weights to make the loss change <em>fastest</em>. We call our process gradient <strong>descent</strong> because it uses the gradient to <em>descend</em> the loss curve towards a minimum. <strong>Stochastic</strong> means "determined by chance." Our training is <em>stochastic</em> because the minibatches are <em>random samples</em> from the dataset. And that's why it's called SGD!
</blockquote>

 Training the network means adjusting its weights in such a way that it can transform the features into the target. If we can successfully train a network to do that, its weights must represent in some way the relationship between those features and that target as expressed in the training data.

In addition to the training data, we need two more things:
- A "*loss function*" that measures how good the network's predictions are.
- An "*optimizer*" that can tell the network how to change its weights.

## The Loss Function ##


* The **loss function** measures the disparity between the the target's true value and the value the model predicts.

* Different problems call for different loss functions.A common loss function for regression problems is the **mean absolute error** or **MAE**. For each prediction `y_pred`, MAE measures the disparity from the true target `y_true` by an absolute difference `abs(y_true - y_pred)`.

The total MAE loss on a dataset is the mean of all these absolute differences.

<figure style="padding: 1em;">
<img src="https://i.imgur.com/VDcvkZN.png" width="5000" alt="A graph depicting error bars from data points to the fitted line..">
<figcaption style="textalign: center; font-style: italic"><center>The mean absolute error is the average length between the fitted curve and the data points.
</center></figcaption>
</figure>

Besides MAE, other loss functions you might see for regression problems are the mean-squared error (MSE) or the Huber loss (both available in Keras).

During training, the model will use the loss function as a guide for finding the correct values of its weights (lower loss is better). In other words, the loss function tells the network its objective.

##  The Optimizer - Stochastic Gradient Descent #

We've described the problem we want the network to solve, but now we need to say *how* to solve it. This is the job of the **optimizer**. The optimizer is an algorithm that adjusts the weights to minimize the loss.

Virtually all of the optimization algorithms used in deep learning belong to a family called **stochastic gradient descent**. They are iterative algorithms that train a network in steps. One **step** of training goes like this:
1. Sample some training data and run it through the network to make predictions.
2. Measure the loss between the predictions and the true values.
3. Finally, adjust the weights in a direction that makes the loss smaller.

Then just do this over and over until the loss is as small as you like (or until it won't decrease any further.)

<figure style="padding: 1em;">
<img src="https://i.imgur.com/rFI1tIk.gif" width="1600" alt="Fitting a line batch by batch. The loss decreases and the weights approach their true values.">
<figcaption style="textalign: center; font-style: italic"><center>Training a neural network with Stochastic Gradient Descent.
</center></figcaption>
</figure>

Each iteration's sample of training data is called a **minibatch** (or often just "batch"), while a complete round of the training data is called an **epoch**.

 The number of epochs you train for is how many times the network will see each training example.

The animation shows the linear model being trained with SGD. The pale red dots depict the entire training set, while the solid red dots are the minibatches. Every time SGD sees a new minibatch, it will shift the weights (`w` the slope and `b` the y-intercept) toward their correct values on that batch. Batch after batch, the line eventually converges to its best fit. You can see that the loss gets smaller as the weights get closer to their true values.

## Learning Rate and Batch Size ##

Notice that the line only makes a small shift in the direction of each batch (instead of moving all the way). The size of these shifts is determined by the **learning rate**. A smaller learning rate means the network needs to see more minibatches before its weights converge to their best values.

The learning rate and the size of the minibatches are the two parameters that have the largest effect on how the SGD training proceeds. Their interaction is often subtle and the right choice for these parameters isn't always obvious. (We'll explore these effects in the exercise.)

Fortunately, for most work it won't be necessary to do an extensive hyperparameter search to get satisfactory results. 

**Adam**  is an SGD algorithm that has an adaptive learning rate that makes it suitable for most problems without any parameter tuning (it is "self tuning", in a sense). Adam is a great general-purpose optimizer.

## Adding the Loss and Optimizer ##

After defining a model, you can add a loss function and optimizer with the model's `compile` method:

```
model.compile(
    optimizer="adam",
    loss="mae",
)
```

You can also access these directly through the Keras API.if you wanted to tune parameters.

