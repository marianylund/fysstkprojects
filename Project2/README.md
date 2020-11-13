## Project 2

Each task from the project has its own python file. 
For example: task a) has `a)SGD.py`
To run that, I would recommend running inside Project2 folder:
```
python -w ignore '.\a)SGD.py'
```
It should start the training with the correct configurations, test the model with sklearn capability and show a plot of the run.

In those files you will also find the "Parameter search" and "Analysis of results" those are mostly commented out since they were mostly used to create the graphs and plots that can be found in the paper.

### Testing
As mentioned earlier, each file will runa test with sklearn. There is also a separate file that only contains a test of the accuracy functions. To run it:
```
python -W ignore '.\test_project2.py'
```

### Report folder
Contains Latex code and pdf of the report

### nnreg folder
The main new library, see how it works together in the diagram:
![](Project2/FysSTK-Project2.png)

### RegLib folder
Regression module mostly the same as in Project 1.

### Results\FigureFiles folder
Contains results of the training (checkpoints and best models), figure and plots, the selected ones are added to the report

### PROJECT_SETUP.py
Contains some global variables, such as result path, seed for randomisation and if the figures should be saved or not

### MNIST.py
File used by the nnreg.dataloader.py to download the MNIST dataset into DataFiles folder.

