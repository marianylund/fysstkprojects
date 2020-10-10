# FYS-STK3155 Applied Data Analysis and Machine Learning

## Project 1 

Each task from the project has its own python file. 
For example: task a) has `a)OLSFranke.py`
To run that, I would recommend running inside Project1 folder:
```
python -w ignore '.\a)OLSFranke.py'
```

In the different files, I have commented out the majority of code for simplicity. Code is separated in regions explaining the part of the task it is for.

It should show a progress bar when running time-consuming tasks, by the end it should show a plot and print out some info to the console.

### Report folder
Contains Latex code and pdf of the report

### PROJECT_SETUP.py
Contains some global variables, such as result path, seed for randomisation and if the figures should be saved or not

### DataFiles folder
Contains Terrain Data

### RegLib folder
The main regression module. See the relationship and overview of functions in the diagram:

![](Project1/Project1FysSTK.png)

### Results\FigureFiles folder
Contains figure and plots, the selected ones are added to the report