# CSCI4220U Project - License Plate Recognition
## Introduction
The following project was created by Eyaz Rehman and Rishabh Patel. 
As the population of the world increases, the number of people requiring transportation will also increase. 
Many of them will turn to cars, motorcycles or some other form of transportation, resulting in an increasing number of vehicles on the road every year. 
Due to this, LPR systems will continue to have an ever-increasing and important role in our modern society. 
Using the LPR system will help solve a wide range of problems, including toll route and parking admission, reporting driving violations, violent crimes, and stolen vehicles, therefore helping law enforcement increase their overall efficiency.

## Installation
- Clone repository
- Ensure Python 3 is installed
- Ensure dependencies are installed (`pip install -r requirements.txt`). PyTorch may have to be installed separately, instructions can be found at their site.

## Running
- To run the demo, use `python .\read_directory.py -d .\data\crnn.pth -p .\data\images\demo\`
- To run the evaluation, do:
    - Firstly, `python .\plate_gen.py -f .\data\fonts\driver_gothic.ttf -i 10 -d .\data\images\generated_plates\`
    - Secondly, `python .\test_accuracy.py -d .\data\crnn.pth -p .\data\images\generated_plates`

## Authors
- Eyaz Rehman ([GitHub](http://github.com/Imposter))
- Rishabh Patel ([GitHub](http://github.com/rpatel97))
