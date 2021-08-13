#!interpreter
#C:\ProgramData\Anaconda3\python.exe
# -*- coding: utf-8 -*-


"""
{Water Quality Detection}
{B.Tech Project}
"""

__author__ = 'sabarish'
__copyright__ = 'Copyright 2021, Main Part'
__credits__ = ['EnergeaTechnolabs']
__license__ = 'GNU'
__version__ = '1.0'
__maintainer__ = 'energeatechsolutions'
__email__ = 'info@energiasolutions.in'
__status__ = 'Completed'

from utilities import *
import time
import csv
import numpy as np
from GUI import *
from yellowbrick.features import Rank1D
from yellowbrick.features import Rank2D
from yellowbrick.features import RadViz

if __name__ == "__main__":

    data_Frame = data_ReaderProcessor(filename="./Water_treatment_part1.csv")

    print(data_Frame)

    X,y,X_train,X_test,y_train,y_test = data_Splitter(data_Frame)

    feature_names = ['PH', 'Turbidity', 'TDS']

    target_name = 'status'

    X = data_Frame[feature_names]
    y = data_Frame[target_name]

    visualizer = Rank1D(features=feature_names, algorithm='shapiro')
    visualizer.fit(X, y)                
    visualizer.transform(X)             
    visualizer.poof()  

    visualizer = Rank2D(features=feature_names, algorithm='covariance') 
    visualizer.fit(X, y)                
    visualizer.transform(X)             
    visualizer.poof()

    features = feature_names
    classes = ['without Quality', 'with Quality']# Instantiate the visualizer
    visualizer = visualizer = RadViz(classes=classes, features=features,size = (800,300))
    visualizer.fit(X, y)      
    visualizer.transform(X)  
    visualizer.poof()

    classifier_1 = Classifier(X_train = X_train,X_test = X_test,y_train = y_train,y_test = y_test,debug=True,classifier_Type="1")
    clf = classifier_1.create_Classifier()
    classifier_1.training_Classifier()
    classifier_1.testing_Classifier()
    classifier_1.evaluation_Criteria()

    classifier_2 = Classifier(X_train = X_train,X_test = X_test,y_train = y_train,y_test = y_test,debug=True,classifier_Type="2")
    clf = classifier_2.create_Classifier()
    classifier_2.training_Classifier()
    classifier_2.testing_Classifier()
    classifier_2.evaluation_Criteria()
    
    classifier_3 = Classifier(X_train = X_train,X_test = X_test,y_train = y_train,y_test = y_test,debug=True,classifier_Type="3")
    clf = classifier_3.create_Classifier()
    classifier_3.training_Classifier()
    classifier_3.testing_Classifier()
    classifier_3.evaluation_Criteria()
    
    classifier_4 = Classifier(X_train = X_train,X_test = X_test,y_train = y_train,y_test = y_test,debug=True,classifier_Type="4")
    clf = classifier_4.create_Classifier()
    classifier_4.training_Classifier()
    classifier_4.testing_Classifier()
    classifier_4.evaluation_Criteria()

    classifier_5 = Classifier(X_train = X_train,X_test = X_test,y_train = y_train,y_test = y_test,debug=True,classifier_Type="5")
    clf = classifier_5.create_Classifier()
    classifier_5.training_Classifier()
    classifier_5.testing_Classifier()
    classifier_5.evaluation_Criteria()


    while True:

        event, values = window.read()
        
        if event == sg.WIN_CLOSED or event == 'Cancel': # if user closes window or clicks cancel
            break
            
        elif event == 'Predict':
            print("prediction on process")
            
            input_List = []
            input_List.append(float(values[0]))
            input_List.append(float(values[1]))
            input_List.append(float(values[2]))

            print(input_List)
            output = classifier_1.manual_Check(list=input_List)
            print(output)
            if output == 0:
                output = "Quality is Good "

            if output == 1:
                output = "No Quality"
            sg.popup("Prediction Output",output)

        
    
    
