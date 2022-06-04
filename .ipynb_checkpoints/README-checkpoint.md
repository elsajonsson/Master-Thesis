# Master Thesis Project: A comparison between fully-supervised and self-supervised deep learning methods for tumour classification in digital pathology data

By Elsa Jonsson, May 2022

This master thesis project was conducted together with AstraZeneca. It uses embeddings created from the AstraZeneca DIME Pipeline found in https://doi.org/10.1101/2021.09.20.461088. The project trains simple binary classifiers on embedding representations of Whole Slide Image patches from the Camelyon16 dataset. The purpose of the project is then to compare the results to fully-supervised methods proposed in the Camelyon16 challange. The finished report is included in this respitory. 

This respitory also contains the implementation of the project. The DIME embeddings are not included aswell as the finished models produced in the project. 

To run the project, a file containing 512-dimension embeddings is required as a substitute for the DIME embeddings used in this project. These embeddings should be stored in a numpy file together with a class numpy file where each embeddings annotation is at the same index as in the embedding file. Change the folder destination in the source-files to the location of your embeddings. 

A third pickle file was provided by AstraZeneca containing relevant information about the embedding and the patch it represents. If you do not have a file such as this, remove that code in the source files.  

Then the steps are as follows:

1. Download the project and open up a linux-based shell in the folder

2. Spin up the Docker-container by running the shellscript: sh run_notebook.sh

3. Click the link provided in your shell and open the jupyter notebook



