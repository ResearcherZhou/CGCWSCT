# CGCWSCT
The "Sequential Optimization" dataset released by the laboratory of Tianjin University. This is a dedicated dataset for the unidirectional grinding of slender silicon carbide ceramic tubes, which contains data such as grinding process parameters, target responses, and codes, and can be used for multi-objective parameter optimization in complex working conditions.
#Paper
This is a dataset applicable to the paper we are planning to publish. If this dataset is helpful for your research, please cite our paper.
#Dataset structure
The dataset contains all the data used for multi-objective optimization, mainly including the setting of grinding parameters and the target responses. All the data are stored in an independent CSV file. The data is saved in a CSV file named "data.CSV". The CSV file of the data contains 8 channels of data: the first four columns are variables: grinding wheel speed, guide wheel speed, grinding depth, and center height. The last four columns are target responses: roundness, cylindricity, roughness, and power.
#Codes
The code mainly includes the code for multi-objective optimization and the code for debugging the surrogate model. The code for debugging the surrogate model - GRF + SCR + RF. The code for multi-objective optimization - NSAG + GRF.
#Download
All the data can be downloaded from one of the following links
1.Baidu Netdisk: https:// Extraction code: sdum
