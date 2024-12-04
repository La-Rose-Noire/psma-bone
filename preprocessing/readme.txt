This folder include feature data extracted independently by two physicians from all modalities with a two-week interval, along with the corresponding inter-class consistency coefficients (ICC) data. 
Features with an ICC greater than 0.7 were normalized using the z-score. 
Within each respective subfolders named "6-4ADASYN-1", the standardized data were then split into training and testing sets to generate "train.csv" and "test.csv".  
ADASYN was applied to the training set to synthesize minority class samples  to generate "train_ADASYN.csv".
One of the features in pairs with a Spearman correlation coefficient greater than 0.7 was removed to generate the final files "train_Spearman.csv" and "test_Spearman.csv", which were used for subsequent feature selection and machine learning modeling. 
Note that each modality's "train_Spearman.csv" and "test_Spearman.csv" in the "preprocessing" folder are renamed as "train.csv" and "test.csv" in the "FS-ML" folder.