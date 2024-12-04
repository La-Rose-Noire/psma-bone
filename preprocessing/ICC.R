if (!require("irr")) install.packages("irr")
library(irr)

# Read data (Using only CT data as an example, PET data and Baseline data operate similarly)
file1 <- read.csv("C:/Users/37427/Desktop/github/preprocessing/RFs_CT_bin50/1CTbin50.csv", header = TRUE)
file2 <- read.csv("C:/Users/37427/Desktop/github/preprocessing/RFs_CT_bin50/2CTbin50.csv", header = TRUE)

# Calculate ICC values for each feature
icc_values <- sapply(2:ncol(file1), function(i) {
  combined_data <- cbind(file1[, i], file2[, i])
  # Calculate ICC(2,1)
  icc_obj <- irr::icc(combined_data, model = "twoway", type = "agreement")
  icc_obj$value
})

# Convert ICC values to a data frame
icc_df <- data.frame(Feature = colnames(file1)[2:ncol(file1)], ICC = icc_values, check.names = FALSE)

# Write ICC results to CSV file
icc_matrix <- as.matrix(icc_df[, -1]) 
write.csv(data.frame(Feature = icc_df$Feature, icc_matrix), "C:/Users/37427/Desktop/github/preprocessing/RFs_CT_bin50/ICC-CTbin50.csv", row.names = FALSE)

# Filter features with ICC greater than 0.7
high_icc_indices <- which(icc_df$ICC > 0.7)

# Select these features from both files
selected_columns <- icc_df$Feature[high_icc_indices]
final_data <- file2[, c("Label", selected_columns)]

# Write the final dataset with high ICC features to a new CSV file
write.csv(final_data, "C:/Users/37427/Desktop/github/preprocessing/RFs_CT_bin50/HighICC-CTbin50.csv", row.names = FALSE)