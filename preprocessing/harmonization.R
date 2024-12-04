# Create the specified file following a similar format for features screened by the post-harmonization-JMIM method.
# For the pre-harmonization method, remove shape features to avoid processing errors.
# Reorganize the final generated file and add correct column names and Label.
rm(list=ls())
gc()
getwd()
setwd("C:/Users/37427/Desktop/github/preprocessing/RFs_PET1_bin0.25")

library(sva)
library(readr)
library(dplyr)
library(tinyarray)
library(limma)

# read data
radiomics <- read_csv("post-harmonization-JMIM.csv")
batch <- read_csv("batch.csv")

#Combat
batch <- batch %>%
  mutate(batch = as.numeric(batch))
batch$batch <- as.factor(batch$batch)
radiomics_data <- radiomics[,-1] %>% mutate(across(everything(), as.numeric))
mod <- model.matrix(~Label, data=batch)
combat <- ComBat(dat = radiomics_data, batch = batch$batch, mod = mod)

#Limma
dat_matrix <- as.matrix(radiomics_data)
rownames(dat_matrix) <- radiomics$X1
batch <- batch %>%
  mutate(batch = as.numeric(batch))
batch_column <- batch$batch
design <- model.matrix(~0 + batch, data=batch)
limma <- removeBatchEffect(x = dat_matrix, batch=batch_column, design=design)

#Save data
combat_transposed <- t(combat)
batch$batch <- as.factor(batch$batch)
limma_transposed <- t(limma)
batch$batch <- as.factor(batch$batch)
write.csv(combat_transposed, "combat_corrected_data_transposed.csv", row.names = FALSE)
write.csv(limma_transposed, "limma_batch_corrected_data_transposed.csv", row.names = TRUE)