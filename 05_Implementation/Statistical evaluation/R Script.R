library("Dict")
library("jsonlite")
library(dplyr)
library(lattice)

##
## Cortical thickness
##
# load datasets
setwd("C:/Users/vinze/OneDrive - Universitaet Bern/Master Studium/Master_Thesis/Diffusion_Lesions/05_Implementation/Statistical evaluation")
#filled_corth_th_file_path <- "csv/dl+direct/lesion_filled/result-thick.csv"
#not_filled_corth_th_file_path <- "csv/dl+direct/lesion_not_filled/result-thick.csv"
#filled_corth_th_file_path <- "csv/ants_deep_learning/lesion_filled/result-thick.csv"
#not_filled_corth_th_file_path <- "csv/ants_deep_learning/lesion_not_filled/result-thick.csv"
#filled_corth_th_file_path <- "csv/ants_old/lesion_filled/result-thick.csv"
#not_filled_corth_th_file_path <- "csv/ants_old/lesion_not_filled/result-thick.csv"
#filled_corth_th_file_path <- "csv/freesurfer/lesion_filled/result-thick.csv"
#not_filled_corth_th_file_path <- "csv/freesurfer/lesion_not_filled/result-thick.csv"
filled_corth_th_data <- read.csv(filled_corth_th_file_path)
not_filled_corth_th_data <- read.csv(not_filled_corth_th_file_path)

# Create a list and a dict
# The list contains all patiens, which have juxacortical lesions.
# The dict has the cortical region of interests as keys and as values the corresponding 
# patients, which have juxtacortical lesion inside the roi 
juxtacortical_patients=vector()
cortical_roi <- Dict$new(
  "new" = c(0)
)
cortical_roi$clear()
filepath="lesions_cortex_dil1.csv" 
con = file(filepath, "r")
while ( TRUE ) {
  line = readLines(con, n = 1)
  if ( length(line) == 0 ) {
    break
  }
  line_elements <- unlist(strsplit(gsub("-", ".",  line, fixed=TRUE), ", "))  
  patients <- line_elements[-1]
  
  # remove # for freesurfer csv 
  #for (i in 1:(length(patients))) { 
  #  patients[i] <- strsplit(patients[i], split="_")[[1]][1] 
  #  if (substr(patients[i], 2, 2) == '0'){
  #    patients[i] <- paste("p", substr(patients[i], 3, 4), sep="") 
  #  }
  #} 
  
  cortical_roi[line_elements[1]] <- patients 
  juxtacortical_patients <- append(juxtacortical_patients, patients)   
}
close(con) 
juxtacortical_patients <- unique(juxtacortical_patients)  


# Check for normal distribution
x_meanRhTh <- not_filled_corth_th_data$rh.MeanThickness 
y_meanRhTh <- filled_corth_th_data$rh.MeanThickness 
x_meanLhTh <- not_filled_corth_th_data$lh.MeanThickness
y_meanLhTh <- filled_corth_th_data$lh.MeanThickness
qqnorm(x_meanRhTh-y_meanRhTh) 
qqline(x_meanRhTh-y_meanRhTh) 
# Check for symmetry
stripplot(x_meanRhTh-y_meanRhTh)

# Differences in mean cortical thickness
boxplot(x_meanRhTh, y_meanRhTh, paired=TRUE)
boxplot(x_meanLhTh, y_meanLhTh, paired=TRUE)

# Absolute changes relative to the mean
mean_xy <- (x_meanRhTh + y_meanRhTh) / 2
changes <- (abs(x_meanRhTh - mean_xy) + abs(y_meanRhTh - mean_xy)) / (2*mean_xy)
rh_changes <- 100 * sum(changes) / length(changes)
mean_xy <- (x_meanLhTh + y_meanLhTh) / 2
changes <- (abs(x_meanLhTh - mean_xy) + abs(y_meanLhTh - mean_xy)) / (2*mean_xy)
lh_changes <- 100 * sum(changes) / length(changes)
global_absolute_changes_relative_to_the_mean = (rh_changes + lh_changes) / 2 
global_absolute_changes_relative_to_the_mean 

# Absolute changes relative to the mean without juxacortical lesion patients
x=not_filled_corth_th_data %>% filter(!(SUBJECT %in% juxtacortical_patients))
y=filled_corth_th_data %>% filter(!(SUBJECT %in% juxtacortical_patients))
mean_xy <- (x$rh.MeanThickness + y$rh.MeanThickness) / 2
changes <- (abs(x$rh.MeanThickness - mean_xy) + abs(y$rh.MeanThickness - mean_xy)) / (2*mean_xy)
rh_changes <- 100 * sum(changes) / length(changes) 
mean_xy <- (x$lh.MeanThickness + y$lh.MeanThickness) / 2
changes <- (abs(x$lh.MeanThickness - mean_xy) + abs(y$lh.MeanThickness - mean_xy)) / (2*mean_xy)
lh_changes <- 100 * sum(changes) / length(changes) 
global_absolute_changes_relative_to_the_mean = (rh_changes + lh_changes) / 2 
global_absolute_changes_relative_to_the_mean

# Absolute changes relative to the mean per ROI
rois <- colnames(filled_corth_th_data)[-c(1, 70, 71)]
changes <- vector()
for (roi in rois) {
  mean_xy <- (not_filled_corth_th_data[[roi]] + filled_corth_th_data[[roi]]) / 2
  change <- (abs(not_filled_corth_th_data[[roi]] - mean_xy) + abs(filled_corth_th_data[[roi]] - mean_xy)) / (2*mean_xy)
  changes <- append(changes, 100 * sum(change) / length(change)) 
}
write.table(data.frame(rois, changes), file = "roi_absolute_changes_relative_to_the_mean.txt", row.names = FALSE, col.names = FALSE, sep=",", quote = FALSE)
roi_absolute_changes_relative_to_the_mean <- sum(changes) / length(changes)
roi_absolute_changes_relative_to_the_mean

# Absolute changes relative to the mean per ROI without juxacortical lesion patients
rois <- colnames(filled_corth_th_data)[-c(1, 70, 71)]
changes <- vector()
for (roi in rois) {
  juxta_patients <- cortical_roi[roi]
  x=filled_corth_th_data %>% filter(!(SUBJECT %in% juxta_patients))
  y=not_filled_corth_th_data  %>% filter(!(SUBJECT %in% juxta_patients))
  if(nrow(x) == 0) next
  x[[roi]] - y[[roi]]
  mean_xy <- (x[[roi]] + y[[roi]]) / 2
  change <- (abs(x[[roi]] - mean_xy) + abs(y[[roi]] - mean_xy)) / (2*mean_xy)
  changes <- append(changes, 100 * sum(change) / length(change)) 
}   
roi_absolute_changes_relative_to_the_mean <- sum(changes) / length(changes)
roi_absolute_changes_relative_to_the_mean
