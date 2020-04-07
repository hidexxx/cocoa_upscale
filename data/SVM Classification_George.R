
library(ggplot2)
library(e1071)

#======================================================================================
# 1.0 Load field dataset into R for Training and Validation
#======================================================================================
library(readxl) # Required for loading Excel files into R


Sample2 <- read_excel("~/Desktop/COCOA LANDSCAPE/Images to classify/training_VIS.xlsx") # Excel file loaded 
                                                            #and saved as object with name Sample2

Sample2$Class <- as.factor(Sample2$Class) # The Classification variable field converted to factors

str(Sample2)
Sample2


#======================================================================================
# 2.0 Split field data set into Training and Validation 
#======================================================================================

library(caTools) # Library required to split dataset

DSplit = sample.split(Sample2$Class, SplitRatio = 0.7) # Function created to split file into 
                                                       #training and test data 
                                                       #(NB: 0.7 means 70% for Training and 30% for validation)

ValidationData = subset(Sample2,DSplit ==FALSE) # Object created to store Validation dataset
TrainingData = subset(Sample2,DSplit ==TRUE) # Object created to store Training dataset

nrow(TrainingData)
nrow(ValidationData)

str(TrainingData)


#======================================================================================
# 3.0 Classification Using SVM 
#======================================================================================

require("e1071")  # Library required for SVM


SVM.model <- svm(formula = Class~., # Library required for SVM
                data = TrainingData, 
                type="C-classification", 
                kernel = "radial")

#SVM.model <- svm(formula = LULC ~., data = TrainingData, type="C-classification", kernel = "linear")
#SVM.model <- svm(formula = LULC ~., data = TrainingData, type="C-classification", kernel = "polynomial")
#SVM.model <- svm(formula = LULC ~., data = TrainingData, type="C-classification", kernel = "radial")
#SVM.model <- svm(formula = LULC ~., data = TrainingData, type="C-classification", kernel = "sigmoid")

summary (SVM.model)

#Accuracy Assessment of Classification with un-tuned svm model
ConfMatrix<- predict(SVM.model,ValidationData)
MatTab <- table(ConfMatrix, ValidationData$Class)
MatTab
#Overall Accuracy
OA<- sum(diag(MatTab))/sum(MatTab)
OA<- OA * 100
OA

#User Accuracy
UA <- (diag(MatTab)/rowSums(MatTab))*100
UA

#Producer Accuracy
PA <- (diag(MatTab)/colSums(MatTab))*100
PA

#======================================================================================
# 4.0 Tuning  SVM 
#======================================================================================

set.seed(123)

svmTune <- tune(svm, Class~., data = TrainingData, ranges = list(epsilon = seq(0,1,0.1), cost = 2^(2:9)))

#plot(svmTune)

#summary(svmTune)

#Choose best model

tuned_svm_model <- svmTune$best.model

summary(tuned_svm_model)




#======================================================================================
# 4.0 Accuracy Assessment of SVM Classification using validation data to generate confuxion matrix/Error Matrix
#======================================================================================
#Confussion/Error Metrix

ConfMatrix<- predict(tuned_svm_model,ValidationData) #Accuracy Assessment of Classification with tuned svm model
MatTab <- table(ConfMatrix, ValidationData$Class)
MatTab

#Overall Accuracy
OA<- sum(diag(MatTab))/sum(MatTab)
OA<- OA * 100
OA

#User Accuracy
UA <- (diag(MatTab)/rowSums(MatTab))*100
UA

#Producer Accuracy
PA <- (diag(MatTab)/colSums(MatTab))*100
PA

library(psych)

cohen.kappa(MatTab)
#======================================================================================
# 5.0 Load Satellite Image
#======================================================================================

library(sp)
library(raster)
library(rgdal)


MyImage <- brick("~/Desktop/COCOA LANDSCAPE/Images to classify/s2viss12.tif") # Load Image Data from Directory and Set as MyImage


names(MyImage)<-c("B1","B2","B3","B4","B5","B6") #SET NAME OF RASTER BANDS SAME AS COLUMN NAMES OF TRAINING DATASET
                                                 # NB: B1. B2 etc... represents image bands as used in training data set columns

MyImage  # Show MyImage Summary
#plotRGB(MyImage, 4,3,2, stretch = 'hist') 


#======================================================================================
# 6.0 Create Classification Map & Write output
#======================================================================================

ImageClasses <-factor(TrainingData$Class)

SVMImage <- predict(MyImage, tuned_svm_model, progress = "text", type = "ImageClasses") #Classify loaded Image using Tuned SVM model

#SVMImage <- predict(MyImage, SVM.model, progress = "text", type = "ImageClasses") #Classify loaded Image using  SVM model (not tuned)

SVMImage                #Display statistics of Classified Image 
plot (SVMImage)         #Plot Classified Image 

writeRaster(SVMImage, filename = ("Desktop/COCOA LANDSCAPE/NWN/SVMs1s2vis13"), #Write Classified Image to output folder as Tiff file 
            format = "GTiff", overwrite = T)
