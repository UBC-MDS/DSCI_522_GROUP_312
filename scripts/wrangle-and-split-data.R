# author: DSCI 522 Group 312
# date: 2020-01-22

"This script takes in a filepath to a CSV, performs relevant wrangling, 
and then splits the CSV into training and testing data, and then saves
training and testing data as CSVs to respective specified filepaths.

Usage: scripts/wrangle-and-split-data.R --filepath_in=<filepath_in> --filepath_out_train=<filepath_out_train> --filepath_out_test=<filepath_out_test>

Options:
--filepath_in=<filepath_in>   Path (including filename) to source data file (from load_data.py)
--filepath_out_train=<filepath_out_train>   Path (including filename) to where the training data should be saved
--filepath_out_test=<filepath_out_test>   Path (including filename) to where the testing data should be saved
" -> doc

# Libraries Required
library(tidyverse)
library(readr)
library(tidyr)
library(caret)
library(docopt)
library(checkmate)
library(testthat)

opt <- docopt(doc)

main <- function(filepath_in, filepath_out_train, filepath_out_test) {
  
  # load the data into this script
  data <- readr::read_csv(filepath_in,)
  
  # Complete Wrangling
  data <- data %>%
    drop_na() %>% 
    mutate(median_income = median_income * 10000) %>% 
    select(housing_median_age, total_rooms, total_bedrooms, population, households,
           median_income, ocean_proximity, latitude, longitude, median_house_value)
  
  # Split into training and testing data
  set.seed(522)
  trainindex <- createDataPartition(y = data$median_house_value, times = 1, p = 0.8, list = FALSE)
  train <- data[trainindex,]
  test <- data[-trainindex,]
  
  readr::write_csv(x = train, path = filepath_out_train)
  readr::write_csv(x = test, path = filepath_out_test)
}

test_files_created <- function() {
  test_that("There was an issue creating the Training file. Please try again.", {
    expect_equal(checkFileExists(x = filepath_out_train), TRUE)
    })
  
  test_that("There was an issue creating the Testing file. Please try again.", {
    expect_equal(checkFileExists(x = filepath_out_test), TRUE)
  })
}

test_files_created

main(opt[["--filepath_in"]], opt[["--filepath_out_train"]], opt[["--filepath_out_test"]])

# References
# Balla, Deepanshu. n.d. SPLITTING Data into Training and Test Sets with R. https://www.listendata.com/2015/02/splitting-data-into-training-and-test.html.
# 
# de Jonge, Edwin 2018. docopt: Command-Line Interface Specification Language. https://CRAN.R-project.org/package=docopt.
# 
# Kuhn, Max. 2020. Caret: Classification and Regression Training. https://CRAN.R-project.org/package=caret.
# 
# Lang, Michael 2017. checkmate: Fast Argument Checks for Defensive R Programming. https://journal.r-project.org/archive/2017/RJ-2017-028/index.html.
# 
# R Core Team. 2019. R: A Language and Environment for Statistical Computing. Vienna, Austria: R Foundation for Statistical Computing. https://www.R-project.org/.
# 
# Wickham, Hadley. 2011. testthat: Get Started with Testing. https://journal.r-project.org/archive/2011-1/RJournal_2011-1_Wickham.pdf.
# 
# Wickham, Hadley. 2017. Tidyverse: Easily Install and Load the ’Tidyverse’. https://CRAN.R-project.org/package=tidyverse.
# 
# Wickham, Hadley, and Lionel Henry. 2019. Tidyr: Tidy Messy Data. https://CRAN.R-project.org/package=tidyr.
# 
# Wickham, Hadley, Jim Hester, and Romain Francois. 2018. Readr: Read Rectangular Text Data. https://CRAN.R-project.org/package=readr.
