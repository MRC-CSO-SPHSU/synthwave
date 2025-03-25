require(mice)
require(lattice)
require(dplyr)
require(arrow)

set.seed(123)

ind <- read_parquet("./adults_non_imputed_middle_fidelity.parquet")

ind[grepl("^(indicator_)", colnames(ind))] <- lapply(ind[grepl("^(indicator_)", colnames(ind))], as.logical)
ind[grepl("^(category_)", colnames(ind))] <- lapply(ind[grepl("^(category_)", colnames(ind))], as.factor)

ind["ordinal_person_age"] <- lapply(ind["ordinal_person_age"], factor, order=TRUE, levels=c('15-17', '18-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80-84', '85+'))

pred <- quickpred(ind, mincor = 0.01) # about 10 minutes
# current version of quickpred does remove complete columns from the list of vars to be predicted
# automatically

#to_predict <- names(which(colSums(is.na(ind)) > 0)) # contain NA values
#do_not_predict <- colnames(ind)[!colnames(ind) %in% to_predict]

#pred <- quickpred(ind, mincor = 0)
#pred[do_not_predict, ] <- 0

options(future.globals.maxSize=10485760000)

start_time <- Sys.time()
imp <- futuremice(ind,
                  parallelseed = 123,
                  n.core = 8,
                  visitSequence = "monotone",
                  m = 1,
                  maxit = 1,
                  method = "pmm",
                  pred = pred)
end_time <- Sys.time()
end_time - start_time

imputed_data <- complete(imp, "long")

# TODO converting from ordinal to integer/double increases the value by one
# TODO education is an ordinal variable
imputed_data <- cbind(ids, imputed_data)

write.csv(imputed_data, "out1.csv")
