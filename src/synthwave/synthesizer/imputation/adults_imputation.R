require(mice)
require(lattice)
require(dplyr)
require(arrow)
require(argparser)

N_CORES_DEFAULT <- 128
MAXIT_DEFAULT <- 20
PROP_DEFAULT <- 1e-2


do_adults_imputation <- function(path.to.data, subset, n_cores, maxit) {

    set.seed(123)

    print("Getting parquet file...")
    adults.file <- "adults_non_imputed_middle_fidelity.parquet"
    path.to.file <- file.path(path.to.data, adults.file)
    ind <- read_parquet(path.to.file)
    print("Done!")


    print("Tidying data...")
    ind[grepl("^(indicator_)", colnames(ind))] <- lapply(ind[grepl("^(indicator_)", colnames(ind))], as.logical)
    ind[grepl("^(mlb_)", colnames(ind))] <- lapply(ind[grepl("^(mlb_)", colnames(ind))], as.logical)

    ind[grepl("^(category_)", colnames(ind))] <- lapply(ind[grepl("^(category_)", colnames(ind))], as.factor)

    ghq <- grepl("^(ordinal_person_ghq)", colnames(ind))
    ind[ghq] <- lapply(ind[ghq], factor, order=TRUE, levels=seq(min(ind[colnames(ind)[ghq]], na.rm = TRUE),
                                                                max(ind[colnames(ind)[ghq]], na.rm = TRUE)))
    # This way we can disregard any potential shifts in the variable; they all also have the same levels

    ind["ordinal_person_sf_1"] <- lapply(ind["ordinal_person_sf_1"], factor, order=TRUE, levels=c(5, 4, 3, 2, 1))

    ind["ordinal_person_sf_2a"] <- lapply(ind["ordinal_person_sf_2a"], factor, order=TRUE, levels=c(1, 2, 3))
    ind["ordinal_person_sf_2b"] <- lapply(ind["ordinal_person_sf_2b"], factor, order=TRUE, levels=c(1, 2, 3))
    ind["ordinal_person_sf_3a"] <- lapply(ind["ordinal_person_sf_3a"], factor, order=TRUE, levels=c(1, 2, 3, 4, 5))
    ind["ordinal_person_sf_3b"] <- lapply(ind["ordinal_person_sf_3b"], factor, order=TRUE, levels=c(1, 2, 3, 4, 5))
    ind["ordinal_person_sf_4a"] <- lapply(ind["ordinal_person_sf_4a"], factor, order=TRUE, levels=c(1, 2, 3, 4, 5))
    ind["ordinal_person_sf_4b"] <- lapply(ind["ordinal_person_sf_4b"], factor, order=TRUE, levels=c(1, 2, 3, 4, 5))
    ind["ordinal_person_sf_5"] <- lapply(ind["ordinal_person_sf_5"], factor, order=TRUE, levels=c(5, 4, 3, 2, 1))
    ind["ordinal_person_sf_6a"] <- lapply(ind["ordinal_person_sf_6a"], factor, order=TRUE, levels=c(5, 4, 3, 2, 1))
    ind["ordinal_person_sf_6b"] <- lapply(ind["ordinal_person_sf_6b"], factor, order=TRUE, levels=c(5, 4, 3, 2, 1))
    ind["ordinal_person_sf_6c"] <- lapply(ind["ordinal_person_sf_6c"], factor, order=TRUE, levels=c(1, 2, 3, 4, 5))
    ind["ordinal_person_sf_7"] <- lapply(ind["ordinal_person_sf_7"], factor, order=TRUE, levels=c(1, 2, 3, 4, 5))

    ind["ordinal_person_financial_situation"] <- lapply(ind["ordinal_person_financial_situation"], factor, order=TRUE, levels=c(5, 4, 3, 2, 1))
    ind["ordinal_person_life_satisfaction"] <- lapply(ind["ordinal_person_life_satisfaction"], factor, order=TRUE, levels=c(1, 2, 3, 4, 5, 6, 7))

    # NOTE for some methods values must be shifted to start from 0, converted to ordinals, processed, converted to int (!), and shifted back
    min_age <- min(ind["ordinal_person_age"], na.rm = TRUE)
    max_age <- max(ind["ordinal_person_age"], na.rm = TRUE)

    ind["ordinal_person_age"] <- lapply(ind["ordinal_person_age"],
                                        factor,
                                        order=TRUE,
                                        levels=seq(min_age, max_age))

    min_year <- min(ind["ordinal_household_year"], na.rm = TRUE)
    max_year <- max(ind["ordinal_household_year"], na.rm = TRUE)

    ind["ordinal_household_year"] <- lapply(ind["ordinal_household_year"],
                                            factor,
                                            order = TRUE,
                                            levels=seq(min_year, max_year))


    ind["total_individuals"] <- lapply(ind["total_individuals"], factor, order=TRUE, levels=seq(min(ind["total_individuals"]), max(ind["total_individuals"])))
    ind["total_children"] <- lapply(ind["total_children"], factor, order=TRUE, levels=seq(min(ind["total_children"]), max(ind["total_children"])))

    ind["has_partner"] <- lapply(ind["has_partner"], as.logical)
    print("Done!")


    if (subset == TRUE)
    {
        print("Subsetting data for testing...")
        print(paste0("Current size: ", nrow(ind), " by ", ncol(ind)))

        prop <- PROP_DEFAULT
        ind <- ind %>% slice_sample(prop=prop, replace=FALSE)
        print(paste0("Length after subsetting (", as.character(prop*100), "%): ", nrow(ind), " by ", ncol(ind)))
    }


    id_names <- grepl("^(id_)", colnames(ind))  # Boolean mask, not actual values
    ids <- ind[id_names]
    ind <- ind[ , !id_names]


    print("Doing quickpred...")
    pred <- quickpred(ind, mincor = 0.01) # about 10 minutes
    # current version of quickpred does remove complete columns from the list of vars to be predicted
    # automatically

    #to_predict <- names(which(colSums(is.na(ind)) > 0)) # contain NA values
    #do_not_predict <- colnames(ind)[!colnames(ind) %in% to_predict]

    #pred <- quickpred(ind, mincor = 0)
    #pred[do_not_predict, ] <- 0
    print("Done!")


    print("Doing imputation via MICE...")
    options(future.globals.maxSize=10485760000)

    start_time <- Sys.time()
    imp <- futuremice(ind,
                      parallelseed = 123,
                      n.core = n_cores,
                      visitSequence = "monotone",
                      m = 1,
                      maxit = maxit,
                      method = "pmm",
                      pred = pred)
    end_time <- Sys.time()
    end_time - start_time
    print("Done!")

    imputed.data <- complete(imp, "long")

    # TODO converting from ordinal to integer/double increases the value by one
    # TODO education is an ordinal variable
    imputed.data <- cbind(ids, imputed.data)


    print("Saving imputed data...")
    out.path <- file.path(path.to.data, "synthwave", "imputed")
    if (!dir.exists(out.path)) {
        dir.create(out.path, recursive=TRUE)
    }
    out.full <- file.path(out.path, "imputed_data.csv")
    write.csv(imputed.data, out.full)
    print(paste0("Saved to: ", out.full))
}


ap <- arg_parser("imputation_stage")
ap <- add_argument(ap, "path_to_data", default=NULL, help="Data source path")
ap <- add_argument(ap, "--subset", default=FALSE, help="Take tiny subset for testing")
ap <- add_argument(ap, "--n_cores", default=N_CORES_DEFAULT, help="Number of cores to use for imputation")
ap <- add_argument(ap, "--maxit", default=MAXIT_DEFAULT, help="Maximum number of iterations in imputation" )
args <- parse_args(ap)

path.to.data <- args$path_to_data
subset.data <- args$subset
n_cores <- args$n_cores
maxit <- args$maxit

paste0('Imputing parquet data in folder ', path.to.data)
do_adults_imputation(path.to.data, subset.data, n_cores, maxit)
print('Done!')
