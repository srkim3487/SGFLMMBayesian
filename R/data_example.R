#' Example Dataset: data_example
#' 
#' This dataset is used in the application section of the main paper and includes data from 1,054 counties in 12 Midwestern United States
#' 
#' @format A list with 4 elements:
#' \describe{
#' \item{\code{data_list}:}{A list of binary response variables. 
#'   The binary response for each county is defined as 1 if its poverty rate in 2020 exceeds the average poverty rate across all states, and 0 otherwise. 
#'   Each list element corresponds to a different state (12 total).}
#' \item{\code{X_list}:}{A list of centered functional covariates. 
#'   Each element contains annual unemployment rate data from 2000 to 2019 for counties within a given state (12 total).}
#' \item{\code{nbd_list}:}{A list of neighborhood structures, with each element corresponding to the spatial adjacency relationships among counties within a single state.}
#' \item{\code{nbd}:}{A unified neighborhood list combining all 12 states, obtained by collapsing \code{nbd_list} into a single spatial adjacency structure across all 1,054 counties.}
#' }
#'
#' @usage data(data_example)
#' @keywords datasets
"data_example"