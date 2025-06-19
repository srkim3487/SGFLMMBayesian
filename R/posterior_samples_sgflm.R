#' @title Bayesian inference for spatial generalized functional linear model (SGFLM)
#'
#' @description 
#' Performs Bayesian inference for a spatial generalized functional linear model (SGFLM), 
#' using Metropolis-Hastings sampling with a random walk proposal distribution. 
#' The proposal variance can be fixed or adaptively tuned
#' during sampling to achieve efficient exploration of the posterior distribution.
#'
#' @param M An integer specifying the number of MCMC iterations to perform.
#'
#' @param initial A (p+2)-dimsional vector giving the initial values for the parameters 
#' (\eqn{\eta}, \eqn{\alpha}, trunctated coefficients \eqn{\beta_j=\int \beta(t) \phi_j(t) dt}, j=1,...,p).
#'
#' @param proposal_var A positive numeric value specifying the diagonal proposal 
#' variance used for the fixed effects (\eqn{\alpha} and \eqn{\beta_j}, j=1,...,p) in the random walk 
#' Metropolis-Hastings algorithm. 
#' 
#' @param autotune Logical; if \code{TRUE}, the sampler adaptively adjusts 
#' \code{proposal_var} based on the acceptance rate during sampling. The 
#' variance is increased if the acceptance rate is high (above 35\%) and decreased 
#' if the rate is too low (below 20\%). If \code{FALSE}, the proposal variance 
#' remains fixed throughout the sampling process.
#' 
#' @param M_auxz The number of full Gibbs sampling used to generate the auxiliary variable.
#' 
#' @param data An n-dimensional vector, representing the response variable.
#'
#' @param nbd A list of length n, whic contains the neighborhood structure (of class \code{nb}).
#'
#' @param X An n by T matrix of functional covariates. Each row represents one observed functional covariate.
#' 
#' @param t A T-dimensional vector containing densely time grid points on the interval \eqn{[0, 1]}, 
#' where the functions are evaluated.
#' This should match the time domain used when creating basis functions via the \code{CreateBasis} function 
#' from the \pkg{fdapace} package.
#' 
#' @param p An natural number indicating the truncation level.
#' 
#' @param basis_type A character string specifying the type of orthogonal basis \eqn{\phi_j, j=1,..., p}. 
#' Must be one of \code{"cos"}, \code{"sin"}, \code{"fourier"}, \code{"legendre01"}, or \code{"poly"}. 
#' These correspond to basis types supported by the \code{CreateBasis} function in the \pkg{fdapace} package.
#' The default is \code{"fourier"}.
#' 
#' @return A list with the following elements:
#' \describe{
#'  \item{samples}{A list containing posterior samples:
#'    \itemize{
#'       \item \code{eta}: An M by 1 matrix of posterior samples for the spatial dependence parameter.
#'       \item \code{alpha}: An M by 1 matrix of posterior samples for the intercept parameter.
#'       \item \code{beta}: An M by p matrix of posterior samples for the trunctated coefficients \eqn{\beta_j}, j=1,...,p.
#'    }
#'  }
#'   \item{diagnostics}{A list containing diagnostic information:
#'     \itemize{
#'      \item \code{final_proposal_var}: The final proposal variance used for the fixed effect.
#'       \item \code{accept_rate}: The acceptance rate for the fixed effect proposals. 
#'       \item \code{elapsed_time_sec}: The total elapsed computation time in seconds.
#'     }
#'   }
#' }
#' @seealso
#' \code{\link{posterior_samples_gflm}}
#' \code{\link{posterior_samples_gflmm}}
#' \code{\link{posterior_samples_sgflmm}}
#' 
#' @examples
#' 
#' library(SGFLMMBayesian)
#' 
#' # Load example data included in the package
#' data(data_example)
#' X_list <- data_example$X_list
#' data_list <- data_example$data_list
#' nbd <- data_example$nbd
#' 
#' X_mat <- Reduce("rbind", X_list)
#' data_vec <- unlist(data_list)
#' 
#' # simulate time point on [0, 1]
#' t <- seq(from = 0, to = 1, length.out = ncol(X_list[[1]]))
#' 
#' initial_values <- c(0.4, 0.1, 0.9, -0.6, 1.4)
#' res <- posterior_samples_sgflm(M=8000, initial=initial_values, 
#' proposal_var=0.005, autotune=FALSE, 
#' M_auxz = 20, 
#' data=data_vec, nbd = nbd,
#' X = X_mat, t=t, p=3, basis_type="fourier") 
#' 
#' # Example: compute the posterior mean of eta
#' mean(res$samples$eta[-(1:4000)])
#' 
#' @export
posterior_samples_sgflm <- function(M, initial, proposal_var, autotune, 
                                    M_auxz,
                                    data, nbd,
                                    X, t, p, basis_type="fourier") {
  
  basis  <- fdapace::CreateBasis(p, t, type = basis_type)
  Xstar <- (X %*% basis)/length(t)
  
  res <- rw_id(M, initial, 
               tune_fixed = proposal_var, 
               autotune,
               nbd, Xstar, data, p, M_auxz)
  
  
  
  list(
    samples = list(
      eta = res$eta_samples,
      alpha = res$alpha_samples,
      beta = res$betaj_samples
    ),
    diagnostics = list(
      final_proposal_var = res$tune_fixed,
      accept_rate = res$rate_fixed,
      time_sec = res$elapsed_time_sec
    )
  )
}
