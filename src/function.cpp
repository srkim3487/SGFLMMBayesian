// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace RcppArmadillo;

// [[Rcpp::export]]
List ext_nb_to_list(List nbd_list, int g){
  List temp = nbd_list[g-1];
  int n = temp.length();
  List result(n);
  
  for (int i = 0; i < n; i++){
    result[i] = temp[i];
  }
  return result;
}

// [[Rcpp::export]]
arma::mat ext_list_to_mat(List temp, int g){
  arma::mat result = temp[g-1];
  return result;
}

// [[Rcpp::export]]
arma::vec ext_list_to_vec(List temp, int g){
  arma::vec result = as<NumericVector>(temp[g-1]);
  return result;
}


// [[Rcpp::export]]
arma::mat DiagMat(int n) {
  arma::mat mat(n, n, arma::fill::eye); // Initialize a square matrix of size n x n with diagonal elements as 1.0
  return mat;
}


// [[Rcpp::export]]
double dlaplaceArma(double x, double location, double scale, bool log = false) {
  
  double density = (1.0 / (2.0 * scale)) * std::exp(-std::abs(x - location) / scale);
  
  if (log) {
    density = std::log(density);
  }
  
  return density;
}

// [[Rcpp::export]]
double dnormArma(double x, double mean, double sd, bool log = false) {
  
  double exponent = -0.5 * std::pow((x - mean) / sd, 2);
  double density = (1.0 / (sd * std::sqrt(2.0 * M_PI))) * std::exp(exponent);
  
  if (log) {
    density = std::log(density);
  }
  
  return density;
}

// [[Rcpp::export]]
double dmvnormArma(arma::vec x, arma::vec mean, arma::mat sigma) {
  int n = x.n_elem;
  double logdet = sum(log(eig_sym(sigma)));
  double constant = -0.5 * n * log(2 * M_PI) - 0.5 * logdet;
  arma::vec x_minus_mean = x - mean;
  double exponent = -0.5 * as_scalar(x_minus_mean.t() * inv_sympd(sigma) * x_minus_mean);
  return constant + exponent;
}


// [[Rcpp::export]]
double dnormArmaVec(arma::vec betaj, double mu, double sigma, bool log = false) {
  arma::vec temp(betaj.size());
  
  for (arma::uword i = 0; i < betaj.size(); ++i) {
    temp(i) = dnormArma(betaj(i), mu, sigma, log);
  }
  
  return sum(temp);
}

// [[Rcpp::export]]
arma::mat mvrnormArma(int n, arma::vec mu, arma::mat sigma){
  int ncols = sigma.n_cols;
  arma::mat Y = arma::randn(n, ncols);
  return arma::repmat(mu, 1, n).t() + Y * arma::chol(sigma);
}


// [[Rcpp::export]]
double rtruncnormRcpp(int n, double a, double b, double mean, double sd){
  // Function that uses an R package
  Rcpp::Environment truncnormEnv = Rcpp::Environment::namespace_env("truncnorm");
  Rcpp::Function rtruncnorm = truncnormEnv["rtruncnorm"];   
  
  // Check if b is Inf, and if so, use a very large value
  if (std::isinf(b)) {
    b = std::numeric_limits<double>::max();
  }
  
  
  // Call the function and retrieve the result
  double result = as<double>(rtruncnorm(Rcpp::Named("n") = n,
                                        Rcpp::Named("a") = a,
                                        Rcpp::Named("b") = b,
                                        Rcpp::Named("mean") = mean,
                                        Rcpp::Named("sd") = sd));
  
  return result;
}

// [[Rcpp::export]]
double dinvgammaArma(double x, double shape, double scale, bool take_log) {
  double log_density = -shape * log(scale) - R::lgammafn(shape) + (-shape - 1) * log(x) - scale / x;
  return take_log ? log_density : exp(log_density);
}

// [[Rcpp::export]]
arma::vec rinvgammaArma(int n, double shape, double scale) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::gamma_distribution<double> gamma_distribution(shape, 1.0 / scale);
  
  arma::vec samples(n);
  for (int i = 0; i < n; i++) {
    samples(i) = 1.0 / gamma_distribution(gen);
  }
  
  return samples;
}

// [[Rcpp::export]]
List cell2nb_rcpp(int nrow, int ncol, int torus = 0){ //false = 0, true = 1
  int n = nrow * ncol;
  List nb(n);
  
  for (int i = 0; i < n; i++) {
    int row = i / ncol;
    int col = i % ncol;
    
    IntegerVector neighbors;
    
    // Top neighbor
    if (row > 0) {
      neighbors.push_back(i - ncol + 1);
    } else if (torus == 1) {
      neighbors.push_back((nrow - 1) * ncol + col + 1);
    }
    
    // Bottom neighbor
    if (row < nrow - 1) {
      neighbors.push_back(i + ncol + 1);
    } else if (torus == 1) {
      neighbors.push_back(col + 1);
    }
    
    // Left neighbor
    if (col > 0) {
      neighbors.push_back(i - 1 + 1);
    } else if (torus == 1) {
      neighbors.push_back(i + ncol - 1 + 1);
    }
    
    // Right neighbor
    if (col < ncol - 1) {
      neighbors.push_back(i + 1 + 1);
    } else if (torus == 1) {
      neighbors.push_back(i - ncol + 1 + 1);
    }
    
    // Sort the neighbor indices
    neighbors.sort();
    
    nb[i] = neighbors;
  }
  
  return nb;
}

// [[Rcpp::export]]
double s1s2_rcpp(List nbd_index, arma::vec& data){
  int lengthData = data.n_elem;
  
  arma::vec result(lengthData);
  
  for (int s1 = 0; s1 < lengthData; ++s1) {
    IntegerVector nbd_temp = nbd_index[s1];
    arma::uvec nbd_id = as<arma::uvec>(wrap(nbd_temp));
    nbd_id = nbd_id-1;
    
    arma::vec temp(nbd_id.size());

    for(arma::uword i=0; i < nbd_id.size(); ++i){
      arma::uword index = nbd_id[i];
      if(index > (unsigned int)s1){
        temp(i) = data(s1) * data(index);
      }else{
        temp(i) = 0;
      }
    }
    result(s1) = sum(temp);
  }
  return sum(result);
}



//[[Rcpp::export]]
arma::vec gibbsSampling_dd(List sub_nbd_index, 
                           arma::mat& sub_Xstar, 
                           double alpha, arma::vec betaj, double eta, double u_g, 
                           arma::vec ini_data, int M){
  
  // Rcpp::List sub_nbd_index = cell2nb_rcpp(sub_lattice, sub_lattice, 1);
  
  arma::vec logitkappa = alpha + (sub_Xstar * betaj) + u_g;
  arma::vec kappa = 1.0 / (1.0 + exp(-logitkappa));
  
  double n = sub_nbd_index.length();  // pow(sub_lattice, 2);
  arma::vec y = ini_data;
  arma::vec logitp(n);
  arma::vec prob(n);
  
  for (int j = 0; j < M; j++){
    for (int i = 0; i < n; i++){
      IntegerVector nbd_temp = sub_nbd_index[i];
      // nbd_temp = nbd_temp-1;
      arma::uvec nbd_id = as<arma::uvec>(wrap(nbd_temp));
      nbd_id = nbd_id-1;
      
      logitp(i) = logitkappa(i) + eta * sum(y(nbd_id) - kappa(nbd_id));
      prob(i) = exp(logitp(i)) / (1 + exp(logitp(i)));
      y(i) = R::rbinom(1, prob(i));
    }
  }
  return y;
}

//[[Rcpp::export]]
double neg_SAE_dd(List sub_nbd_index,
                  arma::mat& sub_Xstar, arma::vec sub_data,
                  double eta_current,
                  double alpha_current, arma::vec betaj_current, double u_g,
                  double s1s2) {
  
  arma::vec logitkappa = alpha_current + sub_Xstar * betaj_current + u_g;
  arma::vec kappa = 1.0 / (1.0 + exp(-logitkappa));
  
  arma::vec temp(kappa.n_elem);
  for (int i = 0; i < sub_nbd_index.size(); ++i) {
    IntegerVector nbd_temp = sub_nbd_index[i];
    arma::uvec nbd_id = as<arma::uvec>(wrap(nbd_temp));
    nbd_id = nbd_id-1;
    temp(i) = arma::accu(kappa(nbd_id));
  }
  
  arma::vec term1_temp(logitkappa - eta_current * temp);
  
  double term1 = arma::accu(sub_data % term1_temp);
  double term2 = eta_current * s1s2;
  
  double negpotential = term1 + term2;
  
  return negpotential;
}

//[[Rcpp::export]]
double u_DMH_dd(arma::mat subX, // sub_X_fun
                arma::vec sub_data,   // sub_data
                List sub_nbd_index,
               int p, arma::mat basis,
               double eta_current, double alpha_current,
               arma::vec betaj_current, double u_g_current,
               double sigmau_squ, 
               double s_vec,
               int M_auxz
){
  int m = subX.n_cols;
  
  // Calculate sub_nbd_index using the cell2nb_rcpp function
  // List sub_nbd_index = cell2nb_rcpp(sub_lattice, sub_lattice, 1);
  
  // Calculate subX
  // arma::uvec X_idx = find(lattice.col(2) == g);
  // arma::mat subX = X_fun.rows(X_idx);
  // arma::rowvec col_mean = mean(subX, 0);
  
  // Calculate sub_Xstar: Subtract column means using a for loop
  // arma::mat centered_subX(subX.n_rows, subX.n_cols);
  // for (arma::uword i = 0; i < subX.n_rows; ++i){
  //   for (arma::uword j = 0; j < subX.n_cols; ++j){
  //     centered_subX(i, j) = subX(i, j) - col_mean(j);
  //   }
  // }
  
  // Calculate sub_Xstar
  arma::mat sub_Xstar = (subX * basis) / m;
  // arma::mat sub_Xstar = (centered_subX * basis) / m;
  
  // Calculate s1s2_data
  // arma::vec sub_data = as<arma::vec>(data[g-1]);
  double s1s2_data = s1s2_rcpp(sub_nbd_index, sub_data);
  
  // DMH (a) proposal
  // double u_g_proposed = R::rnorm(0, sqrt(sigmau_squ));
  
  double u_g_proposed = R::rnorm(u_g_current, s_vec);
  
  // DMH (b)-1 Generate auxiliary variable using Gibbs sampling
  arma::vec aux_z = gibbsSampling_dd(sub_nbd_index, sub_Xstar, 
                                  alpha_current,
                                  betaj_current, eta_current, u_g_proposed, sub_data, M_auxz);
  
  // DMH (b)-2 Calculate s1s2_z
  double s1s2_z = s1s2_rcpp(sub_nbd_index, aux_z);
  
  // Step (c) Compute log(acceptance rate)  
  double log_acce = 
    R::dnorm(u_g_proposed, 0, sqrt(sigmau_squ), true) - R::dnorm(u_g_current, 0, sqrt(sigmau_squ), true) +
    neg_SAE_dd(sub_nbd_index, sub_Xstar, aux_z, eta_current, alpha_current, betaj_current, u_g_current, s1s2_z) -
    neg_SAE_dd(sub_nbd_index, sub_Xstar, sub_data, eta_current, alpha_current, betaj_current, u_g_current, s1s2_data) +
    neg_SAE_dd(sub_nbd_index, sub_Xstar, sub_data, eta_current, alpha_current, betaj_current, u_g_proposed, s1s2_data) -
    neg_SAE_dd(sub_nbd_index, sub_Xstar, aux_z, eta_current, alpha_current, betaj_current, u_g_proposed, s1s2_z);
  
  // Step (d) Update
  int idx = Rcpp::sample(NumericVector::create(1, 2), 1, true, NumericVector::create(std::min(1.0, exp(log_acce)), 1 - std::min(1.0, exp(log_acce))))[0];
  if (idx == 1) {
    u_g_current = u_g_proposed;
  }
  
  return u_g_current;
}





//[[Rcpp::export]]
double fixed_DMH_dd(arma::mat subX, // sub_X_function 
                    arma::vec sub_data, // sub_data
                   List sub_nbd_index, 
                   int p, arma::mat basis,
                   double eta_current, double eta_proposed,
                   double alpha_current, double alpha_proposed,
                   arma::vec betaj_current, arma::vec betaj_proposed,
                   double u_g_current, 
                   int M_auxz
){
  int m = subX.n_cols;
  
  // Calculate sub_nbd_index using the cell2nb_rcpp function
  // List sub_nbd_index = cell2nb_rcpp(sub_lattice, sub_lattice, 1);
  
  // Calculate subX
  // arma::uvec X_idx = find(lattice.col(2) == g);
  // arma::mat subX = X_fun.rows(X_idx);
  // arma::rowvec col_mean = mean(subX, 0);
  
  // Calculate sub_Xstar: Subtract column means using a for loop
  // arma::mat centered_subX(subX.n_rows, subX.n_cols);
  // for (arma::uword i = 0; i < subX.n_rows; ++i){
  //   for (arma::uword j = 0; j < subX.n_cols; ++j){
  //     centered_subX(i, j) = subX(i, j) - col_mean(j);
  //   }
  // }
  
  // Calculate sub_Xstar
  arma::mat sub_Xstar = (subX * basis) / m;
  
  // arma::mat sub_Xstar = (centered_subX * basis) / m;
  
  // Calculate s1s2_data
  // arma::vec sub_data = as<arma::vec>(data[g-1]);
  double s1s2_data = s1s2_rcpp(sub_nbd_index, sub_data);
  
  // DMH (b)-1 Generate auxiliary variable using Gibbs sampling
  arma::vec aux_z = gibbsSampling_dd(sub_nbd_index, sub_Xstar, 
                                  alpha_proposed,
                                  betaj_proposed, eta_proposed, u_g_current, sub_data, M_auxz);
  
  // DMH (b)-2 Calculate s1s2_z
  double s1s2_z = s1s2_rcpp(sub_nbd_index, aux_z);
  
  // Step (c) Compute log(acceptance rate)  
  double log_acce = 
    // dlaplaceArma(eta_proposed, 0, 1, true) - dlaplaceArma(eta_current, 0, 1, true) +
    // SumdnormArma(betaj_proposed, 0, 100, true) - SumdnormArma(betaj_current, 0, 100, true) +
    // R::dnorm(u_g_proposed, 0, 1000, true) - R::dnorm(u_g_current, 0, 1000, true) +
    neg_SAE_dd(sub_nbd_index, sub_Xstar, aux_z, eta_current, alpha_current, betaj_current, u_g_current, s1s2_z) -
    neg_SAE_dd(sub_nbd_index, sub_Xstar, sub_data, eta_current, alpha_current, betaj_current, u_g_current, s1s2_data) +
    neg_SAE_dd(sub_nbd_index, sub_Xstar, sub_data, eta_proposed, alpha_proposed, betaj_proposed, u_g_current, s1s2_data) -
    neg_SAE_dd(sub_nbd_index, sub_Xstar, aux_z, eta_proposed, alpha_proposed, betaj_proposed, u_g_current, s1s2_z);
  
   return log_acce;
}


//[[Rcpp::export]]
double ii_fun(arma::mat& Xstar, arma::vec data, 
                      double alpha,
                      arma::vec betaj
){
  arma::vec logitp = alpha + Xstar * betaj;
  arma::vec pHat = exp(logitp)/(1+exp(logitp));
  
  double result = sum(data % log(pHat) + (1 - data) % log(1-pHat));
  
  return result;
}



// [[Rcpp::export]]
arma::vec gibbsSampling_id(List nbd_index, arma::mat& Xstar, 
                           double alpha, arma::vec betaj, double eta,  
                           arma::vec ini_data, int M){
  
  arma::vec logitkappa = alpha + Xstar * betaj;
  arma::vec kappa = 1.0 / (1.0 + exp(-logitkappa));
  
  double n = Xstar.n_rows;
  arma::vec y = ini_data;
  arma::vec logitp(n);
  arma::vec prob(n);
  
  for (int j = 0; j < M; j++){
    for (int i = 0; i < n; i++){
      IntegerVector nbd_temp = nbd_index[i];
      // nbd_temp = nbd_temp-1;
      arma::uvec nbd_id = as<arma::uvec>(wrap(nbd_temp));
      nbd_id = nbd_id-1;
      
      logitp(i) = logitkappa(i) + eta * sum(y(nbd_id) - kappa(nbd_id));
      prob(i) = exp(logitp(i)) / (1 + exp(logitp(i)));
      y(i) = R::rbinom(1, prob(i));
    }
  }
  return y;
}


//[[Rcpp::export]]
double neg_SAE_id(List nbd_index,
                  arma::mat& Xstar, arma::vec data,
                  double eta_current,
                  double alpha_current, arma::vec betaj_current, 
                  double s1s2) {
  
  arma::vec logitkappa = alpha_current + Xstar * betaj_current;
  arma::vec kappa = 1.0 / (1.0 + exp(-logitkappa));
  
  arma::vec temp(kappa.n_elem);
  for (int i = 0; i < nbd_index.size(); ++i) {
    IntegerVector nbd_temp = nbd_index[i];
    arma::uvec nbd_id = as<arma::uvec>(wrap(nbd_temp));
    nbd_id = nbd_id-1;
    temp(i) = arma::accu(kappa(nbd_id));
  }
  
  arma::vec term1_temp(logitkappa - eta_current * temp);
  
  double term1 = arma::accu(data % term1_temp);
  double term2 = eta_current * s1s2;
  
  double negpotential = term1 + term2;
  
  return negpotential;
}



//[[Rcpp::export]]
double fixed_DMH_id(List nbd_index, 
                    arma::mat& Xstar, arma::vec data, 
                    double eta_current, double eta_proposed,
                    double alpha_current, double alpha_proposed,
                    arma::vec betaj_current, arma::vec betaj_proposed,
                    int M_auxz
){
  // Calculate s1s2_data
  double s1s2_data = s1s2_rcpp(nbd_index, data);
  
  // DMH (b)-1 Generate auxiliary variable using Gibbs sampling
  arma::vec aux_z = gibbsSampling_id(nbd_index, Xstar,
                                     alpha_proposed, betaj_proposed, eta_proposed, 
                                     data, M_auxz);
  
  // DMH (b)-2 Calculate s1s2_z
  double s1s2_z = s1s2_rcpp(nbd_index, aux_z);
  
  // Step (c) Compute log(acceptance rate)  
  double log_acce = 
    dlaplaceArma(eta_proposed, 0, 1, true) - dlaplaceArma(eta_current, 0, 1, true) +
    R::dnorm(alpha_proposed, 0, 100, 1) - R::dnorm(alpha_current, 0, 100, 1) +
    dnormArmaVec(betaj_proposed, 0, 100, 1) - dnormArmaVec(betaj_current, 0, 100, true) +
    neg_SAE_id(nbd_index, Xstar, aux_z, eta_current, alpha_current, betaj_current, s1s2_z) -
    neg_SAE_id(nbd_index, Xstar, data, eta_current, alpha_current, betaj_current, s1s2_data) +
    neg_SAE_id(nbd_index, Xstar, data, eta_proposed, alpha_proposed, betaj_proposed, s1s2_data) -
    neg_SAE_id(nbd_index, Xstar, aux_z, eta_proposed, alpha_proposed, betaj_proposed, s1s2_z);
  
  return log_acce;
}

//[[Rcpp::export]]
double fixed_di_fun(arma::mat subX, // sub_X_function
                    arma::vec sub_data, // sub_data
              arma::mat basis,
              double alpha,
              arma::vec betaj,
              double u_g
){

  int m = subX.n_cols;

  // Calculate sub_nbd_index using the cell2nb_rcpp function
  // List sub_nbd_index = cell2nb_rcpp(sub_lattice, sub_lattice, 1);

  // Calculate subX
  // arma::uvec X_idx = find(lattice.col(2) == g);
  // arma::mat subX = X_fun.rows(X_idx);
  // arma::rowvec col_mean = mean(subX, 0);

  // Calculate sub_Xstar: Subtract column means using a for loop
  // arma::mat centered_subX(subX.n_rows, subX.n_cols);
  // for (arma::uword i = 0; i < subX.n_rows; ++i){
  //   for (arma::uword j = 0; j < subX.n_cols; ++j){
  //     centered_subX(i, j) = subX(i, j) - col_mean(j);
  //   }
  // }

  // Calculate sub_Xstar
  arma::mat sub_Xstar = (subX * basis) / m;
  
  // arma::mat sub_Xstar = (centered_subX * basis) / m;

  // Calculate s1s2_data
  // arma::vec sub_data = as<arma::vec>(data[g-1]);

  // regression part
  arma::vec logitp = alpha + sub_Xstar * betaj + u_g;
  arma::vec pHat = exp(logitp)/(1+exp(logitp));

  double result = sum(sub_data % log(pHat) + (1 - sub_data) % log(1-pHat));

  return result;
}

//[[Rcpp::export]]
double u_MH_di(arma::mat subX, // sub_X_function
               arma::vec sub_data, // sub_data
                arma::mat basis,
                double alpha_current,
                arma::vec betaj_current, 
                double u_g_current,
                double sigmau_squ, 
                double s_vec,
                int M_auxz
){
  // int m = subX.n_cols;
  
  // Calculate sub_nbd_index using the cell2nb_rcpp function
  // List sub_nbd_index = cell2nb_rcpp(sub_lattice, sub_lattice, 1);
  
  // Calculate subX
  // arma::uvec X_idx = find(lattice.col(2) == g);
  // arma::mat subX = X_fun.rows(X_idx);
  // arma::rowvec col_mean = mean(subX, 0);
  
  // Calculate sub_Xstar: Subtract column means using a for loop
  // arma::mat centered_subX(subX.n_rows, subX.n_cols);
  // for (arma::uword i = 0; i < subX.n_rows; ++i){
  //   for (arma::uword j = 0; j < subX.n_cols; ++j){
  //     centered_subX(i, j) = subX(i, j) - col_mean(j);
  //   }
  // }
  
  // Calculate sub_Xstar
  // arma::mat sub_Xstar = (subX * basis) / m;
  
  // arma::mat sub_Xstar = (centered_subX * basis) / m;
  
  // Calculate s1s2_data
  // arma::vec sub_data = as<arma::vec>(data[g-1]);

  // MH (a) proposal
  // double u_g_proposed = R::rnorm(0, sqrt(sigmau_squ));
  
  double u_g_proposed = R::rnorm(u_g_current, s_vec);
  
  // MH (b) Compute log(acceptance rate) 
  double log_acce = fixed_di_fun(subX, sub_data, basis,
                                  alpha_current, betaj_current, u_g_proposed) -
                    fixed_di_fun(subX, sub_data, basis,
                                  alpha_current, betaj_current, u_g_current) +
                    R::dnorm(u_g_proposed, 0, sqrt(sigmau_squ), true) - 
                    R::dnorm(u_g_current, 0, sqrt(sigmau_squ), true);
  
  // Step (c) Update
  int idx = Rcpp::sample(NumericVector::create(1, 2), 1, true, NumericVector::create(std::min(1.0, exp(log_acce)), 1 - std::min(1.0, exp(log_acce))))[0];
  if (idx == 1) {
    u_g_current = u_g_proposed;
  }
  
  return u_g_current;
}


//[[Rcpp::export]]
List rw_dd(int M, arma::vec initial, 
           double tune_fixed, double tune_sigmau2, double tune_u, bool autotune, 
           List true_X, List data, List nbd_list,
           int p, arma::mat basis, int M_auxz,
           int G, double prior_a, double prior_b) {

  arma::vec eta_samples(M);
  arma::vec alpha_samples(M);
  arma::mat betaj_samples(M, p);
  arma::mat u_g_samples(M, G);
  arma::vec sigmau2_samples(M);
  
  double rate_fixed;
  arma::vec rate_u(G);
  double rate_sigmau2;
  
  // initial values
  eta_samples(0) = initial(0);
  alpha_samples(0) = initial(1);
  for (int j = 0; j < p; j++) {
    betaj_samples(0, j) = initial(j + 2);
  }
  for (int g = 0; g < G; g++) {
    u_g_samples(0, g) = initial(p + 2 + g);
  }
  sigmau2_samples(0) = initial(initial.n_elem - 1);
  
  // double rate_fixed = 0.0; // Initialize rate_fixed
  // double rate_u = 0.0; // Initialize rate_u
  // double rate_sigmau2 = 0.0; // Initialize rate_sigmau2
  
  auto start_time = std::chrono::high_resolution_clock::now();
  
  for (int i = 1; i < M; i++) {
    // (1) update fixed effect
    arma::vec mu_fixed(p + 2);
    mu_fixed.subvec(0, 0) = eta_samples(i - 1);
    mu_fixed.subvec(1, 1) = alpha_samples(i - 1);
    mu_fixed.subvec(2, p + 1) = betaj_samples.row(i - 1).t();
    
    arma::vec one_vec(p + 2, arma::fill::ones);
    arma::mat var_fixed = diagmat(one_vec * tune_fixed);
    
    arma::mat proposed = mvrnormArma(1, mu_fixed, var_fixed);
    
    double eta_proposed = proposed(0, 0);
    double alpha_proposed = proposed(0, 1);
    arma::vec betaj_proposed = proposed.submat(0, 2, 0, p + 1).t();

    arma::vec log_acce_temp(G);
    for (int g = 0; g < G; g++) {
      List sub_nbd = ext_nb_to_list(nbd_list, g+1);
      
      log_acce_temp(g) = fixed_DMH_dd(ext_list_to_mat(true_X, g+1), ext_list_to_vec(data, g+1), sub_nbd, p, basis,
                                      eta_samples(i-1), eta_proposed,
                                      alpha_samples(i-1), alpha_proposed,
                                      betaj_samples.row(i - 1).t(), betaj_proposed,
                                      u_g_samples(i - 1, g),
                                      M_auxz);
    }
    
    double log_acce = sum(log_acce_temp) +
      dlaplaceArma(eta_proposed, 0, 1, 1) - dlaplaceArma(eta_samples(i-1), 0, 1, 1) +
      R::dnorm(alpha_proposed, 0, 100, true) - R::dnorm(alpha_samples(i-1), 0, 100, true) +
      dnormArmaVec(betaj_proposed, 0, 100, true) -
      dnormArmaVec(betaj_samples.row(i - 1).t(), 0, 100, true);

    int idx = Rcpp::sample(NumericVector::create(1, 2), 1, true, NumericVector::create(std::min(1.0, exp(log_acce)), 1 - std::min(1.0, exp(log_acce))))[0];
    if (idx == 1) {
      eta_samples(i) = eta_proposed;
      alpha_samples(i) = alpha_proposed;
      betaj_samples.row(i) = betaj_proposed.t();
    } else {
      eta_samples(i) = eta_samples(i - 1);
      alpha_samples(i) = alpha_samples(i - 1);
      betaj_samples.row(i) = betaj_samples.row(i - 1);
    }

    arma::uvec unique_eta = arma::find_unique(eta_samples.head(i + 1));
    rate_fixed = static_cast<double>(unique_eta.n_elem) / (i + 1);
    if (autotune && rate_fixed > 0.35) {
      tune_fixed *= 1.1;
    } else if (autotune && rate_fixed < 0.2) {
      tune_fixed /= 1.1;
    }

    // (2a) propose: sigmau_squ
    double sigmau2_proposed = rtruncnormRcpp(1, 0, std::numeric_limits<double>::infinity(), sigmau2_samples(i - 1), tune_sigmau2);

    // (2b) update random effect u_g using DMH
    for (int g = 0; g < G; g++) {
      List sub_nbd = ext_nb_to_list(nbd_list, g+1);
      
      u_g_samples(i, g) = u_DMH_dd(ext_list_to_mat(true_X, g+1), ext_list_to_vec(data, g+1), sub_nbd, p, basis,
                                               eta_samples(i), alpha_samples(i),
                                               betaj_samples.row(i).t(), u_g_samples(i - 1, g),
                                               sigmau2_proposed, tune_u, M_auxz);
    }
  
    for (int g = 0; g < G; g++) {
      arma::uvec unique_u_g = arma::find_unique(u_g_samples.col(g).head(i + 1));
      rate_u[g] = static_cast<double>(unique_u_g.n_elem) / (i + 1);
    }

    if (autotune && max(rate_u) > 0.35) {
      tune_u *= 1.1;
    } else if (autotune && min(rate_u) < 0.2) {
      tune_u /= 1.1;
    }
    
    // (2c) update sigmau_squ: standard MH
    double log_rate_sigmau2 = dnormArmaVec(u_g_samples.row(i).t(), 0, sigmau2_proposed, true) -
      dnormArmaVec(u_g_samples.row(i).t(), 0, sigmau2_samples(i - 1), true) +
      dinvgammaArma(sigmau2_proposed, prior_a, prior_b, true) - dinvgammaArma(sigmau2_samples(i - 1), prior_a, prior_b, true);

    int idx_sigmau2 = Rcpp::sample(NumericVector::create(1, 2), 1, true, NumericVector::create(std::min(1.0, exp(log_rate_sigmau2)), 1 - std::min(1.0, exp(log_rate_sigmau2))))[0];
    if (idx_sigmau2 == 1) {
      sigmau2_samples(i) = sigmau2_proposed;
    } else {
      sigmau2_samples(i) = sigmau2_samples(i - 1);
    }

    arma::uvec unique_sigmau2 = arma::find_unique(sigmau2_samples.head(i + 1));
    rate_sigmau2 = static_cast<double>(unique_sigmau2.n_elem) / (i + 1);

    if (autotune && rate_sigmau2 > 0.35) {
      tune_sigmau2 *= 1.1;
    } else if (autotune && rate_sigmau2 < 0.2) {
      tune_sigmau2 /= 1.1;
    }
  }
  
  // Record the end time
  auto end_time = std::chrono::high_resolution_clock::now();
  
  // Calculate the elapsed time in seconds
  auto elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
  
  // return proposed;
  return List::create(
    Named("eta_samples") = eta_samples,
    Named("alpha_samples") = alpha_samples,
    Named("betaj_samples") = betaj_samples,
    Named("u_g_samples") = u_g_samples,
    Named("sigmau2_samples") = sigmau2_samples,
    Named("rate_fixed") = rate_fixed,
    Named("tune_fixed") = tune_fixed,
    Named("rate_u") = rate_u,
    Named("tune_u") = tune_u,
    Named("rate_sigmau2") = rate_sigmau2,
    Named("tune_sigmau2") = tune_sigmau2,
    Named("elapsed_time_sec") = elapsed_seconds.count() // Include elapsed time in seconds
  );
}


//[[Rcpp::export]]
List rw_di(int M, arma::vec initial, 
           double tune_fixed, double tune_sigmau2, double tune_u, bool autotune, 
           List true_X, List data, 
           int p, arma::mat basis, int M_auxz,
           int G, double prior_a, double prior_b) {
  
  arma::vec alpha_samples(M);
  arma::mat betaj_samples(M, p);
  arma::mat u_g_samples(M, G);
  arma::vec sigmau2_samples(M);
  
  double rate_fixed;
  arma::vec rate_u(G);
  double rate_sigmau2;
  
  // initial values
  alpha_samples(0) = initial(0);
  for (int j = 0; j < p; j++) {
    betaj_samples(0, j) = initial(j + 1);
  }
  for (int g = 0; g < G; g++) {
    u_g_samples(0, g) = initial(p + 1 + g);
  }
  sigmau2_samples(0) = initial(initial.n_elem - 1);
  
  auto start_time = std::chrono::high_resolution_clock::now();
  
  for (int i = 1; i < M; i++) {
    // (1) update fixed effect
    arma::vec mu_fixed(p + 1);
    mu_fixed.subvec(0, 0) = alpha_samples(i - 1);
    mu_fixed.subvec(1, p) = betaj_samples.row(i - 1).t();
    
    arma::vec one_vec(p + 1, arma::fill::ones);
    arma::mat var_fixed = diagmat(one_vec * tune_fixed);
    
    arma::mat proposed = mvrnormArma(1, mu_fixed, var_fixed);
    
    double alpha_proposed = proposed(0, 0);
    arma::vec betaj_proposed = proposed.submat(0, 1, 0, p).t();
    
    arma::vec log_acce_temp(G);
    for (int g = 0; g < G; g++) {
      log_acce_temp(g) = fixed_di_fun(ext_list_to_mat(true_X, g+1), ext_list_to_vec(data, g+1), basis,
                    alpha_proposed, betaj_proposed, u_g_samples(i - 1, g)) -
                      fixed_di_fun(ext_list_to_mat(true_X, g+1), ext_list_to_vec(data, g+1), basis,
                                   alpha_samples(i-1), betaj_samples.row(i - 1).t(), u_g_samples(i - 1, g));
    }
    
    double log_acce = sum(log_acce_temp) +
      R::dnorm(alpha_proposed, 0, 100, true) - R::dnorm(alpha_samples(i-1), 0, 100, true) +
      dnormArmaVec(betaj_proposed, 0, 100, true) -
      dnormArmaVec(betaj_samples.row(i - 1).t(), 0, 100, true);
    
    int idx = Rcpp::sample(NumericVector::create(1, 2), 1, true, NumericVector::create(std::min(1.0, exp(log_acce)), 1 - std::min(1.0, exp(log_acce))))[0];
    if (idx == 1) {
      alpha_samples(i) = alpha_proposed;
      betaj_samples.row(i) = betaj_proposed.t();
    } else {
      alpha_samples(i) = alpha_samples(i - 1);
      betaj_samples.row(i) = betaj_samples.row(i - 1);
    }
    
    arma::uvec unique_alpha = arma::find_unique(alpha_samples.head(i + 1));
    rate_fixed = static_cast<double>(unique_alpha.n_elem) / (i + 1);
    if (autotune && rate_fixed > 0.35) {
      tune_fixed *= 1.1;
    } else if (autotune && rate_fixed < 0.2) {
      tune_fixed /= 1.1;
    }
    
    // (2a) propose: sigmau_squ
    double sigmau2_proposed = rtruncnormRcpp(1, 0, std::numeric_limits<double>::infinity(), sigmau2_samples(i - 1), tune_sigmau2);
    
    // (2b) update random effect u_g using DMH
    for (int g = 0; g < G; g++) {
      u_g_samples(i, g) = u_MH_di(ext_list_to_mat(true_X, g+1), ext_list_to_vec(data, g+1), basis,
                                  alpha_samples(i), betaj_samples.row(i).t(), u_g_samples(i - 1, g),
                                  sigmau2_proposed, tune_u, M_auxz);
    }
    
    for (int g = 0; g < G; g++) {
      arma::uvec unique_u_g = arma::find_unique(u_g_samples.col(g).head(i + 1));
      rate_u[g] = static_cast<double>(unique_u_g.n_elem) / (i + 1);
    }
    
    if (autotune && max(rate_u) > 0.35) {
      tune_u *= 1.1;
    } else if (autotune && min(rate_u) < 0.2) {
      tune_u /= 1.1;
    }
    
    // (2c) update sigmau_squ: standard MH
    double log_rate_sigmau2 = dnormArmaVec(u_g_samples.row(i).t(), 0, sigmau2_proposed, true) -
      dnormArmaVec(u_g_samples.row(i).t(), 0, sigmau2_samples(i - 1), true) +
      dinvgammaArma(sigmau2_proposed, prior_a, prior_b, true) - dinvgammaArma(sigmau2_samples(i - 1), prior_a, prior_b, true);
    
    int idx_sigmau2 = Rcpp::sample(NumericVector::create(1, 2), 1, true, NumericVector::create(std::min(1.0, exp(log_rate_sigmau2)), 1 - std::min(1.0, exp(log_rate_sigmau2))))[0];
    if (idx_sigmau2 == 1) {
      sigmau2_samples(i) = sigmau2_proposed;
    } else {
      sigmau2_samples(i) = sigmau2_samples(i - 1);
    }
    
    arma::uvec unique_sigmau2 = arma::find_unique(sigmau2_samples.head(i + 1));
    rate_sigmau2 = static_cast<double>(unique_sigmau2.n_elem) / (i + 1);
    
    if (autotune && rate_sigmau2 > 0.35) {
      tune_sigmau2 *= 1.1;
    } else if (autotune && rate_sigmau2 < 0.2) {
      tune_sigmau2 /= 1.1;
    }
  }
  
  // Record the end time
  auto end_time = std::chrono::high_resolution_clock::now();
  
  // Calculate the elapsed time in seconds
  auto elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
  
  // return proposed;
  return List::create(
    Named("alpha_samples") = alpha_samples,
    Named("betaj_samples") = betaj_samples,
    Named("u_g_samples") = u_g_samples,
    Named("sigmau2_samples") = sigmau2_samples,
    Named("rate_fixed") = rate_fixed,
    Named("tune_fixed") = tune_fixed,
    Named("rate_u") = rate_u,
    Named("tune_u") = tune_u,
    Named("rate_sigmau2") = rate_sigmau2,
    Named("tune_sigmau2") = tune_sigmau2,
    Named("elapsed_time_sec") = elapsed_seconds.count() // Include elapsed time in seconds
  );
}



//[[Rcpp::export]]
List rw_id(int M, arma::vec initial,
           double tune_fixed, bool autotune, 
           List nbd_index, arma::mat& Xstar, arma::vec data_vec, 
           int p, int M_auxz
           ) {
  
  arma::vec eta_samples(M);
  arma::vec alpha_samples(M);
  arma::mat betaj_samples(M, p);
  
  double rate_fixed;
  
  // initial values
  eta_samples(0) = initial(0);
  alpha_samples(0) = initial(1);
  for (int j = 0; j < p; j++) {
    betaj_samples(0, j) = initial(j + 2);
  }
  
  auto start_time = std::chrono::high_resolution_clock::now();
  
  for (int i = 1; i < M; i++) {
    // (1) update fixed effect
    arma::vec mu_fixed(p + 2);
    mu_fixed.subvec(0, 0) = eta_samples(i - 1);
    mu_fixed.subvec(1, 1) = alpha_samples(i - 1);
    mu_fixed.subvec(2, p + 1) = betaj_samples.row(i - 1).t();
    
    arma::vec one_vec(p + 2, arma::fill::ones);
    arma::mat var_fixed = diagmat(one_vec * tune_fixed);
    
    arma::mat proposed = mvrnormArma(1, mu_fixed, var_fixed);
    
    double eta_proposed = proposed(0, 0);
    double alpha_proposed = proposed(0, 1);
    arma::vec betaj_proposed = proposed.submat(0, 2, 0, p + 1).t();
    
    double log_acce = fixed_DMH_id(nbd_index, Xstar, data_vec,
                                   eta_samples(i-1), eta_proposed,
                                   alpha_samples(i-1), alpha_proposed,
                                   betaj_samples.row(i - 1).t(), betaj_proposed,
                                   M_auxz);
    
    int idx = Rcpp::sample(NumericVector::create(1, 2), 1, true, NumericVector::create(std::min(1.0, exp(log_acce)), 1 - std::min(1.0, exp(log_acce))))[0];
    if (idx == 1) {
      eta_samples(i) = eta_proposed;
      alpha_samples(i) = alpha_proposed;
      betaj_samples.row(i) = betaj_proposed.t();
    } else {
      eta_samples(i) = eta_samples(i - 1);
      alpha_samples(i) = alpha_samples(i - 1);
      betaj_samples.row(i) = betaj_samples.row(i - 1);
    }
    
    arma::uvec unique_eta = arma::find_unique(eta_samples.head(i + 1));
    rate_fixed = static_cast<double>(unique_eta.n_elem) / (i + 1);
    if (autotune && rate_fixed > 0.35) {
      tune_fixed *= 1.1;
    } else if (autotune && rate_fixed < 0.2) {
      tune_fixed /= 1.1;
    }
  }
  
  // Record the end time
  auto end_time = std::chrono::high_resolution_clock::now();
  
  // Calculate the elapsed time in seconds
  auto elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
  
  // return proposed;
  return List::create(
    Named("eta_samples") = eta_samples,
    Named("alpha_samples") = alpha_samples,
    Named("betaj_samples") = betaj_samples,
    Named("rate_fixed") = rate_fixed,
    Named("tune_fixed") = tune_fixed,
    Named("elapsed_time_sec") = elapsed_seconds.count() // Include elapsed time in seconds
  );
}


//[[Rcpp::export]]
List rw_ii(int M, arma::vec initial,
           double tune_fixed, bool autotune, 
           arma::mat& Xstar, arma::vec data_vec, 
           int p
) {
  
  arma::vec alpha_samples(M);
  arma::mat betaj_samples(M, p);
  
  double rate_fixed;
  
  // initial values
  alpha_samples(0) = initial(0);
  for (int j = 0; j < p; j++) {
    betaj_samples(0, j) = initial(j + 1);
  }
  
  auto start_time = std::chrono::high_resolution_clock::now();
  
  for (int i = 1; i < M; i++) {
    // (1) update fixed effect
    arma::vec mu_fixed(p + 1);
    mu_fixed.subvec(0, 0) = alpha_samples(i - 1);
    mu_fixed.subvec(1, p) = betaj_samples.row(i - 1).t();
    
    arma::vec one_vec(p + 1, arma::fill::ones);
    arma::mat var_fixed = diagmat(one_vec * tune_fixed);
    
    arma::mat proposed = mvrnormArma(1, mu_fixed, var_fixed);
    
    double alpha_proposed = proposed(0, 0);
    arma::vec betaj_proposed = proposed.submat(0, 1, 0, p).t();
    
    double log_acce = ii_fun(Xstar, data_vec, alpha_proposed, betaj_proposed) -
      ii_fun(Xstar, data_vec, alpha_samples(i-1), betaj_samples.row(i - 1).t()) +
      dnormArma(alpha_proposed, 0, 100, 1) - dnormArma(alpha_samples(i-1), 0, 100, 1) +
      dnormArmaVec(betaj_proposed, 0, 100, true) -
      dnormArmaVec(betaj_samples.row(i - 1).t(), 0, 100, true);
    
    int idx = Rcpp::sample(NumericVector::create(1, 2), 1, true, NumericVector::create(std::min(1.0, exp(log_acce)), 1 - std::min(1.0, exp(log_acce))))[0];
    if (idx == 1) {
      alpha_samples(i) = alpha_proposed;
      betaj_samples.row(i) = betaj_proposed.t();
    } else {
      alpha_samples(i) = alpha_samples(i - 1);
      betaj_samples.row(i) = betaj_samples.row(i - 1);
    }
    
    arma::uvec unique_alpha = arma::find_unique(alpha_samples.head(i + 1));
    rate_fixed = static_cast<double>(unique_alpha.n_elem) / (i + 1);
    if (autotune && rate_fixed > 0.35) {
      tune_fixed *= 1.1;
    } else if (autotune && rate_fixed < 0.2) {
      tune_fixed /= 1.1;
    }
  }
  
  // Record the end time
  auto end_time = std::chrono::high_resolution_clock::now();
  
  // Calculate the elapsed time in seconds
  auto elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
  
  // return proposed;
  return List::create(
    Named("alpha_samples") = alpha_samples,
    Named("betaj_samples") = betaj_samples,
    Named("rate_fixed") = rate_fixed,
    Named("tune_fixed") = tune_fixed,
    Named("elapsed_time_sec") = elapsed_seconds.count() // Include elapsed time in seconds
  );
}





// [[Rcpp::export]]
arma::mat varArma(double tau, double rho, arma::mat Dv, arma::mat nbd_mat) {
  // Calculate the variance matrix
  arma::mat var = tau * arma::inv(Dv - rho * nbd_mat);
  
  return var;
}




// [[Rcpp::export]]
List estX_fun_k(int M, arma::vec Xstar_trun, arma::mat Dv, arma::mat nbd_mat, 
              arma::vec initial, double S_tau, double S_rho, bool autotune) {
  
  auto start_time = std::chrono::high_resolution_clock::now();
  
  // Initialize 
  arma::vec tau_samples(M);
  arma::vec rho_samples(M);
  
  double rate_tau;
  double rate_rho;

  int n = Xstar_trun.n_elem;
  
  tau_samples(0) = initial(0);
  rho_samples(0) = initial(1);
  
    for (int m = 1; m < M; m++) {
      // tau
      double tau_proposed = rtruncnormRcpp(1, 0, std::numeric_limits<double>::infinity(), tau_samples(m - 1), S_tau);
      
      arma::mat var_proposed_tau = varArma(tau_proposed, rho_samples(m - 1), Dv, nbd_mat);
      arma::mat var_current_tau = varArma(tau_samples(m - 1), rho_samples(m - 1), Dv, nbd_mat);
      
      double acce_rate_tau = dmvnormArma(Xstar_trun, arma::zeros(n), var_proposed_tau) -
        dmvnormArma(Xstar_trun, arma::zeros(n), var_current_tau);
      
      int idx_tau = Rcpp::sample(NumericVector::create(1, 2), 1, true, NumericVector::create(std::min(1.0, exp(acce_rate_tau)), 1 - std::min(1.0, exp(acce_rate_tau))))[0];
      if (idx_tau == 1) {
        tau_samples(m) = tau_proposed;
      } else {
        tau_samples(m) = tau_samples(m - 1);
      }
      
      arma::uvec unique_tau = arma::find_unique(tau_samples.head(m + 1));
      rate_tau = static_cast<double>(unique_tau.n_elem) / (m + 1);
      if (autotune && rate_tau > 0.4) {
        S_tau *= 1.1;
      } else if (autotune && rate_tau < 0.2) {
        S_tau /= 1.1;
      }
      
      // rho
      double rho_proposed = rtruncnormRcpp(1, 0, 1, rho_samples(m - 1), S_rho);
      
      arma::mat var_proposed_rho = varArma(tau_samples(m), rho_proposed, Dv, nbd_mat);
      arma::mat var_current_rho = varArma(tau_samples(m), rho_samples(m - 1), Dv, nbd_mat);
      
      double accep_rate_rho = dmvnormArma(Xstar_trun, arma::zeros(n), var_proposed_rho) -
        dmvnormArma(Xstar_trun, arma::zeros(n), var_current_rho);
      
      int idx_rho = Rcpp::sample(NumericVector::create(1, 2), 1, true, NumericVector::create(std::min(1.0, exp(accep_rate_rho)), 1 - std::min(1.0, exp(accep_rate_rho))))[0];
      if (idx_rho == 1) {
        rho_samples(m) = rho_proposed;
      } else {
        rho_samples(m) = rho_samples(m - 1);
      }
      
      arma::uvec unique_rho = arma::find_unique(rho_samples.head(m + 1));
      rate_rho = static_cast<double>(unique_rho.n_elem) / (m + 1);
      if (autotune && rate_rho > 0.4) {
        S_rho *= 1.1;
      } else if (autotune && rate_rho < 0.2) {
        S_rho /= 1.1;
      }
    }
  
  // Record the end time
  auto end_time = std::chrono::high_resolution_clock::now();
  
  // Calculate the elapsed time in seconds
  auto elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
  
  // return proposed;
  return List::create(
    Named("tau_samples") = tau_samples,
    Named("rho_samples") = rho_samples,
    Named("rate_tau") = rate_tau,
    Named("S_tau") = S_tau,
    Named("rate_rho") = rate_rho,
    Named("S_rho") = S_rho,
    Named("elapsed_time_sec") = elapsed_seconds.count() // Include elapsed time in seconds
  );
}


// MH Function for truncated functional covariate
// [[Rcpp::export]]
List estX_fun_k_w_mu(int M, arma::vec Xstar_trun, arma::mat Dv, arma::mat nbd_mat,
                     arma::vec initial, double S_tau, double S_rho, bool autotune) {

  auto start_time = std::chrono::high_resolution_clock::now();

  // Initialize
  arma::vec tau_samples(M);
  arma::vec rho_samples(M);
  arma::vec delta_samples(M);
  arma::vec mu_samples(M-1);

  double rate_tau;
  double rate_rho;

  int n = Xstar_trun.n_elem;

  tau_samples(0) = initial(0);
  rho_samples(0) = initial(1);
  delta_samples(0) = initial(2);

  for (int m = 1; m < M; m++) {
    // mu
    arma::vec one_vec = arma::ones<arma::vec>(n);
    arma::mat Sigma_k = tau_samples(m-1) * arma::inv(Dv - rho_samples(m-1) * nbd_mat);
    double sigma_k = 1.0 / arma::as_scalar(arma::as_scalar(one_vec.t() * arma::inv(Sigma_k) * one_vec));
    double mu_mle = sigma_k * arma::as_scalar(one_vec.t() * arma::inv(Sigma_k) * Xstar_trun);
    double mu = mu_mle / (1.0 + sigma_k / delta_samples(m-1));
    double sigma = sigma_k / (1.0 + sigma_k / delta_samples(m-1));

    mu_samples(m-1) = arma::randn() * std::sqrt(sigma) + mu;


    // tau
    double tau_proposed = rtruncnormRcpp(1, 0, std::numeric_limits<double>::infinity(), tau_samples(m - 1), S_tau);

    arma::mat var_proposed_tau = varArma(tau_proposed, rho_samples(m - 1), Dv, nbd_mat);
    arma::mat var_current_tau = varArma(tau_samples(m - 1), rho_samples(m - 1), Dv, nbd_mat);

    double acce_rate_tau = dmvnormArma(Xstar_trun, one_vec * mu_samples(m-1), var_proposed_tau) -
      dmvnormArma(Xstar_trun, one_vec * mu_samples(m-1), var_current_tau);

    int idx_tau = Rcpp::sample(NumericVector::create(1, 2), 1, true, NumericVector::create(std::min(1.0, exp(acce_rate_tau)), 1 - std::min(1.0, exp(acce_rate_tau))))[0];
    if (idx_tau == 1) {
      tau_samples(m) = tau_proposed;
    } else {
      tau_samples(m) = tau_samples(m - 1);
    }

    arma::uvec unique_tau = arma::find_unique(tau_samples.head(m + 1));
    rate_tau = static_cast<double>(unique_tau.n_elem) / (m + 1);
    if (autotune && rate_tau > 0.4) {
      S_tau *= 1.1;
    } else if (autotune && rate_tau < 0.2) {
      S_tau /= 1.1;
    }

    // rho
    double rho_proposed = rtruncnormRcpp(1, 0, 1, rho_samples(m - 1), S_rho);

    arma::mat var_proposed_rho = varArma(tau_samples(m), rho_proposed, Dv, nbd_mat);
    arma::mat var_current_rho = varArma(tau_samples(m), rho_samples(m - 1), Dv, nbd_mat);

    double accep_rate_rho = dmvnormArma(Xstar_trun, one_vec * mu_samples(m-1), var_proposed_rho) -
      dmvnormArma(Xstar_trun, one_vec * mu_samples(m-1), var_current_rho);

    int idx_rho = Rcpp::sample(NumericVector::create(1, 2), 1, true, NumericVector::create(std::min(1.0, exp(accep_rate_rho)), 1 - std::min(1.0, exp(accep_rate_rho))))[0];
    if (idx_rho == 1) {
      rho_samples(m) = rho_proposed;
    } else {
      rho_samples(m) = rho_samples(m - 1);
    }

    arma::uvec unique_rho = arma::find_unique(rho_samples.head(m + 1));
    rate_rho = static_cast<double>(unique_rho.n_elem) / (m + 1);
    if (autotune && rate_rho > 0.4) {
      S_rho *= 1.1;
    } else if (autotune && rate_rho < 0.2) {
      S_rho /= 1.1;
    }

    // delta
    double scale = pow(mu_samples(m-1), 2)/2+1;
    delta_samples(m) = rinvgammaArma(1, 1.5, scale)(0);

  }

  // Record the end time
  auto end_time = std::chrono::high_resolution_clock::now();

  // Calculate the elapsed time in seconds
  auto elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

  // return proposed;
  return List::create(
    Named("tau_samples") = tau_samples,
    Named("rho_samples") = rho_samples,
    Named("mu_samples") = mu_samples,
    Named("delta_samples") = delta_samples,
    Named("rate_tau") = rate_tau,
    Named("S_tau") = S_tau,
    Named("rate_rho") = rate_rho,
    Named("S_rho") = S_rho,
    Named("elapsed_time_sec") = elapsed_seconds.count() // Include elapsed time in seconds
  );
}

