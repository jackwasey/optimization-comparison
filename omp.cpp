#include <Rcpp.h>

// [[Rcpp::plugins(openmp)]]

// [[Rcpp::export]]
int hm(const Rcpp::LogicalMatrix& x)
{
  const int xrows = x.nrow();
  const int xcols = x.ncol();
  int n_all_true = 0;

  for(size_t row = 0; row < xrows; row++) {
    int r_ttl = 0;
    for(size_t col = 0; col < xcols; col++) {
      r_ttl += x(row,col);
    }
    if(r_ttl == xcols){
      n_all_true++;
    }
  }
  return n_all_true;
}

// [[Rcpp::export]]
int hm_jmu(const Rcpp::LogicalMatrix& x)
{
  const int xrows = x.nrow();
  const int xcols = x.ncol();
  int n_all_true = 0;

  for(int row = 0; row < xrows; row++) {
    int r_ttl = 0;
    for(int col = 0; col < xcols; col++) {
      r_ttl += x(row,col);
    }
    if(r_ttl == xcols){
      n_all_true++;
    }
  }
  return n_all_true;
}

// [[Rcpp::export]]
int hm_omp(const Rcpp::LogicalMatrix& x) {
  const int xrows = x.nrow();
  const int xcols = x.ncol();
  int n_all_true = 0;

#pragma omp parallel for reduction(+:n_all_true) schedule(static)
  for(int row = 0; row < xrows; row++) {
    int r_ttl = 0;
    for(int col = 0; col < xcols; col++) {
      r_ttl += x(row,col);
    }
    if(r_ttl == xcols){
      n_all_true++;
    }
  }
  return n_all_true;
}


// [[Rcpp::export]]
int hm_check_omp(const Rcpp::LogicalMatrix& x)
{
  const int xrows = x.nrow();
  const int xcols = x.ncol();
  int n_all_true = 0;
#pragma omp parallel for reduction(+:n_all_true) schedule(static)
  for(int row = 0; row < xrows; ++row) {
#pragma unroll 10
    for(int col = 0; col < xcols; ++col) {
      if (!x(row, col))
        goto finished_row;
    }
    n_all_true += 1;
    finished_row:
    ;
  }
  return n_all_true;
}

// [[Rcpp::export]]
int hm_check_omp_no_sched(const Rcpp::LogicalMatrix& x)
{
  const int xrows = x.nrow();
  const int xcols = x.ncol();
  int n_all_true = 0;
#pragma omp parallel for reduction(+:n_all_true)
  for(int row = 0; row < xrows; ++row) {
#pragma unroll 10
    for(int col = 0; col < xcols; ++col) {
      if (!x(row, col))
        goto finished_row;
    }
    n_all_true += 1;
    finished_row:
      ;
  }
  return n_all_true;
}

// [[Rcpp::export]]
int hm_check(const Rcpp::LogicalMatrix& x)
{
  const int xrows = x.nrow();
  const int xcols = x.ncol();
  int n_all_true = 0;
  bool r_ttl;
  bool r_test;
  for(int row = 0; row < xrows; ++row) {
    for(int col = 0; col < xcols; ++col) {
      r_test = x(row, col);
      if (!r_test)
        goto finished_row;
    }
    n_all_true += 1;
    finished_row:
      ;
  }
  return n_all_true;
}

// [[Rcpp::export]]
int hm_check_unroll(const Rcpp::LogicalMatrix& x)
{
  const int xrows = x.nrow();
  const int xcols = x.ncol();
  int n_all_true = 0;
  bool r_ttl;
  bool r_test;
#pragma unroll
  for(int row = 0; row < xrows; ++row) {
    for(int col = 0; col < xcols; ++col) {
      r_test = x(row, col);
      if (!r_test)
        goto finished_row;
    }
    n_all_true += 1;
    finished_row:
      ;
  }
  return n_all_true;
}

// [[Rcpp::export]]
int hm_check_unroll_10(const Rcpp::LogicalMatrix& x)
{
  const int xrows = x.nrow();
  const int xcols = x.ncol();
  int n_all_true = 0;
  bool r_ttl;
  bool r_test;
#pragma unroll 10
  for(int row = 0; row < xrows; ++row) {
    for(int col = 0; col < xcols; ++col) {
      r_test = x(row, col);
      if (!r_test)
        goto finished_row;
    }
    n_all_true += 1;
    finished_row:
      ;
  }
  return n_all_true;
}

// [[Rcpp::export]]
int hm_check_vectorize(const Rcpp::LogicalMatrix& x)
{
  const int xrows = x.nrow();
  const int xcols = x.ncol();
  int n_all_true = 0;
  bool r_ttl;
  bool r_test;
#pragma unroll(10)
  for(int row = 0; row < xrows; ++row) {
    for(int col = 0; col < xcols; ++col) {
      r_test = x(row, col);
      if (!r_test)
        goto finished_row;
    }
    n_all_true += 1;
    finished_row:
      ;
  }
  return n_all_true;
}

// [[Rcpp::export]]
int hm_check_chunks(const Rcpp::LogicalMatrix& x)
{
  const int xrows = x.nrow();
  const int xcols = x.ncol();
  int n_all_true = 0;
  bool r_ttl;
  bool r_test;
  int unroll_by = 10;
  for (int unroll = 0; unroll < floor(xrows/unroll_by); ++unroll)
  for(int row = 0; row < xrows; ++row) {
    for(int col = 0; col < xcols; ++col) {
      r_test = x(row, col);
      if (!r_test)
        goto finished_row;
    }
    n_all_true += 1;
    finished_row:
      ;
  }
  return n_all_true;
}

// setting up std::vector<bool> vb(xrows * xcols); alone takes ages
