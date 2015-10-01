library(Rcpp)
library(microbenchmark)

m_rows <- 10L
m_cols <- 50000L
rebuild = TRUE

hm3 <- function(m) {
  nc <- ncol(m)
  sum(rowSums(m) == nc)
}

# faster working in cols, because R data is column items close together
hm_transpose <- function(m) {
  nc <- ncol(m)
  n <- t(m)
  sum(colSums(n) == nc)
}

# very slow
hm_apply <- function(m) {
  sum(apply(m, 1, any))
}


macroExpand <- function(NCOL) {
  paste0('int hm_npjc(const LogicalMatrix& x)
{
  const int xrows = x.nrow();
  int n_all_true = 0;

  for(int row = 0; row < xrows; row++) {
  int r_ttl = 0;
  for(int col = 0; col < ',NCOL,'; col++) {
  r_ttl += x(row,col);
  }
  if(r_ttl == ',NCOL,'){
  n_all_true++;
  }
  }
  return n_all_true;
  }')
}

macroExpand_omp <- function(NCOL) {
  paste0('int hm_npjc_omp(const LogicalMatrix& x)
{
  const int xrows = x.nrow();
  int n_all_true = 0;

  #pragma omp parallel for reduction(+:n_all_true)
  for(int row = 0; row < xrows; row++) {
  int r_ttl = 0;
  for(int col = 0; col < ',NCOL,'; col++) {
  r_ttl += x(row,col);
  }
  if(r_ttl == ',NCOL,'){
  n_all_true++;
  }
  }
  return n_all_true;
  }')
}

cppFunction(macroExpand(m_rows), rebuild = rebuild)
cppFunction(macroExpand_omp(m_rows),  plugins = "openmp", rebuild = rebuild)

# using != as inner loop control - no difference, using pre-increment in n_all_true, no diff, static vs dynamic OpenMP, attempted to direct clang and gcc to unroll loops: didn't seem to work

sourceCpp("~/Documents/RProjects/optimization-comparison/omp.cpp", rebuild = rebuild, showOutput = TRUE, verbose = FALSE)

set.seed(21)
m <- matrix(sample(c(TRUE, FALSE), m_cols * m_rows, replace = T), ncol = m_rows)

do_bench <- function(m, msg="", times = 1000) {
  message(msg)
  microbenchmark(
#    t(m), # just transposing the data takes far longer than the best methods.
#    hm(m),
#    hm3(m),
    hm_transpose(m),
    hm_jmu(m),
    hm_npjc(m),
    hm_omp(m),
    hm_check_omp(m),
#    hm_check_omp_no_sched(m),
    hm_npjc_omp(m),
    hm_check(m),
    hm_check_unroll(m),
#    hm_check_unroll_10(m),
    hm_check_vectorize(m),
    # v slow hm_apply(m),
    times = times)
}


all_benches <- list()

if (!rebuild) stop("rebuild is off")
vanilla <- Sys.getenv("PKG_CFLAGS")
on.exit(Sys.setenv("PKG_CXXFLAGS" = vanilla))

flags = c(vanilla,
          "-O3",
          "-O3 -funroll-loops",
          "-O3 -funroll-all-loops",
          "-O3 -fno-align-loops",
          "-O4",
          "-Wunsafe-loop-optimizations -funsafe-loop-optimizations")
# gcc_flags = c("-O3 -fopt-info")

for (flag in flags) {
  Sys.setenv("PKG_CXXFLAGS" = flag)
  sourceCpp("/home/jack/so32810274/omp.cpp", rebuild = rebuild, showOutput = TRUE, verbose = FALSE)
  all_benches[[flag]] = do_bench(m)
}

for (bench in names(all_benches)) {
  message(bench)
  print(all_benches[[bench]])
}
