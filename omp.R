library(Rcpp)
library(microbenchmark)

# from Jim Hester's excellent covr package
is.named <- function(x) {
  !is.null(names(x)) && all(names(x) != "")
}
set_makevars <- function(variables,
                         old_path = file.path("~", ".R", "Makevars"),
                         new_path = tempfile()) {
  if (length(variables) == 0) {
    return()
  }
  stopifnot(is.named(variables))

  old <- NULL
  if (file.exists(old_path)) {
    lines <- readLines(old_path)
    old <- lines
    for (var in names(variables)) {
      loc <- grep(paste(c("^[[:space:]]*", var, "[[:space:]]*", "="), collapse = ""), lines)
      if (length(loc) == 0) {
        lines <- append(lines, paste(sep = "=", var, variables[var]))
      } else if (length(loc) == 1) {
        lines[loc] <- paste(sep = "=", var, variables[var])
      } else {
        stop("Multiple results for ", var, " found, something is wrong.", .call = FALSE)
      }
    }
  } else {
    lines <- paste(names(variables), variables, sep = "=")
  }

  if (!identical(old, lines)) {
    writeLines(con = new_path, lines)
  }

  old
}

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

# using != as inner loop control - no difference, using pre-increment in n_all_true, no diff, static vs dynamic OpenMP, attempted to direct clang and gcc to unroll loops: didn't seem to work

sourceCpp("~/Documents/RProjects/optimization-comparison/omp.cpp", rebuild = rebuild, showOutput = TRUE, verbose = FALSE)

set.seed(21)
m <- matrix(sample(c(TRUE, FALSE), m_cols * m_rows, replace = TRUE), ncol = m_rows)

do_bench <- function(m, msg="", times = 25000) {
  message(msg)
  microbenchmark(
    #    t(m), # just transposing the data takes far longer than the best methods.
    #    hm(m),
    #    hm3(m),
    # hm_transpose(m),
    #hm_jmu(m),
    hm_npjc(m),
    hm_npjc_omp(m),
    hm_omp(m),
    hm_check_omp(m),
    #    hm_check_omp_no_sched(m),
    hm_check(m),
    hm_check_unroll(m),
    hm_check_unroll_10(m),
    hm_check_vectorize(m),
    # v slow hm_apply(m),
    times = times)

  message("in summary, -O3 makes the difference regardless of -march=native and any fancy loop unrolling things. Manually setting loop size to an integer rather than a variable was the biggest code difference. Perhaps pragmas for clang and gcc could improve on this...")
}


all_benches <- list()
if (!rebuild) stop("rebuild is off")
flags = c("",
          "-Os",
          "-O2",
          "-O3",
          "-O3 -march=native",
          "-O3 -funroll-loops",
          "-O3 -funroll-all-loops",
          "-O3 -fno-align-loops",
          "-O3 -march=native -funroll-all-loops -Wunsafe-loop-optimizations -funsafe-loop-optimizations")
# gcc only: flags = c("-O3 -fopt-info")

for (flag in flags) {
  message(flag)
  thisflag = list("CXXFLAGS" = flag)
  set_makevars(thisflag, "~/.R/Makevars", "~/.R/Makevars")
  sourceCpp("omp.cpp", rebuild = rebuild, showOutput = TRUE, verbose = FALSE)

  # do the hard coded loop size variants:
  cppFunction(macroExpand(m_rows), rebuild = rebuild)
  cppFunction(macroExpand_omp(m_rows),  plugins = "openmp", rebuild = rebuild)

  all_benches[[flag]] = do_bench(m)
}

for (bench in names(all_benches)) {
  message(bench)
  print(all_benches[[bench]])
}
