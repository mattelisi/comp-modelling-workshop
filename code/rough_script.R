# prepare data for modelling workshop

rm(list=ls())
hablar::set_wd_to_script_path()
library(tidyverse)

# d <- read_csv("../task/analysis/analysis_all_data.csv")
# str(d)
# 
# # general prep
# d <- d %>%
#   mutate(PID = str_sub(edf_file, 1, 4)) %>%
#   rename(session_id = participant_id) %>%
#   mutate(win=ifelse(win==-1, NA, win),
#          P=ifelse(win==-1, NA, P)) %>%
#   filter(!is.na(win))
# 
# 
# # subset AA04
# d_s1 <- d %>%
#   filter(PID == "AA13") %>%
#   select(trial_n, block_n, tar_choice, P, win, prob_1, prob_2, session, PID)
# 
# write_csv(d_s1, file="dAA13.csv")

# d_stan <- prep_stan_data(unique(d_s1$PID), data=d_s1)


d_s1 <- read_csv("../data/dAA13.csv")


# Example code

# Negative log-likelihood for a simple Q-learning model
# pars[1] = eta        (learning rate, constrained to 0-1)
# pars[2] = beta_temp  (inverse temperature, constrained to >0)

nll_qlearning <- function(pars, dat) {
  
  eta <- pars[1]
  beta_temp <- pars[2]
  
  # simple parameter bounds
  if (eta < 0 || eta > 1 || beta_temp <= 0) {
    return(1e10)
  }
  
  # make a block id like in your Stan prep
  dat$block_id <- paste(dat$session, dat$block_n, sep = "_")
  
  block_ids <- unique(dat$block_id)
  
  # initial Q values
  initQ <- c(0.5, 0.5)
  
  loglik <- 0
  
  # loop over blocks
  for (j in 1:length(block_ids)) {
    
    this_block <- block_ids[j]
    d_j <- dat[dat$block_id == this_block, ]
    
    # choice coded as:
    # 1 = chose option with higher reward probability
    # 0 = chose option with lower reward probability
    C <- ifelse(d_j$P > 0.5, 1, 0)
    R <- d_j$win
    
    Q <- initQ
    
    # loop over trials
    for (t in 1:nrow(d_j)) {
      
      VD <- Q[2] - Q[1]
      
      # p(choose high-reward option)
      p_choice1 <- 1 / (1 + exp(-beta_temp * VD))
      
      # add log-probability of observed choice
      if (C[t] == 1) {
        loglik <- loglik + log(p_choice1)
      } else {
        loglik <- loglik + log(1 - p_choice1)
      }
      
      # prediction error for chosen option
      PE <- R[t] - Q[C[t] + 1]
      
      # value update
      Q[C[t] + 1] <- Q[C[t] + 1] + eta * PE
    }
  }
  
  return(-loglik)
}



fit <- optim(
  par = c(0.2, 2),          # starting values: eta, beta_temp
  fn = nll_qlearning,
  dat = d_s1,
  method = "L-BFGS-B",
  lower = c(0, 0.0001),
  upper = c(1, 30),
  hessian=TRUE
)

fit

# SE approximated
sqrt(diag(solve(fit$hessian)))


#### data simulation function

simulate_qlearning <- function(dat, eta, beta_temp) {
  
  dat$block_id <- paste(dat$session, dat$block_n, sep = "_")
  block_ids <- unique(dat$block_id)
  
  initQ <- c(0.5, 0.5)
  
  # loop over blocks
  for (j in 1:length(block_ids)) {
    
    this_block <- block_ids[j]
    idx <- which(dat$block_id == this_block)
    d_j <- dat[idx, ]
    
    # assume reward probabilities are fixed within block
    p_high <- max(d_j$prob_1[1], d_j$prob_2[1])
    p_low  <- min(d_j$prob_1[1], d_j$prob_2[1])
    
    Q <- initQ
    
    # loop over trials
    for (t in 1:length(idx)) {
      
      i <- idx[t]
      
      # value difference: better option minus worse option
      VD <- Q[2] - Q[1]
      
      # probability of choosing the better option
      p_choose_high <- 1 / (1 + exp(-beta_temp * VD))
      
      # simulate choice: 1 = chose better option, 0 = chose worse option
      C_t <- rbinom(1, size = 1, prob = p_choose_high)
      
      # chosen reward probability
      if (C_t == 1) {
        dat$P[i] <- p_high
      } else {
        dat$P[i] <- p_low
      }
      
      # simulate reward
      dat$win[i] <- rbinom(1, size = 1, prob = dat$P[i])
      
      # prediction error
      PE <- dat$win[i] - Q[C_t + 1]
      
      # update chosen option
      Q[C_t + 1] <- Q[C_t + 1] + eta * PE
    }
  }
  
  dat$block_id <- NULL
  
  return(dat)
}



d_sim <- simulate_qlearning(
  dat = d_s1,
  eta = fit$par[1],
  beta_temp = fit$par[2]
)

### refit simulated data
fit_sim <- optim(
  par = c(0.2, 2),          # starting values: eta, beta_temp
  fn = nll_qlearning,
  dat = d_sim,
  method = "L-BFGS-B",
  lower = c(0, 0.0001),
  upper = c(1, 30),
  hessian=TRUE
)

fit_sim


############# paramtric bootstrapping

# number of bootstrap samples
B <- 200

# matrix to store parameter estimates
boot_par <- matrix(NA, nrow = B, ncol = 2)
colnames(boot_par) <- c("eta", "beta_temp")

# fitted parameters from original data
eta_hat <- fit$par[1]
beta_hat <- fit$par[2]

for (b in 1:B) {
  
  # simulate one new dataset from fitted model
  d_sim <- simulate_qlearning(
    dat = d_s1,
    eta = eta_hat,
    beta_temp = beta_hat
  )
  
  # refit model to simulated data
  fit_b <- optim(
    par = c(eta_hat, beta_hat),
    fn = nll_qlearning,
    dat = d_sim,
    method = "L-BFGS-B",
    lower = c(0, 0.0001),
    upper = c(1, 20)
  )
  
  # store estimates
  boot_par[b, ] <- fit_b$par
}

# standard errors
apply(boot_par, 2, sd)


########################################### alternative model
# win-stay-loose-shift, with a possibility of random response with probaiblity lambda


nll_wsls <- function(pars, dat) {
  
  lambda <- pars[1]
  
  if (lambda < 0 || lambda > 1) {
    return(1e10)
  }
  
  dat$block_id <- paste(dat$session, dat$block_n, sep = "_")
  block_ids <- unique(dat$block_id)
  
  loglik <- 0
  
  for (j in 1:length(block_ids)) {
    
    this_block <- block_ids[j]
    d_j <- dat[dat$block_id == this_block, ]
    
    C <- ifelse(d_j$P > 0.5, 1, 0)
    R <- d_j$win
    
    # first trial in each block: no previous trial, so set p = 0.5
    loglik <- loglik + log(0.5)
    
    for (t in 2:nrow(d_j)) {
      
      stay <- (C[t] == C[t - 1])
      
      if (R[t - 1] == 1) {
        # previous trial was a win -> win-stay
        if (stay) {
          p_t <- 0.5 * lambda + (1 - lambda)
        } else {
          p_t <- 0.5 * lambda
        }
      } else {
        # previous trial was a loss -> lose-shift
        if (stay) {
          p_t <- 0.5 * lambda
        } else {
          p_t <- 0.5 * lambda + (1 - lambda)
        }
      }
      
      loglik <- loglik + log(p_t)
    }
  }
  
  return(-loglik)
}


fit_wsls <- optim(
  par = 0.2,
  fn = nll_wsls,
  dat = d_s1,
  method = "L-BFGS-B",
  lower = 1e-15,
  upper = 1,
  hessian = TRUE
)

fit_wsls


simulate_wsls <- function(dat, lambda) {
  
  dat$block_id <- paste(dat$session, dat$block_n, sep = "_")
  block_ids <- unique(dat$block_id)
  
  for (j in 1:length(block_ids)) {
    
    this_block <- block_ids[j]
    idx <- which(dat$block_id == this_block)
    d_j <- dat[idx, ]
    
    p_high <- max(d_j$prob_1[1], d_j$prob_2[1])
    p_low  <- min(d_j$prob_1[1], d_j$prob_2[1])
    
    # first trial: random choice
    C_prev <- rbinom(1, size = 1, prob = 0.5)
    
    if (C_prev == 1) {
      dat$P[idx[1]] <- p_high
    } else {
      dat$P[idx[1]] <- p_low
    }
    
    dat$win[idx[1]] <- rbinom(1, size = 1, prob = dat$P[idx[1]])
    R_prev <- dat$win[idx[1]]
    
    # remaining trials
    for (t in 2:length(idx)) {
      
      if (R_prev == 1) {
        # after win: stay with probability 1 - lambda/2
        p_stay <- 0.5 * lambda + (1 - lambda)
      } else {
        # after loss: stay with probability lambda/2
        p_stay <- 0.5 * lambda
      }
      
      stay <- rbinom(1, size = 1, prob = p_stay)
      
      if (stay == 1) {
        C_t <- C_prev
      } else {
        C_t <- 1 - C_prev
      }
      
      if (C_t == 1) {
        dat$P[idx[t]] <- p_high
      } else {
        dat$P[idx[t]] <- p_low
      }
      
      dat$win[idx[t]] <- rbinom(1, size = 1, prob = dat$P[idx[t]])
      
      C_prev <- C_t
      R_prev <- dat$win[idx[t]]
    }
  }
  
  dat$block_id <- NULL
  
  return(dat)
}


######################### parameter recovery


library(ggplot2)

# grid of true parameter values
lambda_grid <- seq(0.05, 0.95, length.out = 10)

# number of repetitions for each true value
n_rep <- 10

# empty object to store results
recov <- data.frame(
  lambda_true = numeric(),
  lambda_fit = numeric()
)

for (i in 1:length(lambda_grid)) {
  
  lambda_true <- lambda_grid[i]
  
  for (r in 1:n_rep) {
    
    # simulate dataset
    d_sim <- simulate_wsls(
      dat = d_s1,
      lambda = lambda_true
    )
    
    # refit model
    fit_wsls_sim <- optim(
      par = 0.2,
      fn = nll_wsls,
      dat = d_sim,
      method = "L-BFGS-B",
      lower = 1e-15,
      upper = 1
    )
    
    # store results
    recov <- rbind(
      recov,
      data.frame(
        lambda_true = lambda_true,
        lambda_fit = fit_wsls_sim$par
      )
    )
  }
}


# summarise across repetitions
recov_sum <- aggregate(
  lambda_fit ~ lambda_true,
  data = recov,
  FUN = function(x) c(mean = mean(x), sd = sd(x))
)

recov_sum <- data.frame(
  lambda_true = recov_sum$lambda_true,
  lambda_mean = recov_sum$lambda_fit[, "mean"],
  lambda_sd = recov_sum$lambda_fit[, "sd"]
)

recov_sum


# parameter recovery plot

ggplot(recov_sum, aes(x = lambda_true, y = lambda_mean)) +
  geom_point() +
  geom_errorbar(
    aes(
      ymin = lambda_mean - lambda_sd,
      ymax = lambda_mean + lambda_sd
    ),
    width = 0.02
  ) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  labs(
    x = "True lambda",
    y = "Fitted lambda",
    title = "Parameter recovery for the WSLS model"
  ) +
  theme_classic()



########################################### Plot models fit

# binomial standard error
binomSEM <- function(v) {
  sqrt((mean(v) * (1 - mean(v))) / length(v))
}

# functions to compute trial-by-trial predicted probabilities

predict_qlearning <- function(pars, dat) {
  
  eta <- pars[1]
  beta_temp <- pars[2]
  
  dat$block_id <- paste(dat$session, dat$block_n, sep = "_")
  block_ids <- unique(dat$block_id)
  
  pred <- numeric(nrow(dat))
  
  initQ <- c(0.5, 0.5)
  
  for (j in 1:length(block_ids)) {
    
    this_block <- block_ids[j]
    idx <- which(dat$block_id == this_block)
    d_j <- dat[idx, ]
    
    C <- ifelse(d_j$P > 0.5, 1, 0)
    R <- d_j$win
    
    Q <- initQ
    
    for (t in 1:nrow(d_j)) {
      
      VD <- Q[2] - Q[1]
      p_high <- 1 / (1 + exp(-beta_temp * VD))
      
      pred[idx[t]] <- p_high
      
      PE <- R[t] - Q[C[t] + 1]
      Q[C[t] + 1] <- Q[C[t] + 1] + eta * PE
    }
  }
  
  dat$block_id <- NULL
  
  return(pred)
}

# and 

predict_wsls <- function(pars, dat) {
  
  lambda <- pars[1]
  
  dat$block_id <- paste(dat$session, dat$block_n, sep = "_")
  block_ids <- unique(dat$block_id)
  
  pred <- numeric(nrow(dat))
  
  for (j in 1:length(block_ids)) {
    
    this_block <- block_ids[j]
    idx <- which(dat$block_id == this_block)
    d_j <- dat[idx, ]
    
    C <- ifelse(d_j$P > 0.5, 1, 0)
    R <- d_j$win
    
    # first trial: random choice
    pred[idx[1]] <- 0.5
    
    for (t in 2:nrow(d_j)) {
      
      if (R[t - 1] == 1) {
        # after win: stay
        p_stay <- 0.5 * lambda + (1 - lambda)
      } else {
        # after loss: shift
        p_stay <- 0.5 * lambda
      }
      
      if (C[t - 1] == 1) {
        pred[idx[t]] <- p_stay
      } else {
        pred[idx[t]] <- 1 - p_stay
      }
    }
  }
  
  dat$block_id <- NULL
  
  return(pred)
}

# compute and add

d_plot <- d_s1

# observed choice of high-probability option
d_plot$C <- ifelse(d_plot$P > 0.5, 1, 0)

# model predictions
d_plot$pred_qlearning <- predict_qlearning(fit$par, d_plot)
d_plot$pred_wsls <- predict_wsls(fit_wsls$par, d_plot)


# summarise

sum_qlearning <- data.frame(
  trial_n = sort(unique(d_plot$trial_n)),
  obs_mean = NA,
  obs_se = NA,
  pred_mean = NA,
  pred_se = NA
)

for (tt in 1:length(sum_qlearning$trial_n)) {
  
  tr <- sum_qlearning$trial_n[tt]
  d_tt <- d_plot[d_plot$trial_n == tr, ]
  
  sum_qlearning$obs_mean[tt] <- mean(d_tt$C)
  sum_qlearning$obs_se[tt] <- binomSEM(d_tt$C)
  sum_qlearning$pred_mean[tt] <- mean(d_tt$pred_qlearning)
  sum_qlearning$pred_se[tt] <- sd(d_tt$pred_qlearning) / sqrt(nrow(d_tt))
}


sum_wsls <- data.frame(
  trial_n = sort(unique(d_plot$trial_n)),
  obs_mean = NA,
  obs_se = NA,
  pred_mean = NA,
  pred_se = NA
)

for (tt in 1:length(sum_wsls$trial_n)) {
  
  tr <- sum_wsls$trial_n[tt]
  d_tt <- d_plot[d_plot$trial_n == tr, ]
  
  sum_wsls$obs_mean[tt] <- mean(d_tt$C)
  sum_wsls$obs_se[tt] <- binomSEM(d_tt$C)
  sum_wsls$pred_mean[tt] <- mean(d_tt$pred_wsls)
  sum_wsls$pred_se[tt] <- sd(d_tt$pred_wsls) / sqrt(nrow(d_tt))
}


# make plots: Q-learning

ggplot(sum_qlearning, aes(x = trial_n)) +
  geom_ribbon(
    aes(
      ymin = pred_mean - pred_se,
      ymax = pred_mean + pred_se
    ),
    alpha = 0.2
  ) +
  geom_line(aes(y = pred_mean), linewidth = 1, color="dark grey") +
  geom_point(aes(y = obs_mean)) +
  geom_errorbar(
    aes(
      ymin = obs_mean - obs_se,
      ymax = obs_mean + obs_se
    ),
    width = 0.2
  ) +
  labs(
    x = "Trial number",
    y = "P(choice of high-probability option)",
    title = "Observed choices and\nQ-learning predictions"
  ) +
  theme_classic()

# WSLS

ggplot(sum_wsls, aes(x = trial_n)) +
  geom_ribbon(
    aes(
      ymin = pred_mean - pred_se,
      ymax = pred_mean + pred_se
    ),
    alpha = 0.2
  ) +
  geom_line(aes(y = pred_mean), linewidth = 1, color="dark grey") +
  geom_point(aes(y = obs_mean)) +
  geom_errorbar(
    aes(
      ymin = obs_mean - obs_se,
      ymax = obs_mean + obs_se
    ),
    width = 0.2
  ) +
  labs(
    x = "Trial number",
    y = "P(choice of high-probability option)",
    title = "Observed choices and\nWSLS predictions"
  ) +
  theme_classic()



################################## model comparison

# 1: information criteria

# number of observations
n_obs <- nrow(d_s1)

# number of free parameters
k_qlearning <- 2
k_wsls <- 1

# maximized log-likelihoods
loglik_qlearning <- -fit$value
loglik_wsls <- -fit_wsls$value

# AIC
AIC_qlearning <- 2 * k_qlearning - 2 * loglik_qlearning
AIC_wsls <- 2 * k_wsls - 2 * loglik_wsls

# BIC
BIC_qlearning <- log(n_obs) * k_qlearning - 2 * loglik_qlearning
BIC_wsls <- log(n_obs) * k_wsls - 2 * loglik_wsls

# display results
comparison_data <- data.frame(
  model = c("Q-learning", "WSLS"),
  k = c(k_qlearning, k_wsls),
  logLik = c(loglik_qlearning, loglik_wsls),
  AIC = c(AIC_qlearning, AIC_wsls),
  BIC = c(BIC_qlearning, BIC_wsls)
)


################################## model via cross-validation


cv_block <- function(dat, nll_fun, par_start, lower, upper) {
  
  dat$block_id <- paste(dat$session, dat$block_n, sep = "_")
  block_ids <- unique(dat$block_id)
  
  cv_loglik <- 0
  
  for (j in 1:length(block_ids)) {
    
    holdout_block <- block_ids[j]
    
    d_train <- dat[dat$block_id != holdout_block, ]
    d_test  <- dat[dat$block_id == holdout_block, ]
    
    fit_j <- optim(
      par = par_start,
      fn = nll_fun,
      dat = d_train,
      method = "L-BFGS-B",
      lower = lower,
      upper = upper
    )
    
    # evaluate log-likelihood on hold-out block
    loglik_test <- -nll_fun(fit_j$par, d_test)
    
    cv_loglik <- cv_loglik + loglik_test
  }
  
  dat$block_id <- NULL
  
  return(cv_loglik)
}



# leave-one-block-out predictive log-likelihood
cval_qlearning <- cv_block(
  dat = d_s1,
  nll_fun = nll_qlearning,
  par_start = c(0.2, 2),
  lower = c(0, 0.0001),
  upper = c(1, 20)
)

cval_wsls <- cv_block(
  dat = d_s1,
  nll_fun = nll_wsls,
  par_start = 0.2,
  lower = 1e-15,
  upper = 1
)

comparison_data$CVAL <- c(cval_qlearning, cval_wsls)




########################## BOnus: Q-learning model in a Bayesian framework

prep_stan_data <- function(dat){
  
  dat$block_id <- paste(dat$session,dat$block_n,sep="_")
  
  J <- length(unique(dat$block_id))
  N <- tapply(dat$block_id,dat$block_id, length)
  
  if(any(N)<10){
    xbid <- names(N)[which(N<10)] 
    dat <- dat %>%
      filter(!is.element(block_id, xbid))
    J <- length(unique(dat$block_id))
    N <- tapply(dat$block_id,dat$block_id, length)
  }
  
  maxN <- max(N)
  
  dat$C <- ifelse(dat$P>0.5,1,0)
  C <- array(dim=c(J,maxN))
  R <- array(dim=c(J,maxN))
  for(j in 1:J){
    c_j <- unique(dat$block_id)[j]
    C[j,1:N[j]] <- dat$C[dat$block_id==c_j]
    R[j,1:N[j]] <- dat$win[dat$block_id==c_j]
  }
  
  # check NA
  C[is.na(C)] <- -99
  R[is.na(R)] <- -99
  
  d_stan <- list(J=J, N=N, maxN=maxN, C=C, R=R)

  str(d_stan)
  return(d_stan)
}


d_stan <- prep_stan_data(d_s1)

library(rstan)

options(mc.cores = parallel::detectCores()) # indicate stan to use multiple cores if available
# run sampling: 4 chains in parallel on separate cores

stan_fit <- stan(file = "Qlearning_M1.stan", data = d_stan, iter = 2000, chains = 4)

library(tidybayes)


# extract posterior draws
post_draws <- stan_fit %>%
  spread_draws(eta, beta_temp) %>%
  data.frame()

# put in long format
post_long <- rbind(
  data.frame(parameter = "eta", value = post_draws$eta),
  data.frame(parameter = "beta_temp", value = post_draws$beta_temp)
)

# frequentist estimates
mle_df <- data.frame(
  parameter = c("eta", "beta_temp"),
  mle = fit$par
)

# plot
ggplot(post_long, aes(x = value)) +
  geom_density(aes(fill=parameter), alpha=0.5) +
  geom_vline(
    data = mle_df,
    linewidth=1,
    aes(xintercept = mle),
    linetype = "dashed"
  ) +
  guides(fill = "none") +
  facet_wrap(~ parameter, scales = "free") +
  labs(
    x = "Parameter value",
    y = "Posterior density",
    title = "Bayesian posterior and frequentist estimate",
    caption = "Dashed line = maximum likelihood estimate"
  )



