data {
  int<lower=1> J;                    // n block
  array[J] int<lower=1> N;           // n trials x block
  int<lower=1> maxN;                 // max number of trials
  array[J, maxN] int<lower=-99, upper=1> C;   // choice (of option with high reward prob)
  array[J, maxN] int<lower=-99, upper=1> R;    // feedback (positive or negative)
}

transformed data {
  array[2] real initQ; // initial Q values for the 2 options
  int Nobs;            // number of total observations
  
  initQ = rep_array(0.5, 2);
  Nobs = sum(N);
}

parameters {
  real<lower=0,upper=1> eta; // learning rate
  real<lower=0> beta_temp;   // inverse temperature
}

model {
  // PRIORS
  // from: https://doi.org/10.1016/j.jmp.2016.01.006
  
  eta ~ beta(0.007, 0.018);
  beta_temp ~ gamma(4.83, 0.73);
  
  // LIKELIHOOD
  // block and trial loop
  for (i in 1:J) {

    array[2] real Q;  // Q values (expected value)
    real VD;          // value difference
    real PE;          // prediction error

    Q = initQ;

    for (t in 1:N[i]) {

      VD = Q[2] - Q[1];

      // prediction error
      PE = R[i,t] - Q[C[i,t]+1];

      // choice probability (softmax)
      C[i,t] ~ bernoulli(1/(1+exp(-beta_temp * VD )));

      // value updating (learning)
      Q[C[i,t]+1] = Q[C[i,t]+1] + eta * PE;

    }
  }
}


