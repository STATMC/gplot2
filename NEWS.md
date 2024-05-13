# exampleRPackage 0.2.0

* Changed name from exampleDataPackage to exampleRPackage
* Added a `NEWS.md` file to track changes to the package.
##### Varience Redumption #####

### Antithetic Variates
#cdf sol  
MC.Phi <- function(x, R = 10000, antithetic = TRUE){  
  u <- runif(R/2)  
  if (!antithetic) v <- runif(R/2) else  
    v <- 1-u  
  u <- c(u, v)  
  cdf <- numeric(length(x))  
  for (i in 1:length(x)) {  
    g <- x[i]*exp(-(u*x[i])**2/2)  
    cdf[i] <- mean(g)/ sqrt(2*pi) + 0.5  
  }  
  cdf  
}  

x <- seq(0.1,2.5, length=5)  
Phi <- pnorm(x)  
MC1 <- MC.Phi(x, antithetic = FALSE)  
MC2 <- MC.Phi(x)  
print(round(rbind(x,MC1,MC2,Phi),5))  
m <- 1000  
mc1 <- mc2 <- numeric(m)  

xx <- 1.95  
for (i in 1:m) {  
  mc1[i] <- MC.Phi(xx, R = 1000, antithetic = FALSE)  
  mc2[i] <- MC.Phi(xx, R = 1000)  
}  
print(sd(mc1))  
print(sd(mc2))  
print((var(mc1)-var(mc2))/var(mc1))  
#the antithetic variance reduced to 99.5% at x = 1.95  

#for all the functions  

Ix <- function(x) {  
  exp(-2*x) / (1 + x^2) + (4*x + 2)^2  
}  


Bx <- function(x,a = 2, b = 3) (factorial(a+b-1)/(factorial(a-1)*factorial(b-1)))*(x**(a-1))*(x-1)**(b-1)  

monteCarloIntegration <- function(a, b, n, antithetic = TRUE) {  
  if (antithetic) {  
    u1 <- runif(n/2, min = a, max = b)  
    u2 <- a + b - u1  
    
    f_u1 <- Bx(u1)  
    f_u2 <- Bx(u2)  
    # Combine the antithetic pairs  
    estimates <- (f_u1 + f_u2) / 2  
  } else {  
    u <- runif(n, min = a, max = b)  
    estimates <- Bx(u)  
  }  
  #Calculate the Monte Carlo estimate of the integral  
  integral_estimate <- round((b - a) * mean(estimates),5)  
  #Calculate the variance of the estimates  
  variance_estimate <- round(((b - a)^2 * var(estimates)),5)  
  
  #Return the estimate and its variance  
  list(integral = integral_estimate, variance = variance_estimate)  
}  

compareMethods <- function(a, b, n) {  
  results_antithetic <- monteCarloIntegration(a, b, n, antithetic = TRUE)  
  results_standard <- monteCarloIntegration(a, b, n, antithetic = FALSE)  
  
  list(antithetic = results_antithetic,  
       standard = results_standard)  
}  

#Example usage:  
results_comparison <- compareMethods(0, 0.3, 10000)  
print(results_comparison)  


############  
###Control Variates####################  

monteCarloIntegrationcontrol <- function(Ix, fX, a, b, n = 100000) {  
  #Generate uniform random numbers between a and b  
  x <- runif(n, a, b)  
  
  #Evaluate the integrand and control function at these points  
  gX_values <- sapply(x, Ix)  
  fX_values <- sapply(x, fX)  
  
  #Classical Monte Carlo estimation  
  classical_estimate <- mean(gX_values) * (b - a)  
  classical_variance <- var(gX_values) * (b - a)^2  
  
  #Compute c* for control variates  
  c_star <- -cov(gX_values, fX_values) / var(fX_values)  
  expected_fX <- integrate(fX, lower = a, upper = b, subdivisions = 1000)$value / (b - a)  
  
  #Control Variate estimation  
  control_estimate <- mean(gX_values + c_star * (fX_values - expected_fX)) * (b - a)  
  control_variance <- var(gX_values + c_star * (fX_values - expected_fX)) * (b - a)^2  
  
  #Variance reduction percentage  
  variance_reduction <- 100 * (1 - control_variance / classical_variance)  
  
  #Results  
  list(  
    Classical_Estimate = classical_estimate,  
    Classical_Variance = classical_variance,  
    Control_Estimate = control_estimate, 
    Control_Variance = control_variance,  
    Variance_Reduction_Percent = variance_reduction  
  )  
}  

#Example usage  
Ix <- function(x) 1 / (1 + x)   # Your integrand function  
fX <- function(x) 1 + x         # Control function, correlated with the integrand  
a <- 0                          # Lower limit of integration  
b <- 1                          # Upper limit of integration  

#Compute the integrations  
results <- monteCarloIntegrationcontrol(Ix, fX, a, b)  
print(results)  
start1 <- proc.time()  
runtime1 <- proc.time()-start1  

###################  
compareAllMethods <- function(a, b, n) {  
  #Antithetic Variate Method  
  results_antithetic <- monteCarloIntegration(a, b, n, antithetic = TRUE)  
  
  #Classical Method (non-antithetic)  
  results_classical <- monteCarloIntegration(a, b, n, antithetic = FALSE)  
  
  #Control Variates Method  
  results_control <- monteCarloIntegrationcontrol(Ix, fX, a, b, n)  
  
  #Organize the results into a list for easy comparison  
  list(  
    Classical = results_classical,  
    Antithetic = results_antithetic,  
    Control_Variate = results_control  
  )  
}  

#Example usage  
#Compare methods between limits 0 and 1 with 10,000 samples  
comparison_results <- compareAllMethods(0, 1, 10000)  
print(comparison_results)  
