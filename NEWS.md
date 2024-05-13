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
# the antithetic variance reduced to 99.5% at x = 1.95  

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
  # Calculate the Monte Carlo estimate of the integral  
  integral_estimate <- (b - a) * mean(estimates)  
  # Calculate the variance of the estimates  
  variance_estimate <- (b - a)^2 * var(estimates) / n  
  
  # Return the estimate and its variance  
  list(integral = integral_estimate, variance = variance_estimate)  
}

compareMethods <- function(a, b, n) {  
  results_antithetic <- monteCarloIntegration(a, b, n, antithetic = TRUE)  
  results_standard <- monteCarloIntegration(a, b, n, antithetic = FALSE)  
  
  list(antithetic = results_antithetic,  
       standard = results_standard)  
}

# Example usage:  
results_comparison <- compareMethods(0, 0.3, 10000)  
print(results_comparison)  



### Control Variates  

# c = -(cov(g(x),f(x)))/var(f(x))  

# int 1 to 0 | exp(-x)/(1+x**2) dx  is g(x)  
# f(x) = e**(-0.5)/(1+x**2)  

fx <- function(u){  
  exp(-0.5)/(1+u**2)  
}  

gx <- function(u){  
  exp(-u)/(1+u**2)  
}  

u <- runif(1000)  
B <- fx(u)  
A <- gx(u)  

#we want bigger than 0.90  
cor(A,B)  

a <- -cov(A,B)/var(B)  

T1 <- gx(u)  

T2 <- T1 + a* (fx(u)-exp(-0.5)*pi/4)  
c(mean(T1), (mean(T2)))  

c(var(T1), var(T2))  

(var(T1) - var(T2)) / var(T1)  