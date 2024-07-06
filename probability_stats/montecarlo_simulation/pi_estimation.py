import numpy as np
import random

def generate_point():
    return (random.random(), random.random())

def get_stats(estimates):
    avg = np.average(estimates)
    sd = np.std(estimates)
    # Calculate 95% confidence intervals
    ci = (avg - 1.96*sd, avg + 1.96*sd)
    return avg, sd, ci
def find_pi(trials = 1000):
    # This method estimates the value of pi by performing the following experiment:
    # It generates a random point (x,y) where x and y are real numbers between 0 & 1.
    # All such generated points will be within the unit sqaure with corners at (0,0), (1,0), (0,1) & (1,1)
    # However only some of the generated points will be within the unit circle with center at origin.
    # We know the ratio of the points inside the unit circle to the points inside unit square will be 4/pi
    # From this we can estimate the value of pi.
    # As this a random experiement, we will be able to estimate PI more confidently by repeating
    # the experiment for many trials.
    # This method repeats the experiment for the specified number of trials and
    # returns a list containing the pi_estimates from all the trials

    assert(trials <= 1000000 and trials >= 1000)
    # Initialize number of random points
    n = 500
    pi_estimates = []
    for trial in range(trials):
        n_inside_unit_circle = 0
        for i in range(n):
            x,y = generate_point()
            if np.sqrt(x**2 + y**2) <= 1.0:
                n_inside_unit_circle += 1
        pi_estimates.append(4 * n_inside_unit_circle/n)
    return pi_estimates

# Test Code

pi_actual = 3.1415926535

# For 1000 trials
pi_estimates = find_pi()
avg, sd, ci = get_stats(pi_estimates)
print("Expected PI value after 1000 trials: ", avg)
print("95% Confidence Interval for PI after 1000 trials: ", ci)
error1 = abs(avg - pi_actual)
# Test that the actual PI value falls within the confidence interval
assert(pi_actual >= ci[0] and pi_actual <= ci[1])

# For 100000 trials

pi_estimates = find_pi(10000)
avg, sd, ci = get_stats(pi_estimates)
print("Expected PI value after 10000 trials: ", avg)
print("95% Confidence Interval for PI after 10000 trials: ", ci)
error2 = abs(avg - pi_actual)
# Test that the actual PI value falls within the confidence interval
assert(pi_actual >= ci[0] and pi_actual <= ci[1])

# For 1000000 trials
pi_estimates = find_pi(100000)
avg, sd, ci = get_stats(pi_estimates)
print("Expected PI value after 100000 trials: ", avg)
print("95% Confidence Interval for PI after 100000 trials: ", ci)
error3 = abs(avg - pi_actual)
# Test that the actual PI value falls within the confidence interval
assert(pi_actual >= ci[0] and pi_actual <= ci[1])

# Sample output:

# Expected PI value after 1000 trials:  3.14284
# 95% Confidence Interval for PI after 1000 trials:  (np.float64(2.9991903390695267), np.float64(3.2864896609304735))
# Expected PI value after 10000 trials:  3.1409248
# 95% Confidence Interval for PI after 10000 trials:  (np.float64(2.997011579509045), np.float64(3.2848380204909553))
# Expected PI value after 100000 trials:  3.14138304
# 95% Confidence Interval for PI after 100000 trials:  (np.float64(2.99727534975568), np.float64(3.28549073024432))