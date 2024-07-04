def find_e():
    # This method returns euler's constant upto 5 decimal places
    # Euler's constant is given by the formula:
    # e = (1+1/n)^n as n -> infinity

    n = pow(10, 6)
    e = pow(1+1/n, n)
    return round(e,5)

assert(find_e() == 2.71828)

