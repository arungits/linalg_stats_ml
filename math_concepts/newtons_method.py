def find_root(number, root):
    # Note: If root is 2, this method will return the square root of "number",
    # If root is 3, this method will return the cube root, etc.

    assert(number > 0)
    assert(root >= 2)

    guess = number/root
    epsilon = pow(10, -10) # Used to check for convergence
    # Repeat for a maximum of 100 times and if there is no convergence, quit and raise error
    for iteration in range(100):
        new_guess = guess - (pow(guess, root) - number) / (root * pow(guess, root-1))
        if abs(new_guess - guess) < epsilon:
            print(f"Found answer in {iteration} iterations")
            return new_guess
        guess = new_guess

    raise "Unable to compute"

# Test Code
assert(round(find_root(2,2 ), 3) == 1.414)
assert(round(find_root(16,2 ), 3) == 4)
assert(round(find_root(101,2 ), 3) == 10.05)
assert(round(find_root(2,3 ), 3) == 1.26)
assert(round(find_root(8,3 ), 3) == 2)
assert(round(find_root(1,2 ), 3) == 1)
assert(round(find_root(1,3 ), 3) == 1)