import time

class Random:
    '''
    This class implements a pseudo random number gemeratpr using "Middle Square Method"
    '''
    def __init__(self, seed=None, digits=8):
        if digits <= 0 or digits > 10:
            raise ValueError
        if seed is None:
            self.seed = self.generate_seed(digits)
        else:
            self.seed = seed
        self.digits = digits
        self.max_possible_n = pow(10, digits) - 1

    def random(self):
        return self.randint() / self.max_possible_n

    def randint(self, start=0, end=None):
        if end is None:
            end = self.max_possible_n
        range = end - start
        if range > self.max_possible_n:
            raise ValueError
        n = str(self.seed * self.seed)
        if len(n) < (self.digits * 2):
            n = "0" + n
        start_index = self.digits // 2
        end_index = start_index + self.digits
        if n != "00":
            n = int(n[start_index:end_index])
        else:
            n = 0
        self.seed = n
        n = n % (range + 1)
        return n + start

    def randomchoice(self, list):
        size = len(list)
        assert(size <= self.max_possible_n)
        random_index = self.randint(0, size - 1)
        return list[random_index]

    @classmethod
    def generate_seed(cls, digits):
        good_8_digit_seeds = [20424812, 20424863, 20424871]
        if digits == 8:
            random_index = int(time.time()) % 3
            seed = good_8_digit_seeds[random_index]
        else:
            seed = int(time.time()) % pow(10, digits)
        return seed

# Test Code
# Some good 8 digit seeds that result in more even distribution - 20424812, 20424863, 20424871
# Examples of bad 8 digit seeds - 20424844, 20424878

def test_random(random):
    rand_counts = {}
    for i in range(1000000):
        n = random.random()
        assert (0 <= n and n <= 1)
        key = int(n / 0.1)
        if key not in rand_counts:
            rand_counts[key] = 0
        rand_counts[key] += 1
    print(rand_counts)
    for i in range(10):
        assert(i in rand_counts)

    for key in rand_counts:
        count = rand_counts[key]
        assert (90000 <= count and count <= 110000)

def test_randint(random, low, high):
    randint_counts = {}
    for i in range(1000000):
        n = random.randint(low, high)
        assert (low <= n and n <= high)
        if n not in randint_counts:
            randint_counts[n] = 0
        randint_counts[n] += 1
    print(randint_counts)
    for i in range(low, high+1):
        assert(i in randint_counts)
    for key in randint_counts:
        count = randint_counts[key]
        assert (85000 <= count and count <= 115000)

random = Random()
# Print the seed selected
print(random.seed)
test_random(random)
test_randint(random, 0, 9)
test_randint(random, 101, 110)


