import random

import mmh3
import numpy as np
import string
import random

'''
Bloom filter is a probabilistic data structure. Itis a space efficient fast way to 
probabilistically check if an element is present in a large collection or not.
It is extensively used in distributed data technologies such as Cassandra, DynamoDB, etc.
'''
class BloomFilter:
    def __init__(self, count, false_positive_probability=0.1):
        self.count = count
        self.false_positive_probability = false_positive_probability
        self.bit_array_size = self.get_array_size(count, false_positive_probability)
        self.hash_count = self.get_hash_count(self.bit_array_size, count)
        self.bit_array = [False for i in range(self.bit_array_size)]

    def add(self, element):
        '''
        When an element is added to the bloom filter, murumur3 hashing algorithm is used to get
        as many hashes for the element as specified by hash_count using different repeatable seeds.
        The hash is then used to set the corresponding bit_array index to True
        :param element: Element to be added to bit array
        :return: None
        '''
        for seed in range(self.hash_count):
            # Generate a hash for the element using "seed" as the seed and mod it by bit_array_size
            # to get the index between 0 and (bit_array_size-1) to be set to True in the bit_array
            index = mmh3.hash(element, seed) % self.bit_array_size
            self.bit_array[index] = True

    def contains(self, element):
        '''
        This method checks the bloom filter to see if the element has already been added to the bloom filter.
        Note that since conflicts are possible (the probability of which is given by fp_prob), this method
        can return false positives (i.e. return True when the element is actually not present in the Bloom Filter.
        However it is guaranteed that this will never return false negatives (i.e. return False when the element is
        in fact present in th Bloom Filter.
        :param element: Element to be checked in the Bloom Filter
        :return: Returns a boolean that indicates whether the element is already in the BLoom Filter.
        '''
        for seed in range(self.hash_count):
            # Check all indexes for hashes computed for the element are set to True in bit_array
            index = mmh3.hash(element, seed) % self.bit_array_size
            if not self.bit_array[index]:
                return False
        return True

    @classmethod
    def get_array_size(cls, count, fp_prob):
        '''
        The size of the bit array depends on the number of elements that bloom filter will save (count)
        and the desired false positve probability (fp_prob)
        The bit array size will be greater if the count is higher as well as if lower fp_prob is desired.
        If lower fp_prob is desired more information about the elements need to be stored which increases
        the space requirements for the bit array. The same reasoning goes for increased count.
        bit_array_size is given by the formula -count*ln(fp_prob)/ln(2)^2
        :return: bit_array_size for the bloom filter
        '''
        assert (0 < fp_prob <= 1)
        bit_array_size = -count * np.log(fp_prob) / pow(np.log(2), 2)
        return int(bit_array_size)

    @classmethod
    def get_hash_count(cls, bit_array_size, count):
        '''
        This method computes and returns the number hashes to be computed and stored for each element
        stored in the bloom filter. If the ratio of bit_array_size and count is higher, hash count will be
        higher as more information can be stored per element to reduce the likelihood of conflicts and thus
        achieving a lower false_positive probability.
        :param bit_array_size: Size of the bit array
        :param count: Number of elements/items to be stored in the bloom filter
        :return: Number of hashes to be computed for each element
        '''
        hash_count = bit_array_size * np.log(2) / count
        return int(hash_count)

# Test Code

def generate_random_string():
    return ''.join([random.choice(string.ascii_lowercase) for i in range(3)])

def test_bloom_filter(count, fp_prob):
    bf = BloomFilter(count, fp_prob)
    items_present = []
    items_present_hash = {}
    items_absent = []
    for i in range(count):
        s = generate_random_string()
        items_present.append(s)
        items_present_hash[s] = True
        bf.add(s)
    for i in range(count):
        s = generate_random_string()
        if s not in items_present_hash:
            items_absent.append(s)
    fp_count = 0
    # Check that BloomFilter.contains never returns false negatives
    for item in items_present:
        assert(bf.contains(item))
    for item in items_absent:
        if bf.contains(item):
            # This is a false positive, so increment fp_count
            fp_count += 1
    fp_prob_actual = fp_count / len(items_absent)
    print("False positive rate: ", fp_prob_actual)
    assert(fp_prob_actual <= fp_prob)

test_bloom_filter(10000, 0.1)
test_bloom_filter(10000, 0.05)

