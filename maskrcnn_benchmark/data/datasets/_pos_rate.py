import random
import pdb
from collections import defaultdict
import numpy
import numpy as np
import math
class PosRateControllerLength():
    def __init__(self, max_length = 9, center_length = 8):
        self.leng_to_controller = [PosRateController() for i in range(max_length + 1)]
        self.max_length = max_length
        self.center_length = center_length
        self.pos_rates = []
        self.lengths = []
    def __call__(self, pos_num, neg_num):
        # first sample the query length
        length = numpy.random.normal(self.center_length, 5.0)
        # cap to 1 and max_length
        length = max(1, min(self.max_length, length))
        length = round(length)
        length = min(pos_num + neg_num, length)

        pos_num, neg_num = self.leng_to_controller[length](pos_num, neg_num, desired_length = length)
        return pos_num, neg_num
    
    def update_true_pos_rate(self, pos_num, total_num):
        if total_num == 0:
            return
        self.pos_rates.append(pos_num / total_num)
        self.lengths.append(total_num)
        total_num = int(min(total_num, self.max_length))
        self.leng_to_controller[total_num].update_true_pos_rate(pos_num, total_num)
        
        # if len(self.pos_rates) % 1000 == 0:
        #     print(self.pos_rates)
        #     print(self.lengths)
        #     for i in range(len(self.leng_to_controller)):
        #         print("length: ", i)
        #         print("overall pos rate: ", sum(self.leng_to_controller[i].pos_rates) / max(1.0, len(self.leng_to_controller[i].pos_rates)))
    
class PosRateController():
    def __init__(self, bin_num = 10, adhoc_bin_weights = {}, control_length = -1): 
        self.bins = [1.0 / bin_num * i for i in range(bin_num + 1)]
        self.bin_counter = [0 for i in range(bin_num + 1)]

        self.adhoc_bin_weights = adhoc_bin_weights # this is a list of weights for each bin
        self.slack = 20 # we can allow some slack for the pos rate control
        self.pos_rates = []
        self.lengths = []
        
    def _find_closest_bin(self, pos_rate, valid_bins):
        valid_bins_rate = [self.bins[i] for i in valid_bins]
        # determine the pos rate is in which bin
        # find the closes bin to the current pos rate
        bin_index = valid_bins[0]
        min_diff = abs(pos_rate - valid_bins_rate[0])

        for i in range(1, len(valid_bins)):
            diff = abs(pos_rate - valid_bins_rate[i])
            if diff < min_diff:
                bin_index = valid_bins[i]
                min_diff = diff
            if diff == min_diff and random.random() > 0.5:
                bin_index = valid_bins[i]
                min_diff = diff
        return bin_index

    def __call__(self, pos_num, neg_num, desired_length = -1):
        if pos_num == 0 and neg_num == 0:
            return 0, 0
        if pos_num == 1 and neg_num == 0:
            return 1, 0

        pos_now = pos_num / (pos_num + neg_num)
        
        min_bin_counter = min([self.bin_counter[i] * self.adhoc_bin_weights.get(i, 1.0) for i in range(len(self.bin_counter)) ])
        valid_bins = [i for i in range(len(self.bin_counter)) if self.bin_counter[i] * self.adhoc_bin_weights.get(i, 1.0) <= min_bin_counter + self.slack] # these are the bins this example could go to
        bin_index = random.choice(valid_bins)
        #self._find_closest_bin(pos_now, valid_bins)

        if desired_length > 0:
            # control to the desired length
            desired_pos = round(desired_length * self.bins[bin_index])
            pos_num = min(pos_num, desired_pos)
            if self.bins[bin_index] == 0:
                neg_num = min(neg_num, desired_length)
            else:
                neg_num = min(neg_num, round(pos_num / self.bins[bin_index] * (1 - self.bins[bin_index])))
        else:
            # let's control the pos_rate to the desired rate
            if pos_now == self.bins[bin_index]:
                pass
            elif pos_now < self.bins[bin_index]:
                # this means we need to drop some negative examples
                neg_num = round(pos_num / self.bins[bin_index] - pos_num)
            else:
                # this means we need to drop some positive examples
                pos_num = round(neg_num * self.bins[bin_index] / (1 - self.bins[bin_index]))
        
        # new_bin_index = self._find_closest_bin(pos_num / (pos_num + neg_num), list(range(len(self.bins))))
        # if new_bin_index != bin_index and len(self.pos_rates) > 1000:
        #     pdb.set_trace()

        # self.bin_counter[new_bin_index] += 1
        # self.pos_rates.append(pos_num / (pos_num + neg_num))
        # make sure we don't have all 0s
        if pos_num == 0 and neg_num == 0:
            pos_num = 1
            neg_num = 0

        return pos_num, neg_num
    
    def update_true_pos_rate(self, pos_num, total_num):
        if total_num == 0: # ignore
            return
        pos_rate = pos_num / total_num
        bin_index = self._find_closest_bin(pos_rate, list(range(len(self.bins))))
        self.bin_counter[bin_index] += 1
        self.pos_rates.append(pos_rate)
        self.lengths.append(total_num)
        # if len(self.pos_rates) % 1000 == 0:
        #     print(self.pos_rates)
        #     for i in self.pos_rate_by_lengths:
        #         print(i, len(self.pos_rate_by_lengths[i]), sum(self.pos_rate_by_lengths[i]) / len(self.pos_rate_by_lengths[i]))
    def report(self,):
        #print(self.lengths)
        print(np.mean(self.lengths), self.bin_counter)
from scipy.stats import norm
class PosRateControllerV2():
    def __init__(self, max_length, center_length, scale = 4.0):
        self.max_length = max_length
        self.center_length = center_length
        self.bins = defaultdict(int)
        for i in range(1, max_length + 1):
            for j in range(0, i + 1):
                self.bins[(i, j)] = 0
        
        # calculate the weights according to a normal distribution centered on center_length
        dis = norm(loc = center_length, scale = scale)

        self.weights = {}
        for i in range(1, max_length+1):
            self.weights[i] = dis.cdf(i + 0.5) - dis.cdf(i - 0.5)
        # print(self.weights)
        # renormalize the weights
        total_weight = sum(self.weights.values())
        for i in self.weights:
            self.weights[i] /= total_weight

        self.weights_pos_rate = {}
        
        # do a slight reweight
        self.pos_rates = []

        self.slack = 10

    def __call__(self, pos_num, neg_num, max_cap_num = -1):
        # find the most good matching bin
        
        valid_keys = []
        for key in self.bins:
            if key[0] <= pos_num + neg_num and key[1] <= pos_num and key[0] - key[1] <= neg_num and (max_cap_num == -1 or key[0] <= max_cap_num):
                valid_keys.append(key)
        # find the min count in the valid keys
        if len(valid_keys) == 0:
            print(pos_num, neg_num)
            return pos_num, neg_num
        min_counter = min([self.bins[key] / self.weights[key[0]] for key in valid_keys])
        valid_keys = [key for key in valid_keys if self.bins[key] / self.weights[key[0]] <= min_counter + self.slack] # rescreened

        # find the counter where we drop the minimal number of examples
        closest_key = None
        min_diff = 100
        for key in valid_keys:
            diff = abs(key[1] - pos_num)
            if diff < min_diff:
                closest_key = key
                min_diff = diff
    
        if closest_key is None:
            return pos_num, neg_num
    
        return closest_key[1], closest_key[0] - closest_key[1]
    
    def update_true_pos_rate(self, pos_num, total_num):
        if total_num == 0:
            return
        self.bins[(total_num, pos_num)] += 1
        self.pos_rates.append(pos_num / total_num)
    
    def report(self):
        if len(self.pos_rates) % 1000 != 0:
            return

        for i in range(1, self.max_length + 1):
            print("length", i, sum([self.bins[(i, j)] for j in range(0, i + 1)]))
            for j in range(0, i + 1):
                print("  pos", j, " ", self.bins[(i, j)])
        print("\n\n")
        

'''
import matplotlib.pyplot as plt
# drop a histogram
plt.hist(data, bins = 10)
plt.show()
'''


