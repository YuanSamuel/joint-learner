#ifndef GLIDER_PREDICTOR_H
#define GLIDER_PREDICTOR_H

using namespace std;
#include <map>
#include <numeric>
#include <vector>

#include "helper_function.h"
#include "optgen.h"

#define MAX_PCMAP 31
#define PCMAP_SIZE 16 // Assume 16

// Constants
#define GLIDER_THRESHOLD_HIGH 60 // Prediction threshold
#define GLIDER_THRESHOLD_LOW 0

enum class Prediction { High, Medium, Low };

class IntegerSVM
{
private:
  vector<int> weights;

public:
  IntegerSVM(int num_weights) { weights.resize(num_weights, 0); }

  void update_weights(const vector<uint64_t>& indices, int should_cache, int thresholds, int learning_rate = 1, int lambda = 1,
                      int regularization_threshold = 5)
  {
    // Calculate the sum of the weights
    int weight_sum = accumulate(indices.begin(), indices.end(), 0, [this](int sum, int idx) { return sum + this->weights[idx]; });

    // Check if the threshold is exceeded
    if (abs(weight_sum) >= thresholds) {
      return; // Do not update weights if exceeded
    }

    // Update weights based off OPTGen decisions
    for (auto idx : indices) {
      weights[idx] += learning_rate * should_cache;
    }

    // Step regularization
    for (auto& weight : weights) {
      if (abs(weight) > regularization_threshold) {
        if (weight > 0) {
          weight -= lambda;
        } else {
          weight += lambda;
        }
      }
    }
  }

  // Cakcykate the sum of weights based on the PCHR index
  int calculate_weights(vector<uint64_t> pchr)
  {
    int weight_sum = 0;
    for (uint64_t pc : pchr) {
      weight_sum += weights[pc];
    }
    return weight_sum;
  }

  // Calculate the sum of all weights
  int calculate_sum() const { return accumulate(weights.begin(), weights.end(), 0); }

  Prediction predict(const vector<int>& indices)
  {
    // Calcualte predicted value
    int prediction = accumulate(indices.begin(), indices.end(), 0, [this](int sum, int idx) { return sum + this->weights[idx]; });
    return prediction >= 60 ? Prediction::High : prediction >= 0 ? Prediction::Medium : Prediction::Low;
  }

  // Print weights (debug)
  void print_weights() const
  {
    for (int weight : weights) {
      cout << weight << " ";
    }
    cout << endl;
  }
};

class Glider_Predictor
{
private:
  vector<uint64_t> pchr;                                          // PCHR history
  const int k_sparse = 5;                                         // PCHR max length
  const vector<int> dynamic_thresholds = {0, 30, 100, 300, 3000}; // Dynamic thresholds
  vector<IntegerSVM> isvms;

public:
  // Init OPTGen and PCHR
  Glider_Predictor()
  {
    pchr.resize(k_sparse, 0);

    for (int i = 0; i < PCMAP_SIZE; i++) {
      isvms.push_back(IntegerSVM(PCMAP_SIZE));
    }
  }

  // Dynamically adjust thresholds
  int select_dynamic_threshold()
  {
    // Sum current weights in ISVM
    int weight_sum = 0;
    for (const auto& svm : isvms) {
      weight_sum += svm.calculate_sum();
    }

    // Choose approx. threshold based on sum of weights
    size_t index;
    if (weight_sum < 500) {
      index = 4;
    } else if (weight_sum < 1000) {
      index = 3;
    } else if (weight_sum < 2000) {
      index = 2;
    } else if (weight_sum < 4000) {
      index = 1;
    } else {
      index = 0;
    }
    return dynamic_thresholds[index];
  }

  // Return prediction results
  Prediction get_prediction(uint64_t PC)
  {
    // Push PC to end of PCHR history, pop beginning
    uint64_t encoded_pc = CRC(PC) % PCMAP_SIZE; // CRC hash with mod 16
    pchr.push_back(encoded_pc);
    if (pchr.size() > (size_t)k_sparse) {
      pchr.erase(pchr.begin());
    }

    int weight_sum = isvms[encoded_pc].calculate_weights(pchr);

    if (weight_sum >= GLIDER_THRESHOLD_HIGH) {
      return Prediction::High;
    } else if (weight_sum < GLIDER_THRESHOLD_LOW) {
      return Prediction::Low;
    }
    return Prediction::Medium;
  }

  void increase(uint64_t PC)
  {
    uint64_t encoded_pc = CRC(PC) % PCMAP_SIZE;

    // Choose appropriate dynamic threshold
    int selected_threshold = select_dynamic_threshold();

    isvms[encoded_pc].update_weights(pchr, 1, selected_threshold);
  }

  void decrease(uint64_t PC)
  {
    uint64_t encoded_pc = CRC(PC) % PCMAP_SIZE;

    // Choose appropriate dynamic threshold
    int selected_threshold = select_dynamic_threshold();

    isvms[encoded_pc].update_weights(pchr, -1, selected_threshold);
  }

  // Print all weights of all ISVMs
  void print_all_weights() const
  {
    cout << "Printing all SVM weights:" << endl;
    int svm_index = 1;
    for (const auto& svm : isvms) {
      cout << "SVM #" << svm_index << ": ";
      svm.print_weights();
      svm_index++;
    }
    cout << "----------" << endl;
  }
};

#endif