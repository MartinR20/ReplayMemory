// Copyright 2019 Reichholf Martin
#define CPP_ONLY

#include "ReplayMemory.cpp"
#include <torch/torch.h>
#include <chrono>

float randf() {
  return (float)(rand()) / (float)(RAND_MAX);
}

int main() {
    const unsigned int size = 10000;
    const unsigned int action_space = 5;
    const unsigned int max_rand = 50;

    utils::Rnd rnd = utils::Rnd();

    ReplayMemory mem = ReplayMemory(size, 4, 3, 0.99, 0.4, 0.5, "");

    for(unsigned int i = 0; i < size; ++i) {
      const int sample1 = rnd.sample(0, action_space);
      const int sample2 = rnd.sample(0, max_rand);
      const int bool_sample = rnd.sample(0, 2);

      mem.append(torch::ones({1, 84, 84}, torch::kInt8), sample1, sample2 * randf(), (bool)bool_sample);    
    }

    mem.update_priorities(torch::arange({(float)size}, torch::kInt32), torch::randint(0, max_rand, {size}, torch::kFloat32));

    using namespace std::chrono;
    auto start = high_resolution_clock::now();

    for(unsigned int i = 0; i < 50; ++i) {
        auto sample = mem.sample(32);
    }

    auto end = high_resolution_clock::now();
    std::cout << duration_cast<milliseconds>(end - start).count() << '\n';

    exit(0);
}
