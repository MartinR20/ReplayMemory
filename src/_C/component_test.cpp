// Copyright 2019 Reichholf Martin
#define CPP_ONLY

#include <torch/torch.cpp>
#include "ReplayMemory.cpp"
#include <stdlib.h>


float randf() {
  return (float)(rand()) / (float)(RAND_MAX);
}

int main(int argc, char *argv[]) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now(); 

    const unsigned int size = 100000;
    const unsigned int action_space = 5;
    const unsigned int max_rand = 50;

    Rnd rnd = Rnd();

    ReplayMemory mem = ReplayMemory(size, 4, 3, 0.99, 0.4, 0.5, "");

    for(unsigned int i = 0; i < size; ++i) {
      const int sample1 = rnd.sample(0, action_space);
      const int sample2 = rnd.sample(0, max_rand);
      const int bool_sample = rnd.sample(0, 2);

      mem.append(torch::ones({1, 84, 84}, torch::kInt8), sample1, sample2 * randf(), (bool)bool_sample);    
    }

    mem.update_priorities(torch::arange({(float)size}, torch::kInt32), torch::randint(0, max_rand, {size}, torch::kFloat32));

    auto sample = mem.sample(32);

    std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms taken" << std::endl;

    if(argc > 1 && std::string(argv[1]) == std::string("verbose")) {
      std::cout << "---------------------------------------- idx ----------------------------------------\n";
      std::cout << sample[0] << '\n';
      std::cout << "---------------------------------------- state ----------------------------------------\n";
      std::cout << sample[1][0][0] << '\n';
      std::cout << "---------------------------------------- action ----------------------------------------\n";
      std::cout << sample[2] << '\n';
      std::cout << "---------------------------------------- R ----------------------------------------\n";
      std::cout << sample[3] << '\n';
      std::cout << "---------------------------------------- next_state ----------------------------------------\n";
      std::cout << sample[4][0][0] << '\n';
      std::cout << "---------------------------------------- nonterminal ----------------------------------------\n";
      std::cout << sample[5] << '\n';
      std::cout << "---------------------------------------- weights ----------------------------------------\n";
      std::cout << sample[6] << '\n';
    }

    exit(0);
}
