#define CPP_ONLY

#include <torch/torch.h>
#include "ReplayMemory.cpp"

float randf() {
  return (float)(rand()) / (float)(RAND_MAX);
}

int main() {
    const unsigned int size = 50;
    const unsigned int action_space = 5;
    const unsigned int max_rand = 50;

    Rnd rnd = Rnd();

    ReplayMemory mem = ReplayMemory(size, 4, 3, 0.99, 0.4, 0.5, "");

    for(unsigned int i = 0; i < size; ++i) {
      const int sample1 = rnd.sample(0, action_space);
      const int sample2 = rnd.sample(0, max_rand);
      const int bool_sample = rnd.sample(0, 2);

      mem.append(torch::ones({84, 84}), sample1, sample2 * randf(), (bool)bool_sample);    
    }

    mem.update_priorities(torch::arange({(float)size}), torch::randint(0, max_rand, {size}));

    auto sample = mem.sample(5);

    std::cout << "---------------------------------------- idx ----------------------------------------\n";
    std::cout << std::get<0>(sample) << '\n';
    std::cout << "---------------------------------------- state ----------------------------------------\n";
    std::cout << std::get<1>(sample)[0][0] << '\n';
    std::cout << "---------------------------------------- action ----------------------------------------\n";
    std::cout << std::get<2>(sample) << '\n';
    std::cout << "---------------------------------------- R ----------------------------------------\n";
    std::cout << std::get<3>(sample) << '\n';
    std::cout << "---------------------------------------- next_state ----------------------------------------\n";
    std::cout << std::get<4>(sample)[0][0] << '\n';
    std::cout << "---------------------------------------- nonterminal ----------------------------------------\n";
    std::cout << std::get<5>(sample) << '\n';
    std::cout << "---------------------------------------- weights ----------------------------------------\n";
    std::cout << std::get<6>(sample) << '\n';

    return 0;
}