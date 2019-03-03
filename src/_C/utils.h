#include <random>

#define CHAR_BIT 8

namespace utils {
  int max(int x, int y) {
    return x - ((x - y) & ((x - y) >> (sizeof(int) * CHAR_BIT - 1)));
  }

  class Rnd {
    private:
      std::mt19937 gen; //Standard mersenne_twister_engine seeded with std::random_device

    public:
      Rnd() : gen((std::random_device())()) {}
        
      float sample(float lower, float upper) {
        return std::uniform_real_distribution<>(lower, upper)(gen);
      }
  };
};
