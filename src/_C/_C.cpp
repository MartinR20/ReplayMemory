#include <iostream>
#include <cstring>

// hack to get boost python to compile under Winodws with mingw64
#ifdef _WIN32
#include <cmath>

#define _hypot hypot
#endif

const unsigned int STRIDE = 21;

//from https://www.geeksforgeeks.org/compute-the-minimum-or-maximum-max-of-two-integers-without-branching/
int min(int x, int y)
{
  return  y + ((x - y) & ((x - y) >> 8));
}

int max(int x, int y)
{
  return x - ((x - y) & ((x - y) >> 8));
}

// this could potentially be made faster
unsigned int argmax(double* ptr, unsigned int length) {
  double m = *ptr;
  unsigned int idx = 0;

  for(unsigned int x = 1; x < length; ++x) {
    if(m < *++ptr) {
      m = *ptr;
      idx = x;
    }
  }

  return idx;
}

//from http://graphics.stanford.edu/~seander/bithacks.html
bool sign(signed int x) {
    return (bool)(x >> (sizeof(int) * 8 - 1));
}

class Map{
    private:
      unsigned char* grid;
      const unsigned int stride;

    public:
      Map(const unsigned int size) : grid(new unsigned char[size * size]{0}), stride(size) {}

      void print() const{
        for(unsigned int idx = 0; idx < stride * stride; ++idx)
            std::cout << (!(idx % stride) ? '\n' : ' ') << (int)grid[idx];
	std::cout << '\n';
      }

      unsigned char* operator() (unsigned x, unsigned y) const {
        return grid + y * stride + x;
      }

      unsigned char* neigh4(unsigned char* point) const {
        if(*point == *(point + 1))
          return point + 1;
        else if(*point == *(point - 1))
          return point - 1;
        else if(*point == *(point + stride))
          return point + stride;
        else
          return point - stride;
      }

      void line(unsigned char* point, unsigned char color, unsigned int length) {
        memset(point, color, length);
      }

      void clear(){
        memset(grid, 0, stride * stride);
      }
};

struct Snake{
    unsigned int  x, y;
    unsigned char* head;
    unsigned char* tail;
    unsigned int length;

    void reset(unsigned int x, unsigned int y, unsigned int length) {
      this->x = x + length - 1;
      this->y = y;
      this->length = length;
    }
};

class Game{
    private:
      Map map;
      Snake snake;

      unsigned char* food;

      signed int reward = 0;
      signed int last_reward = 0;
      unsigned int ticker = 0;

      unsigned int death_penalty = 20;
      unsigned int food_reward = 10;
      unsigned int living_reward = 1;

      void init_snake(unsigned int x, unsigned int y) {
        snake.reset(x, y, 4);

        snake.tail = map(x, y);
        snake.head = snake.tail + snake.length;

        map.line(snake.tail, 1, snake.length);
      }

      void init_food(){
        food = map(rand() % STRIDE, rand() % STRIDE);

        while(true){
          // check for value of random picked position
          if(*food){
            ++food;
            continue;

          // check if moved out of array (not safe)
          } else if(food > map(STRIDE - 1, STRIDE - 1)) {
            food = map(0,0);

          // set food if free space is found found
          } else {
            *food = 255;
            break;
          }
        }
      }

    public:
      Game() : map(STRIDE) {
        init_snake(1, 1);
        init_food();
      }

      void print() const {
        map.print();
      }

      void seed(const int seed) const {
        srand(seed);
      }

      void step(int action){
        switch(action){
            case 0: // RIGHT
                snake.x = min(20, snake.x + 1);
                break;
            case 1: // LEFT
                snake.x = max(0, (int)snake.x - 1);
                break;
            case 2: // UP
                snake.y = max(0, (int)snake.y - 1);
                break;
            case 3: // DOWN
                snake.y = min(20, snake.y + 1);
                break;
            default:
                throw "Not a valid Action!";
        }

        snake.head = map(snake.x, snake.y);

        if(*snake.head == 1) {
            // death_penalty
            reward -= death_penalty;

            reset();
        } else {
            *snake.head = 1;
        }

        if(snake.head != food) {
          unsigned char* new_tail = map.neigh4(snake.tail);

          *snake.tail = 0;
          snake.tail = new_tail;
        } else {
          // food_reward
          reward += food_reward;

          ++snake.length;
          init_food();
        }

        // living_reward
        reward += living_reward;

        ++ticker;
      }

      signed int get_reward() {
        signed int m_reward = this->reward - this->last_reward;
        this->last_reward = this->reward;

        return m_reward;
      }

      void reset(){
        map.clear();
        init_snake(1,1);
        init_food();
      }

      unsigned char* get_map(){
        return map(0,0);
      }
};

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

namespace np = boost::python::numpy;
namespace p =  boost::python;

class GameWrapper : public Game {
  private:
    const np::dtype dt = np::dtype::get_builtin<unsigned char>();

    const p::tuple shape = p::make_tuple(STRIDE * STRIDE);
    const p::tuple stride = p::make_tuple(sizeof(unsigned char));
    const p::object own;

  public:
    p::tuple step(np::ndarray __action) {
      unsigned int action = argmax((double*)__action.get_data(), 4);

      Game::step(action);

      np::ndarray obs = np::from_data(Game::get_map(), dt, shape, stride, own);
      signed int reward = Game::get_reward();

      return p::make_tuple(obs, reward, sign(reward), 0);
    }

    np::ndarray reset() {
      Game::reset();

      return np::from_data(Game::get_map(), dt, shape, stride, own);
    }
};

BOOST_PYTHON_MODULE(_C)
{
  using namespace boost::python;

  Py_Initialize();
  np::initialize();

  class_<GameWrapper>("Game")
      .def("print", &GameWrapper::print)
      .def("seed", &GameWrapper::seed)
      .def("step", &GameWrapper::step)
      .def("reset", &GameWrapper::reset)
  ;
}

/*
int main() {
    Game game = Game();

    game.step(rand() % 4);

    return 0;
}
*/
