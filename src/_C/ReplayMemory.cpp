#include <torch/extension.h>
#include "SegmentTree.cpp"

class ReplayMemory {
  private:
    unsigned int capacity;
    unsigned int history;
    unsigned int idx = 0;
    unsigned int t = 0;
    unsigned int steps; //n
    AppendableSegmentTree<float> segtree;
    
    unsigned int* timesteps;
    at::Tensor* states;
    unsigned int* actions;
    float* rewards;
    bool* nonterminals;

  public:
    ReplayMemory(const unsigned int capacity, 
                  const unsigned int history, 
                  const unsigned int steps)
                : capacity(capacity),
                  history(history),
                  steps(steps),
                  segtree(AppendableSegmentTree<float>(capacity)),
                  timesteps(new unsigned int[capacity]), 
                  states(new at::Tensor[capacity]), 
                  actions(new unsigned int[capacity]),
                  rewards(new float[capacity]),
                  nonterminals(new bool[capacity]) {}

    void append(unsigned int timestep,
            at::Tensor state,
            unsigned int action,
            float reward,
            bool terminal) 
    {
      segtree.append(segtree.total());
      
      timesteps[idx] = t;
      states[idx] = state[-1].mul(255).to(torch::kInt8).to(torch::kCPU); // TODO: probably better way to do this
      actions[idx] = action;
      rewards[idx] = reward;
      nonterminals[idx] = !terminal;
      
      ++idx;
      t = terminal ? 0 : t+1;
    }

    py::tuple sample(unsigned int batch_size) 
    {
      return py::make_tuple(0,0);
    }
};