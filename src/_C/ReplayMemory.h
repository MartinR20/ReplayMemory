#include "utils.h"
#include <torch/extension.h>
#include "SegmentTree.h"
#include <random>
#include <vector>
#include <cmath>
#include <unistd.h>
#include <functional>

struct TransitionReference {
  unsigned int** _timesteps;
  at::Tensor** _states;
  unsigned int** _actions;
  float** _rewards;
  uint8_t** _nonterminals;

  TransitionReference(unsigned int capacity) :
                  _timesteps(new unsigned int*[capacity]), 
                  _states(new at::Tensor*[capacity]), 
                  _actions(new unsigned int*[capacity]),
                  _rewards(new float*[capacity]),
                  _nonterminals(new uint8_t*[capacity]) {}
  
  unsigned int timesteps(unsigned int i) {
    return *_timesteps[i];
  }

  at::Tensor states(unsigned int i) {
    return *_states[i];
  }

  unsigned int actions(unsigned int i) {
    return *_actions[i];
  }

  float rewards(unsigned int i) {
    return *_rewards[i];
  }

  uint8_t nonterminals(unsigned int i) {
    return *_nonterminals[i];
  }
};

struct TransitionContainer {
  unsigned int* timesteps;
  at::Tensor* states;
  unsigned int* actions;
  float* rewards;
  uint8_t* nonterminals;
  unsigned int idx = 0;

  TransitionContainer(unsigned int capacity) :
                  timesteps(new unsigned int[capacity]), 
                  states(new at::Tensor[capacity]), 
                  actions(new unsigned int[capacity]),
                  rewards(new float[capacity]),
                  nonterminals(new uint8_t[capacity]) {}

  TransitionContainer(unsigned int timestep, 
                      at::Tensor state, 
                      unsigned int action, 
                      float reward, 
                      bool nonterminal) : TransitionContainer(1)
  {
    timesteps[0] = timestep;
    states[0] = state;
    actions[0] = action;
    rewards[0] = reward;
    nonterminals[0] = (uint8_t)nonterminal;
  }

  void append(unsigned int timestep,
            at::Tensor state,
            unsigned int action,
            float reward,
            bool terminal) 
  {
    timesteps[idx] = timestep;
    /*
      * note removing .mul(255).to(torch::kInt8).to(torch::kCPU) yield a performence 
      * increase of about 2.5 fold
    */
    states[idx] = state[-1].mul(255).to(torch::kInt8).to(torch::kCPU); 
    actions[idx] = action;
    rewards[idx] = reward;
    nonterminals[idx] = (uint8_t)!terminal;
    
    ++idx;
  }

  void set(TransitionContainer* other, unsigned int m, unsigned int n) const {
    other->timesteps[m] = timesteps[n];
    other->states[m] = states[n];
    other->actions[m] = actions[n];
    other->rewards[m] = rewards[n];
    other->nonterminals[m] = nonterminals[n];
  }

  void set(TransitionReference* other, unsigned int m, unsigned int n) const {
    other->_timesteps[m] = &timesteps[n];
    other->_states[m] = &states[n];
    other->_actions[m] = &actions[n];
    other->_rewards[m] = &rewards[n];
    other->_nonterminals[m] = &nonterminals[n];
  }
};

const TransitionContainer blank_trans = TransitionContainer(0, torch::zeros({84, 84}), INT_MAX, 0, false);  

class ReplayMemory {
  private:
    const unsigned int capacity;
    const unsigned int history;
    unsigned int t = 0;
    unsigned int current_idx = 0;
    const unsigned int steps; //n
    const float discount;
    const float priority_exponent;
    unsigned int priority_max = 1;
    const at::Device device;
    AppendableSegmentTree<float> segtree;
    TransitionContainer transitions;

  public:
    float priority_weight;

    ReplayMemory(const unsigned int capacity, 
                  const unsigned int history, 
                  const unsigned int steps,
                  const float discount,
                  const float priority_weight,
                  const float priority_exponent,
                  const std::string device)
                : capacity(capacity),
                  history(history),
                  steps(steps),
                  discount(discount),
                  priority_exponent(priority_exponent),
                  device(device == "gpu" ? torch::kCUDA : torch::kCPU),
                  segtree(AppendableSegmentTree<float>(capacity)),
                  transitions(capacity),
                  priority_weight(priority_weight)

    {
      /*
        * Arguments:
        *   capacity: Capacity of Buffer
        *   history: ...
        *   steps: multistepnumber
        *   discount: discount value
        *   priority_weight: ...
        *   priority_exponent: ...
        *   device: "gpu" if GPU Tensors are intended, everything else for CPU
      */
    }

    void append(at::Tensor state,
            unsigned int action,
            float reward,
            bool terminal) 
    {
      segtree.append(priority_max);
      transitions.append(t, state, action, reward, !terminal);
      t = terminal ? 0 : t+1;
    }

#ifndef CPP_ONLY
    py::tuple sample(unsigned int batch_size) 
#else
    std::vector<at::Tensor> sample(unsigned int batch_size) 
#endif
    {
      /*
        * Samples Transitions based on a standard uniform distribution
        * "weighted" by a priorty score for each frame
      */

#ifndef CPP_ONLY
      if(!(history > 2)) throw py::value_error();
#else
      assert(history > 2);
#endif

      float p_total = segtree.total();
      float segment = p_total / batch_size;
      utils::Rnd rnd = utils::Rnd();

      std::vector<float> probs(batch_size);
      std::vector<unsigned int> idxs(batch_size);
      at::Tensor t_states = torch::empty({batch_size, history, 84, 84}, torch::kUInt8);
      std::vector<unsigned long> actions(batch_size);
      std::vector<float> R(batch_size);
      at::Tensor t_next_states = torch::empty({batch_size, history, 84, 84}, torch::kUInt8);
      std::vector<uint8_t> nonterminals(batch_size);

      for(unsigned int i = 0; i < batch_size; ++i) {
        //_get_sample_from_segment
        float sample;
        unsigned int idx;
        float prob;
        
        bool valid = false;

        // in this loop there is a bug that causes it to get an infinite loop
        // TODO: find bug
        while(!valid) {
          sample = rnd.sample(i * segment, (i + 1) * segment);
          idx = this->segtree.find(sample);
          prob = this->segtree.get(idx);

          if((this->segtree.idx - idx) % this->capacity > this->steps && (idx - this->segtree.idx) % this->capacity >= this->history && prob != 0) 
            valid = true;
        }

        idxs[i] = idx;
        probs[i] = prob;

        ////_get_transition
        TransitionReference transition = TransitionReference(this->history + this->steps);
        this->transitions.set(&transition, this->history - 1, idx);

        for(signed int t = this->history - 2; t >= 0; --t) {
          if(transition.timesteps(t + 1) != 0) {
            this->transitions.set(&transition, t, idx - this->history + 1 + t);
          } else {
            blank_trans.set(&transition, t, 0);
          }
        }

        for(unsigned int t = this->history; t < this->history + this->steps; ++t) {
          if(transition.nonterminals(t - 1)) {
            this->transitions.set(&transition, t, idx - this->history + 1 + t);
          } else {
            blank_trans.set(&transition, t, 0);
          }
        }
        ////

        for(unsigned int t = 0; t < this->history; t++) {
          t_states[i][t] = *transition._states[t];
          t_next_states[i][t] = *transition._states[this->steps + t];
        }

        float sum = 0;
        for(unsigned int k = 0; k < this->steps; ++k)
          sum += std::pow(this->discount, k) * transition.rewards(this->history + k - 1);

        actions[i] = transition.actions(this->history - 1);
        R[i] = sum;
        nonterminals[i] = transition.nonterminals(this->history + this->steps - 1);
        //
      }

      at::Tensor weights;
      at::Tensor t_idxs;
      at::Tensor t_actions;
      at::Tensor t_R;     
      at::Tensor t_nonterminals;

      weights = torch::empty({batch_size}, torch::dtype(torch::kFloat32).requires_grad(false));
      std::copy_n(probs.data(), batch_size, weights.data<float>()); 
      weights
        .div_((float)p_total)
        .mul_((float)(segtree.full() ? capacity : transitions.idx))
        .pow_(-priority_weight)
        .div_(at::max(weights));
      weights = weights.to(device);

      t_idxs = torch::empty({batch_size}, torch::dtype(torch::kInt32).requires_grad(false));
      std::copy_n(idxs.data(), batch_size, t_idxs.data<int>()); 

      t_actions = torch::empty({batch_size}, torch::dtype(torch::kInt64).requires_grad(false));
      std::copy_n(actions.data(), batch_size, t_actions.data<long>()); 
      t_actions = t_actions.to(device);

      t_R = torch::empty({batch_size}, torch::dtype(torch::kFloat32).requires_grad(false)); 
      std::copy_n(R.data(), batch_size, t_R.data<float>()); 
      t_R = t_R.to(device);

      t_nonterminals = torch::empty({batch_size}, torch::dtype(torch::kUInt8).requires_grad(false));
      std::copy_n(nonterminals.data(), batch_size, t_nonterminals.data<uint8_t>());
      t_nonterminals = t_nonterminals.view({batch_size, 1}).to(torch::kFloat32).to(device);        
  
      t_states = t_states.to(torch::kFloat32).to(device);
      t_next_states = t_next_states.to(torch::kFloat32).to(device);

#ifndef CPP_ONLY
      return py::make_tuple(t_idxs, t_states, t_actions, t_R, t_next_states, t_nonterminals, weights);
#else
      return std::vector<at::Tensor>({t_idxs, t_states, t_actions, t_R, t_next_states, t_nonterminals, weights});
#endif
    }

    void update_priorities(at::Tensor __idxs, at::Tensor __priorities) {
#ifndef CPP_ONLY
      if(__idxs.size(0) != __priorities.size(0)) throw py::index_error();
#else
      assert(__idxs.size(0) == __priorities.size(0));
#endif

      __priorities.pow_(priority_exponent);

      int* idxs = __idxs.data<int>();
      float* priorities = __priorities.data<float>();

      for(unsigned int i = 0; i < __idxs.size(0); ++i) {
        priority_max = utils::max(priority_max, priorities[i]);
        segtree.update(idxs[i], priorities[i]);
      }
    }

    ReplayMemory* __iter__() {
      current_idx = 0;
      return this;
    }

    at::Tensor __next__() {
#ifndef CPP_ONLY
        if(current_idx == capacity) throw py::stop_iteration();
#else
        assert(current_idx != capacity);        
#endif

      at::Tensor state_stack = torch::empty({history, 84, 84});
      state_stack[history - 1] = transitions.states[current_idx];
      unsigned int prev_timestep = transitions.timesteps[current_idx];

      for(signed int t = history - 1; t >= 0; --t) {
        if(prev_timestep) {
          state_stack[t] = transitions.states[current_idx + t - history + 1];
          prev_timestep -= 1;
        } else {
          state_stack[t] = blank_trans.states[0];
        }
      }

      ++current_idx;
      return state_stack;
    }

};

/*
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

    mem.sample(1);

    return 0;
}
*/