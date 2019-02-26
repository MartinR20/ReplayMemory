#include <torch/extension.h>
#include "SegmentTree.cpp"
#include <random>
#include <vector>
#include <cmath>

class Rnd {
  private:
    std::mt19937 gen; //Standard mersenne_twister_engine seeded with std::random_device

  public:
    Rnd() : gen((std::random_device())()) {}
  
    int sample(int lower, int upper) {
      return std::uniform_int_distribution<>(lower, upper)(gen);
    }
};

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
    return (*_timesteps)[i];
  }

  at::Tensor states(unsigned int i) {
    return (*_states)[i];
  }

  unsigned int actions(unsigned int i) {
    return (*_actions)[i];
  }

  float rewards(unsigned int i) {
    return (*_rewards)[i];
  }

  uint8_t nonterminals(unsigned int i) {
    return (*_nonterminals)[i];
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
    states[idx] = state[-1].mul(255).to(torch::kInt8).to(torch::kCPU); // TODO: probably better way to do this
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

const TransitionContainer blank_trans = TransitionContainer(0, torch::zeros({84, 84}), UINT_MAX - 1, 0, false);  

class ReplayMemory {
  private:
    const unsigned int capacity;
    const unsigned int history;
    unsigned int t = 0;
    unsigned int current_idx = 0;
    const unsigned int steps; //n
    const float discount;
    const float priority_weight;
    const float priority_exponent;
    const at::Device device;
    AppendableSegmentTree<float> segtree;
    TransitionContainer transitions;

  public:
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
                  priority_weight(priority_weight),
                  priority_exponent(priority_exponent),
                  device(device == "gpu" ? torch::kCUDA : torch::kCPU),
                  segtree(AppendableSegmentTree<float>(capacity)),
                  transitions(capacity) {}

    void append(at::Tensor state,
            unsigned int action,
            float reward,
            bool terminal) 
    {
      segtree.append(segtree.total());
      transitions.append(t, state, action, reward, !terminal);
      t = terminal ? 0 : t+1;
    }

#ifndef CPP_ONLY
    py::tuple sample(unsigned int batch_size) 
#else
    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
               at::Tensor, at::Tensor> sample(unsigned int batch_size) 
#endif
    {
      unsigned int p_total = (unsigned int)segtree.total();
      assert(p_total != 0);
      
      unsigned int segment = p_total / batch_size;
      Rnd rnd = Rnd();

      std::vector<float> probs(batch_size);
      std::vector<unsigned int> idxs(batch_size);
      std::vector<at::Tensor> states(batch_size * history);
      std::vector<unsigned int> actions(batch_size * (history + steps));
      std::vector<float> R(batch_size);
      std::vector<at::Tensor> next_states(batch_size * history);
      std::vector<uint8_t> nonterminals(batch_size);

      for(unsigned int i = 0; i < batch_size; ++i) {
        //_get_sample_from_segment
        unsigned int sample;
        unsigned int idx;
        float prob;
        
        bool valid = false;

        while(!valid) {
          sample = rnd.sample(i * segment, (i + 1) * segment);
          idx = segtree.find(sample);
          prob = segtree.get(idx);

          if( (segtree.idx - idx) % capacity > steps && (idx - segtree.idx) % capacity >= history && prob != 0)
            valid = true;
        }

        idxs[i] = idx;
        probs[i] = prob;

        ////_get_transition
        TransitionContainer transition = TransitionContainer(history + steps);
        transitions.set(&transition, history - 1, idx);

        assert(history > 2);

        for(signed int t = history - 2; t >= 0; --t) {
          if(transition.timesteps[t + 1] != 0) {
            transitions.set(&transition, t, idx - history + 1 + t);
          } else {
            blank_trans.set(&transition, t, 0);
          }
        }

        for(unsigned int t = history; t < history + steps; ++t) {
          if(transition.nonterminals[t - 1]) {
            transitions.set(&transition, t, idx - history + 1 + t);
          } else {
            blank_trans.set(&transition, t, 0);
          }
        }
        ////

        for(unsigned int t = 0; t < history; t++) {
          states[i + t] = transition.states[t];
          next_states[i + t] = transition.states[steps + t];
        }

        float sum = 0;
        for(unsigned int k = 0; k < steps; ++k)
          sum += std::pow(discount, k) * transition.rewards[history + k - 1];

        R[i] = sum;

        for(unsigned int t = 0; t < history + steps; ++t)
          actions[i + t] = transition.actions[t];

        nonterminals[i] = transition.nonterminals[history + steps - 1];
        //
      }

      at::Tensor t_probs = torch::from_blob(probs.data(), {batch_size});
      at::Tensor t_idxs = torch::from_blob(idxs.data(), {batch_size});
      at::Tensor t_states = torch::from_blob(states.data(), {batch_size * history, 84, 84});
      at::Tensor t_actions = torch::from_blob(actions.data(), {batch_size* (history + steps)});
      at::Tensor t_R = torch::from_blob(R.data(), {batch_size});      
      at::Tensor t_next_states = torch::from_blob(next_states.data(), {batch_size * history, 84, 84});
      at::Tensor t_nonterminals = torch::from_blob(nonterminals.data(), {batch_size});

      t_probs
        .div_((float)p_total)
        .mul_((float)(segtree.full() ? capacity : transitions.idx))
        .pow_(-priority_weight)
      ;

      at::Tensor weights = t_probs;
      weights
        .div_(at::max(weights))
        .to(torch::kFloat32)
        .to(device)
      ;
#ifndef CPP_ONLY
      return py::make_tuple(t_idxs, t_states, t_actions, t_R, t_next_states, t_nonterminals, weights);
#else
      return std::make_tuple(t_idxs, t_states, t_actions, t_R, t_next_states, t_nonterminals, weights);
#endif
    }

    void update_priorities(at::Tensor idxs, at::Tensor priorities) {
      assert(idxs.size(0) == priorities.size(0));

      priorities.pow_(priority_exponent);

      for(unsigned int i = 0; i < idxs.size(0); ++i) 
        segtree.update(idxs[i].item<int>(), priorities[i].item<float>());
    }

    ReplayMemory* __iter__() {
      current_idx = 0;
      return this;
    }

    at::Tensor __next__() {
      if(current_idx == capacity)
#ifndef CPP_ONLY
        throw py::stop_iteration();
#else
        throw std::exception();        
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