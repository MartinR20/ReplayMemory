#include <torch/extension.h>
#include "SegmentTree.cpp"
#include <random>
#include <vector>
#include <cmath>

class Rnd {
  std::random_device rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937 gen; //Standard mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<> dis;

  Rnd() : gen(rd()) {}

  int sample(int lower, int upper) {
    return dis(lower, upper)(gen);
  }
};

struct blank_trans {
  const unsigned int timestep = 0;
  const at::Tensor state      = torch::zeros({84, 84});
  const unsigned int action   = UINT_MAX;
  const float reward          = 0;
  const bool nonterminal      = false;
};

class TransitionContainer {
  unsigned int* timesteps;
  at::Tensor* states;
  unsigned int* actions;
  float* rewards;
  bool* nonterminals;
  unsigned int idx = 0;

  TransitionContainer(unsigned int capacity) :
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
    timesteps[idx] = t;
    states[idx] = state[-1].mul(255).to(torch::kInt8).to(torch::kCPU); // TODO: probably better way to do this
    actions[idx] = action;
    rewards[idx] = reward;
    nonterminals[idx] = !terminal;
    
    ++idx;
  }

  void set(TransitionContainer* other, unsigned int m, unsigned int n) {
    other->timesteps[m] = timesteps[n];
    other->states[m] = states[n];
    other->actions[m] = actions[n];
    other->rewards[m] = rewards[n];
    other->nonterminals[m] = nonterminals[n];
  }
};

class ReplayMemory {
  private:
    unsigned int capacity;
    unsigned int history;
    unsigned int t = 0;
    unsigned int steps; //n
    float discount;
    float priority_weight;
    torch::device device;
    AppendableSegmentTree<float> segtree;
    TransitionContainer transitions;

  public:
    ReplayMemory(const unsigned int capacity, 
                  const unsigned int history, 
                  const unsigned int steps,
                  float discount,
                  float priority_weight,
                  std::string device)
                : capacity(capacity),
                  history(history),
                  steps(steps),
                  discount(discount),
                  priority_weight(priority_weight),
                  device(device == "gpu" ? torch::kCUDA : torch::kCPU),
                  segtree(AppendableSegmentTree<float>(capacity)),
                  transitions(capacity) {}

    void append(unsigned int timestep,
            at::Tensor state,
            unsigned int action,
            float reward,
            bool terminal) 
    {
      segtree.append(segtree.total());
      
      transitions.append(t,
                          state[-1].mul(255).to(torch::kInt8).to(torch::kCPU), // TODO: probably better way to do this
                          action,
                          reward,
                          !terminal);
      
      t = terminal ? 0 : t+1;
    }

    py::tuple sample(unsigned int batch_size) 
    {
      unsigned int p_total = (unsigned int)segtree.total();
      unsigned int segment = p_total / batch_size;
      Rnd rnd = Rnd()

      std::vector<float> probs(batch_size);
      std::vector<unsigned int> idxs(batch_size);
      std::vector<at::Tensor> states(batch_size * history);
      std::vector<unsigned int> actions(batch_size * (history + n));
      std::vector<float> R(batch_size);
      std::vector<at::Tensor> next_states(batch_size * history);
      std::vector<bool> nonterminals(batch_size);

      for(unsigned int i = 0; i < batch_size; ++i) {
        //_get_sample_from_segment
        unsigned int sample;
        
        bool valid = false;

        while(!valid) {
          sample = rnd.sample(i * segment, (i + 1) * segment);
          idxs[i] = segtree.find(sample);
          probs[i] = segtree.get(idx);

          if( (segtree.idx - idx) % capacity > steps && (idx - segtree.idx) % capacity >= history && prob != 0)
            valid = true;
        }

        ////_get_transition
        std::vector<unsigned int> timestemp(history + n);
        std::vector<at::Tensor> state(history + n)
        std::vector<bool> nonterminal(history + n);
        unsigned int offset = i * (history + n);

        timesteps[history - 1]        = transitions.timesteps[idx];
        state[history - 1]            = transitions.states[idx];        
        actions[offset + history - 1] = transitions.actions[idx];
        nonterminal[history - 1]      = transitions.nonterminals[idx];

        for(unsigned int t = history - 2; t >= 0; --t) {
          if(timesteps[t + 1] != 0) {
            timesteps[t]        = transitions.timesteps[idx - history + 1 + t];
            state[t]            = transitions.states[idx - history + 1 + t];        
            actions[offset + t] = transitions.actions[idx - history + 1 + t];
            nonterminal[t]      = transitions.nonterminals[idx - history + 1 + t];
          } else {
            timesteps[t]        = blank_trans.timestep
            state[t]            = blank_trans.state;        
            actions[offset + t] = blank_trans.action;
            nonterminal[t]      = blank_trans.nonterminal;
          }
        }

        for(unsigned int t = history; t < history + n; ++t) {
          if(nonterminal[t - 1]) {
            timesteps[t]        = transitions.timesteps[idx - history + 1 + t];
            state[t]            = transitions.states[idx - history + 1 + t];        
            actions[offset + t] = transitions.actions[idx - history + 1 + t];
            nonterminal[t]      = transitions.nonterminals[idx - history + 1 + t];
          } else {
            timesteps[t]        = blank_trans.timestep
            state[t]            = blank_trans.state;        
            actions[offset + t] = blank_trans.action;
            nonterminal[t]      = blank_trans.nonterminal;
          }
        }
        ////
        for(unsigned int t = 0; t < history; t++) {
          states[t] = state[t];
          next_states[t] = state[n + t];
        }

        float sum = 0
        for(unsigned int k = 0; k < n; ++k)
          sum += std::pow(discount, k) * transition.rewards[history + k - 1]

        R[i] = sum;
        nonterminals[i] = nonterminal[history + n - 1];
        //
      }

      at::Tensor t_probs = torch::from_blob(probs.data(), {batch_size});
      at::Tensor t_idxs = torch::from_blob(idxs.data(), {batch_size});
      at::Tensor t_states = torch::from_blob(states.data(), {batch_size * history, 84, 84});
      at::Tensor t_actions = torch::from_blob(actions.data(), {batch_size* (history + n)});
      at::Tensor t_R = torch::from_blob(R.data(), {batch_size});      
      at::Tensor t_next_states = torch::from_blob(next_states.data(), {batch_size * history, 84, 84});
      at::Tensor t_nonterminals = torch::from_blob(nonterminals.data(), {batch_size});      

      t_probs.div_(p_total)
      unsigned int cap = transitions.full() ? capacity : transitions.idx;
      at::Tensor weights = (t_probs.dot(cap)).pow(-priority_weight);
      weights.div_(at::max(weights)).to(torch::kFloat32).to(device);

      return py::make_tuple(t_idxs, t_states, t_actions, t_R, t_next_states, t_nonterminals, weights);
    }
};