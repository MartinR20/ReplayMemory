#include <torch/extension.h>

struct Transition {
  unsigned int timestep;
  at::Tensor state;
  unsigned int action;
  float reward;
  bool nonterminal;

  Transition(unsigned int timestep,
              at::Tensor state,
              unsigned int action,
              float reward,
              bool nonterminal) 
              : timestep(timestep), 
                state(state), 
                action(action),
                reward(reward),
                nonterminal(nonterminal) {}
};