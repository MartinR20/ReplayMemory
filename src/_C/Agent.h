#define CPP_ONLY

#include "ReplayMemory.h"
#include <torch/extension.h>
#include <random>
#include <cmath>

float randf() {
    return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}

unsigned int randint(unsigned int high) {
    return rand() % high;
}

class Agent {
    private:
        const unsigned int action_space;
        const unsigned int atoms;
        const float V_min;
        const float V_max;
        const at::Tensor support;
        const float delta_z;
        const unsigned int batch_size;
        const unsigned int steps;
        const float discount;
        const torch::nn::Module *online_net;
        const torch::nn::Module *target_net;
        const torch::Device device;
        torch::optim::Adam optimiser;


    public:
        Agent(const unsigned int action_space,
                const unsigned int atoms,
                const float V_min,
                const float V_max,
                const at::Tensor support,
                const float delta_z,
                const unsigned int batch_size,
                const unsigned int multi_step,
                const float discount,
                torch::nn::Module *online_net,
                torch::nn::Module *target_net,
                const float lr,
                const unsigned int adam_eps,
                torch::Device device) :
                    action_space(action_space),
                    atoms(atoms),
                    V_min(V_min),
                    V_max(V_max),
                    support(torch::linspace(V_min, V_max, atoms, device=device)),
                    delta_z((V_max - V_min) / (atoms - 1)),
                    batch_size(batch_size),
                    steps(multi_step),
                    discount(discount),
                    online_net(online_net),
                    target_net(target_net),
                    device(device)
        { 
            online_net->train();

            update_target_net();
            target_net->train();

            std::vector<at::Tensor> params = target_net->parameters();
            for(unsigned int i = 0; i < params.size(); ++i)
                params[i].grad_ = false;

            optimiser = torch::optim::Adam(online_net->parameters(), lr, adam_eps);
        }

        void reset_noise() {
            online_net.reset_noise();
        }

        // Acts based on single state (no batch)
        const unsigned int act(at::Tensor state) {
            return (online_net(state.unsqueeze(0)) * support).sum(2).argmax(1).data<unsigned int>();
        }

        // Acts with an ε-greedy policy (used for evaluation only)
        const unsigned int act_e_greedy(at::Tensor state, const float epsilon=0.001) {  // High ε can reduce evaluation scores drastically
            if(randf() < epsilon)
              return randint(action_space);
            else 
              return act(state);
        }

        void learn(ReplayMemory& mem) {
            // Sample transitions
            //0: idxs, 1: states, 2: actions, 3: returns, 4: next_states, 5: nonterminals, 6: weights
            std::vector<at::Tensor> sample(mem.sample(batch_size));    
            at::Tensor idxs = sample[0];
            at::Tensor states = sample[1];
            at::Tensor actions = sample[2];
            at::Tensor returns = sample[3];
            at::Tensor next_states = sample[4];
            at::Tensor nonterminals = sample[5];
            at::Tensor weights = sample[6];

            // Calculate current state probabilities (online network noise already sampled)
            at::Tensor log_ps = online_net->forward(states, log=true);  // Log probabilities log p(s_t, ·; θonline)
            
            at::Tensor log_ps_a = torch::empty({batch_size, actions.size(0)});
            for(unsigned int b = 0; b < batch_size; ++b) {
                for(unsigned int a = 0; a < actions.size(0); ++a) {
                    unsigned int _a = actions[a].data<unsigned int>();
                    log_ps_a = log_ps[b][_a]; // log p(s_t, a_t; θonline)
                }
            }

            // no grad opertaions
            // Calculate nth next state probabilities
            at::Tensor pns = online_net->forward(next_states);  // Probabilities p(s_t+n, ·; θonline)
            at::Tensor dns = support.expand_as(pns) * pns;  // Distribution d_t+n = (z, p(s_t+n, ·; θonline))
            at::Tensor argmax_indices_ns = dns.sum(2).argmax(1);  // Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
            target_net->reset_noise();  // Sample new target net noise
            pns = target_net->forward(next_states);  // Probabilities p(s_t+n, ·; θtarget)

            at::Tensor pns_a = torch::empty({batch_size, argmax_indices_ns.size(0)});
            for(unsigned int b = 0; b < batch_size; ++b) {
                for(unsigned int a = 0; a < argmax_indices_ns.size(0); ++a) {
                    unsigned int _a = argmax_indices_ns[a].data<unsigned int>();
                    log_ps_a = log_ps[b][_a]; // Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)
                }
            }

            // Compute Tz (Bellman operator T applied to z)
            at::Tensor Tz = returns.unsqueeze(1) + nonterminals * std::pow(discount, steps) * support.unsqueeze(0);  // Tz = R^n + (γ^n)z (accounting for terminal states)
            Tz = Tz.clamp(min=self.Vmin, max=self.Vmax);  // Clamp between supported values
            // Compute L2 projection of Tz onto fixed support z
            at::Tensor b = (Tz - Vmin) / delta_z;  // b = (Tz - Vmin) / Δz
            at::Tensor l = b.floor().to(torch::kInt64);
            at::Tensor u = b.ceil().to(torch::kInt64);
            // Fix disappearing probability mass when l = b = u (b is int)
            l[(u > 0) * (l == u)] -= 1;
            u[(l < (atoms - 1)) * (l == u)] += 1;

            // Distribute probability of Tz
            at::Tensor m = torch::zeros({batch_size, atoms}, device);
            at::Tensor offset = torch::linspace(0, ((batch_size - 1) * atoms), batch_size).unsqueeze(1).expand(batch_size, atoms).to(actions);
            m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.to(torch::kFloat32) - b)).view(-1));  // m_l = m_l + p(s_t+n, a*)(u - b)
            m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.to(torch::kFloat32))).view(-1));  // m_u = m_u + p(s_t+n, a*)(b - l)

            at::Tensor loss = -torch::sum(m * log_ps_a, 1);  // Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
            online_net->zero_grad();
            (weights * loss).mean().backward();  // Backpropagate importance-weighted minibatch loss
            optimiser.step();

            mem.update_priorities(idxs, loss.detach().cpu());  // Update priorities of sampled transitions
        }

        void update_target_net() {
            target_net->load_state_dict(online_net.state_dict());
        }

        // Save model parameters on current device (don't move model between devices)
        void save(std::string path) {
            torch::save(online_net->state_dict(), path);
        }

        // Evaluates Q-value based on single state (no batch)
        float evaluate_q(at::Tensor state) {
            return (online_net->forward(state.unsqueeze(0)) * support).sum(2).max(1)[0].data<float>();
        }

        void train() {
            online_net->train();
        }

        void eval() {
            online_net->eval();
        }
};