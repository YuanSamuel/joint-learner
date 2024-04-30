#include <torch/script.h>
#include <torch/torch.h>

#include <vector>

#ifndef MLP_REPLACEMENT_H
#define MLP_REPLACEMENT_H

class MLP_REPLACEMENT {
    torch::jit::script::Module module;

   public:
    MLP_REPLACEMENT(std::string model_path) {
        try {
            // Load the traced model
            module = torch::jit::load(model_path);
        } catch (const c10::Error& e) {
            std::cerr << "Error loading the model.\n";
        }
    }

    int forward(uint64_t curr_ip, std::vector<uint64_t>& prev_ips) {
        // std::vector<int> vec = {1, 2, 3, 4};
        std::vector<int64_t> compatible_ips(prev_ips.begin(), prev_ips.end());

        torch::Tensor tensor_single = torch::tensor(curr_ip, torch::dtype(torch::kInt64));
        torch::Tensor tensor_array = torch::tensor(compatible_ips, torch::dtype(torch::kInt64));

        torch::Tensor input_tensor = torch::cat({tensor_single.unsqueeze(0), tensor_array}, 0).toType(torch::kFloat32);

        // std::vector<torch::jit::IValue> inputs;
        // inputs.push_back(input_tensor);

        // Forward pass through the model
        at::Tensor output = module.forward({input_tensor}).toTensor();
        std::cout << "Model output: " << output << std::endl;

        return 1;
        // return output;
    }
}

#endif
