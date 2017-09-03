#include <nbla_utils/nnp.hpp>

#include <iostream>
#include <string>
#include <cmath>

int main(int argc, char *argv[]) {
  nbla::CgVariablePtr y;
  float in_x;
  const float *y_data;

  // Load NNP files and prepare net
  nbla::Context ctx{"cpu", "CpuCachedArray", "0", "default"};
  nbla::utils::nnp::Nnp nnp(ctx);
  nnp.add("exp_net.nnp");
  auto executor = nnp.get_executor("runtime");
  executor->set_batch_size(1);
  nbla::CgVariablePtr x = executor->get_data_variables().at(0).variable;
  float *data = x->variable()->cast_data_and_get_pointer<float>(ctx);

  for(int i=1;i<10;i++){
    // set input data
    in_x = 0.1*i;
    *data = in_x;

    // execute
    executor->execute();
    y = executor->get_output_variables().at(0).variable;
    y_data= y->variable()->get_data_pointer<float>(ctx);

    // print output
    std::cout << "exp(" << in_x <<"):" << "predict: " << y_data[0] << ", actual: " << std::exp(in_x) <<std::endl;
  }

  return 0;
}
