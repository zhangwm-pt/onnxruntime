// Copyright (C) 2016-2023 T-Head Semiconductor Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#include "onnx_converter.h"
#include "shl_common.h"

using std::string;
using std::vector;

namespace onnxruntime {
namespace shl_ep {

#define HAS(map, key) \
  (map.find(key) != map.end())

void OnnxToShlConverter::FuseDQDTensor(const GraphViewer& graph_viewer) {
  std::set<const Node*> processed_q_nodes;
  for (auto& node_idx : graph_viewer.GetNodesInTopologicalOrder()) {
    const auto& node = graph_viewer.GetNode(node_idx);
    if (node->OpType() == "DequantizeLinear") {
      bool is_const_dq = graph_viewer.IsInitializedTensor(node->InputDefs()[0]->Name());
      ;
      if (is_const_dq) {
        ProcessDQNodes(node);
      }
      if (!is_const_dq) {
        auto dq_in_node_iter = node->InputNodesBegin();
        auto dq_in_node_index = dq_in_node_iter->Index();
        auto dq_in_node = graph_viewer.GetNode(dq_in_node_index);
        if (dq_in_node->OpType() != "QuantizeLinear" || processed_q_nodes.count(dq_in_node) == 0) {
          throw std::invalid_argument("Get q node error.");
        }
        ProcessDQDNodes(dq_in_node, node);
      }
    } else if (node->OpType() == "QuantizeLinear") {
      ProcessQNodes(node);
      processed_q_nodes.insert(node);
    }
  }
}

void OnnxToShlConverter::InitAllTensor(const GraphViewer& graph_viewer) {
  const auto& init_tensors = graph_viewer.GetAllInitializedTensors();
  const std::vector<const NodeArg*>& all_nodes = graph_viewer.GetInputsIncludingInitializers();

  // input and constant
  for (const auto* node : all_nodes) {
    csinn_tensor* shl_tensor = shl_ep::CreateShlTensor(node, session_);
    shl_tensor_map[node->Name()] = shl_tensor;
  }

  // per layer output
  for (const auto& layer : graph_viewer.Nodes()) {
    for (const auto output : layer.OutputDefs()) {
      csinn_tensor* shl_tensor = shl_ep::CreateShlTensor(output, session_);
      shl_tensor_map[output->Name()] = shl_tensor;
    }
  }

  const auto& input_tensors = graph_viewer.GetInputs();
  const auto& output_tensors = graph_viewer.GetOutputs();

  // set constant buf
  for (const auto& tensor : init_tensors) {
    Initializer unpacked_tensor(*(tensor.second));
    const uint8_t* data_buf = unpacked_tensor.DataAsByteSpan().data();
    csinn_tensor* shl_tensor = shl_tensor_map.at(tensor.first);
    int size = csinn_tensor_byte_size(shl_tensor);
    shl_tensor->data = shl_mem_alloc(size);
    shl_tensor->layout = GetShlWeightLayoutEnum(shl_tensor->dim_count);
    memcpy(shl_tensor->data, data_buf, size);
    shl_tensor->is_const = 1;
  }

  csinn_set_input_number(input_tensors.size(), session_);
  csinn_set_output_number(output_tensors.size(), session_);

  auto set_sess_dynamic_shape = [session = session_](csinn_tensor* t) {
    for (int i = 0; i < t->dim_count; i++) {
      if (t->dim[i] < 0) {
        if (t->dim[i] == -1) {
          session->dynamic_shape = true;
          break;
        } else {
          throw std::invalid_argument("Error obtaining shape value.");
        }
      }
    }
  };

  // fuse dqd node
  FuseDQDTensor(graph_viewer);

  for (uint i = 0; i < input_tensors.size(); i++) {
    auto tensor = input_tensors[i];
    auto shl_tensor = shl_tensor_map.at(tensor->Name());
    set_sess_dynamic_shape(shl_tensor);
    csinn_set_tensor_entry(shl_tensor, session_);
    csinn_set_input(i, shl_tensor, session_);
  }

  if (session_->dynamic_shape && session_->base_api == CSINN_TH1520) {
    throw std::invalid_argument("When use th1520 backend, input shape must be a fix value. model can be converted to fixed by onnxsim tool");
  }
}

void OnnxToShlConverter::InitSHLSession(const GraphViewer& graph_viewer) {
  session_->base_dtype = CSINN_DTYPE_FLOAT32;
  session_->base_quant_type = CSINN_QUANT_FLOAT32;
  csinn_dtype_enum quant_dtype = CSINN_DTYPE_SIZE;

  for (const auto& node : graph_viewer.Nodes()) {
    const auto& op_type = node.OpType();
    if (op_type == "QuantizeLinear") {
      auto q_out_node = node.OutputDefs()[0];
      auto c_quant_dtype = shl_ep::GetShlDtypeEnum(q_out_node->TypeAsProto()->tensor_type());
      if (quant_dtype == CSINN_DTYPE_SIZE) {
        quant_dtype = c_quant_dtype;
      } else {
        if (quant_dtype != c_quant_dtype) {
          throw std::invalid_argument("differ quant op dtype must same.");
        }
      }
    }
  }
  session_->base_layout = CSINN_LAYOUT_NCHW;
  session_->model.save_mode = CSINN_RUN_ONLY;
  session_->base_dtype = quant_dtype;
  if (quant_dtype == CSINN_DTYPE_UINT8) {
    session_->base_quant_type = CSINN_QUANT_UINT8_ASYM;
  } else if (quant_dtype == CSINN_DTYPE_INT8) {
    session_->base_quant_type = CSINN_QUANT_INT8_ASYM;
  }

  marked_fusible_map = shl_ep::MarkfusibleNodes(graph_viewer);
  all_fusible_nodes = shl_ep::GetAllFusionNode(marked_fusible_map);

  if (!marked_fusible_map.count("QDQFusion") && session_->base_api == CSINN_TH1520) {
    LOGS_DEFAULT(WARNING) << "Shl: The current model is not a uint8 quantized model. CPU will be used instead of NPU for execution.";
    session_->base_api = CSINN_C920;
    session_->base_run_mode = CSINN_RM_CPU_GRAPH;
  }

  csinn_session_init(session_);
}

void OnnxToShlConverter::Convert(const GraphViewer& graph_viewer) {
  // 1.  create all shl tensor and set constant data
  InitSHLSession(graph_viewer);
  InitAllTensor(graph_viewer);

  // 2. set attr and build shl graph
  bool clear_buffer = true;
  for (const auto& node : graph_viewer.Nodes()) {
    NodeConvert(node, clear_buffer);
  }

  // 3. set output tensor
  const auto& output_tensors = graph_viewer.GetOutputs();
  for (uint i = 0; i < output_tensors.size(); i++) {
    auto tensor_name = output_tensors[i];
    auto shl_tensor = shl_tensor_map[tensor_name->Name()];
    csinn_set_output(i, shl_tensor, session_);
  }
  csinn_session_setup(session_);
}
void OnnxToShlConverter::Conv2D(const onnxruntime::Node& node) {
  NodeAttrHelper helper(node);
  auto params = shl_ep::GetShlParams<csinn_conv2d_params>(session_, node);
  const auto strides = helper.Get("strides", std::vector<int32_t>{1, 1});
  const auto pads = helper.Get("pads", std::vector<int32_t>{0, 0, 0, 0});
  const auto dilations = helper.Get("dilations", std::vector<int32_t>{1, 1});

  params->base.name = const_cast<char*>(node.Name().c_str());
  params->group = helper.Get("group", 1);
  params->stride_height = strides[0];
  params->stride_width = strides[1];
  params->pad_top = pads[0];
  params->pad_left = pads[1];
  params->pad_down = pads[2];
  params->pad_right = pads[3];
  params->dilation_height = dilations[0];
  params->dilation_width = dilations[1];

  std::string bias;
  if (node.InputDefs().size() >= 3) {
    bias = node.InputDefs()[2]->Name();
  }

  std::string input = node.InputDefs()[0]->Name();
  std::string weight = node.InputDefs()[1]->Name();
  auto input_tensor = shl_tensor_map.at(input);
  auto weight_tensor = shl_tensor_map.at(weight);
  auto bias_tensor = bias.size() ? shl_tensor_map.at(bias) : csinn_alloc_tensor(session_);
  std::string output = node.OutputDefs()[0]->Name();
  auto output_tensor = shl_tensor_map.at(output);

  // checek dtype
  CheckQuantDtype<csinn_tensor>(input_tensor, weight_tensor, output_tensor);

  csinn_conv2d_init(input_tensor, output_tensor, weight_tensor, bias_tensor, params);
  csinn_conv2d(input_tensor, output_tensor, weight_tensor, bias_tensor, params);
}

void OnnxToShlConverter::Conv(const onnxruntime::Node& node) {
  auto weight = node.InputDefs()[1];
  switch (shl_tensor_map[weight->Name()]->dim_count) {
    case 3:
      Conv1D(node);
      break;
    case 4:
      Conv2D(node);
      break;
    case 5:
      Conv3D(node);
      break;
    default:
      throw std::invalid_argument("Unsupported dims of Conv.");
      ;
  }
}

void OnnxToShlConverter::ReLU(const onnxruntime::Node& node) {
  auto params = shl_ep::GetShlParams<csinn_relu_params>(session_, node);
  params->base.name = const_cast<char*>(node.Name().c_str());
  std::string input = node.InputDefs()[0]->Name();
  auto input_tensor = shl_tensor_map.at(input);
  std::string output = node.OutputDefs()[0]->Name();
  auto output_tensor = shl_tensor_map.at(output);

  CheckQuantDtype<csinn_tensor>(input_tensor, output_tensor);
  csinn_relu_init(input_tensor, output_tensor, params);
  csinn_relu(input_tensor, output_tensor, params);
}

void OnnxToShlConverter::LeakyRelu(const onnxruntime::Node& node) {
  NodeAttrHelper helper(node);
  const auto alpha = helper.Get("alpha", 0.01f);
  auto params = shl_ep::GetShlParams<csinn_relu_params>(session_, node);
  params->base.name = const_cast<char*>(node.Name().c_str());
  params->n = alpha;

  std::string input = node.InputDefs()[0]->Name();
  auto input_tensor = shl_tensor_map.at(input);
  std::string output = node.OutputDefs()[0]->Name();
  auto output_tensor = shl_tensor_map.at(output);

  CheckQuantDtype<csinn_tensor>(input_tensor, output_tensor);
  csinn_leaky_relu_init(input_tensor, output_tensor, params);
  csinn_leaky_relu(input_tensor, output_tensor, params);
}

void OnnxToShlConverter::SetPoolAttr(csinn_pool_params* params, const onnxruntime::Node& node) {
  NodeAttrHelper helper(node);
  const auto strides = helper.Get("strides", std::vector<int32_t>{1, 1});
  const auto pads = helper.Get("pads", std::vector<int32_t>{0, 0, 0, 0});
  const auto kernel_shape = helper.Get("kernel_shape", std::vector<int32_t>{1, 1});
  const auto count_include_pad = helper.Get("count_include_pad", 0);

  params->base.name = const_cast<char*>(node.Name().c_str());
  params->stride_height = strides[0];
  params->stride_width = strides[1];
  params->pad_top = pads[0];
  params->pad_left = pads[1];
  params->pad_down = pads[2];
  params->pad_right = pads[3];
  params->filter_height = kernel_shape[0];
  params->filter_width = kernel_shape[1];
  params->count_include_pad = count_include_pad;
  params->ceil_mode = helper.Get("ceil_mode", 0);
}

void OnnxToShlConverter::AvgPool2d(const onnxruntime::Node& node) {
  auto params = shl_ep::GetShlParams<csinn_pool_params>(session_, node);
  SetPoolAttr(params, node);

  std::string input = node.InputDefs()[0]->Name();
  auto input_tensor = shl_tensor_map.at(input);
  std::string output = node.OutputDefs()[0]->Name();
  auto output_tensor = shl_tensor_map.at(output);
  CheckQuantDtype<csinn_tensor>(input_tensor, output_tensor);
  csinn_avgpool2d_init(input_tensor, output_tensor, params);
  csinn_avgpool2d(input_tensor, output_tensor, params);
}

void OnnxToShlConverter::MaxPool2d(const onnxruntime::Node& node) {
  auto params = shl_ep::GetShlParams<csinn_pool_params>(session_, node);
  SetPoolAttr(params, node);

  std::string input = node.InputDefs()[0]->Name();
  auto input_tensor = shl_tensor_map.at(input);
  std::string output = node.OutputDefs()[0]->Name();
  auto output_tensor = shl_tensor_map.at(output);
  CheckQuantDtype<csinn_tensor>(input_tensor, output_tensor);
  csinn_maxpool2d_init(input_tensor, output_tensor, params);
  csinn_maxpool2d(input_tensor, output_tensor, params);
}

void OnnxToShlConverter::GlobalAvgPool(const onnxruntime::Node& node) {
  auto params = shl_ep::GetShlParams<csinn_pool_params>(session_, node);
  SetPoolAttr(params, node);

  std::string input = node.InputDefs()[0]->Name();
  auto input_tensor = shl_tensor_map.at(input);
  std::string output = node.OutputDefs()[0]->Name();
  auto output_tensor = shl_tensor_map.at(output);
  CheckQuantDtype<csinn_tensor>(input_tensor, output_tensor);
  csinn_global_avgpool2d_init(input_tensor, output_tensor, params);
  csinn_global_avgpool2d(input_tensor, output_tensor, params);
}

void OnnxToShlConverter::AveragePool(const onnxruntime::Node& node) {
  auto input = node.InputDefs()[0];
  switch (shl_tensor_map[input->Name()]->dim_count) {
    case 4:
      AvgPool2d(node);
      break;
    default:
      throw std::invalid_argument("Unsupported dims of AvgPool.");
  }
}

void OnnxToShlConverter::MaxPool(const onnxruntime::Node& node) {
  auto input = node.InputDefs()[0];
  switch (shl_tensor_map[input->Name()]->dim_count) {
    case 4:
      MaxPool2d(node);
      break;
    default:
      throw std::invalid_argument("Unsupported dims of AvgPool.");
  }
}

void OnnxToShlConverter::Softmax(const onnxruntime::Node& node) {
  NodeAttrHelper helper(node);
  auto params = shl_ep::GetShlParams<csinn_softmax_params>(session_, node);

  params->base.name = const_cast<char*>(node.Name().c_str());
  params->axis = helper.Get("axis", -1);

  std::string input = node.InputDefs()[0]->Name();
  auto input_tensor = shl_tensor_map.at(input);
  std::string output = node.OutputDefs()[0]->Name();
  auto output_tensor = shl_tensor_map.at(output);
  params->axis = params->axis < 0 ? params->axis + input_tensor->dim_count : params->axis;
  CheckQuantDtype<csinn_tensor>(input_tensor, output_tensor);
  csinn_softmax_init(input_tensor, output_tensor, params);
  csinn_softmax(input_tensor, output_tensor, params);
}

void OnnxToShlConverter::Erf(const onnxruntime::Node& node) {
  NodeAttrHelper helper(node);
  auto params = shl_ep::GetShlParams<csinn_siso_params>(session_, node);
  params->base.name = const_cast<char*>(node.Name().c_str());

  std::string input = node.InputDefs()[0]->Name();
  auto input_tensor = shl_tensor_map.at(input);
  std::string output = node.OutputDefs()[0]->Name();
  auto output_tensor = shl_tensor_map.at(output);
  CheckQuantDtype<csinn_tensor>(input_tensor, output_tensor);
  csinn_erf_init(input_tensor, output_tensor, params);
  csinn_erf(input_tensor, output_tensor, params);
}

void OnnxToShlConverter::Gemm(const onnxruntime::Node& node) {
  NodeAttrHelper helper(node);
  auto params = shl_ep::GetShlParams<csinn_fc_params>(session_, node);
  params->base.name = const_cast<char*>(node.Name().c_str());

  std::string bias;
  if (node.InputDefs().size() >= 3) {
    bias = node.InputDefs()[2]->Name();
  }

  std::string input = node.InputDefs()[0]->Name();
  std::string weight = node.InputDefs()[1]->Name();
  auto input_tensor = shl_tensor_map.at(input);
  auto weight_tensor = shl_tensor_map.at(weight);
  // auto bias_tensor = bias.size() ? shl_tensor_map.at(bias) : csinn_alloc_tensor(session_);

  csinn_tensor* bias_tensor;
  if (bias.size()) {
    bias_tensor = shl_tensor_map.at(bias);
  } else if (session_->base_api == CSINN_TH1520) {
    bias_tensor = csinn_alloc_tensor(session_);
    int32_t size = weight_tensor->dim[0];
    bias_tensor->data = shl_mem_calloc(4, size);
    std::string bias_name = input + "_bias";
    bias_tensor->name = const_cast<char*>(bias_name.c_str());
    bias_tensor->dim_count = 1;
    bias_tensor->dim[0] = size;
    if (input_tensor->dtype == CSINN_DTYPE_FLOAT32) {
      bias_tensor->dtype = CSINN_DTYPE_FLOAT32;
    } else if (input_tensor->dtype == CSINN_DTYPE_FLOAT16) {
      bias_tensor->dtype = CSINN_DTYPE_FLOAT16;
    } else {
      bias_tensor->dtype = CSINN_DTYPE_INT32;
    }
    csinn_quant_info* b_q_infos = new csinn_quant_info[input_tensor->quant_channel];
    for (int i = 0; i < input_tensor->quant_channel; i++) {
      b_q_infos[i].scale = input_tensor->qinfo->scale * weight_tensor->qinfo->scale;
      b_q_infos[i].zero_point = 0;
    }
    bias_tensor->qinfo = b_q_infos;
    bias_tensor->layout = CSINN_LAYOUT_O;
    bias_tensor->is_const = 1;
  } else {
    bias_tensor = csinn_alloc_tensor(session_);
  }

  std::string output = node.OutputDefs()[0]->Name();
  auto output_tensor = shl_tensor_map.at(output);

  params->units = weight_tensor->dim[0];

  CheckQuantDtype<csinn_tensor>(input_tensor, output_tensor, weight_tensor);
  csinn_fullyconnected_init(input_tensor, output_tensor, weight_tensor, bias_tensor, params);
  csinn_fullyconnected(input_tensor, output_tensor, weight_tensor, bias_tensor, params);
}

void OnnxToShlConverter::MatMul(const onnxruntime::Node& node) {
  NodeAttrHelper helper(node);
  auto params = shl_ep::GetShlParams<csinn_matmul_params>(session_, node);
  params->base.name = const_cast<char*>(node.Name().c_str());

  std::string input_a = node.InputDefs()[0]->Name();
  std::string input_b = node.InputDefs()[1]->Name();
  auto a_tensor = shl_tensor_map.at(input_a);
  auto b_tensor = shl_tensor_map.at(input_b);
  std::string output = node.OutputDefs()[0]->Name();
  auto output_tensor = shl_tensor_map.at(output);
  params->trans_a = false;
  params->trans_b = false;

  CheckQuantDtype<csinn_tensor>(a_tensor, b_tensor, output_tensor);
  csinn_matmul_init(a_tensor, b_tensor, output_tensor, params);
  csinn_matmul(a_tensor, b_tensor, output_tensor, params);
}

void OnnxToShlConverter::Add(const onnxruntime::Node& node) {
  NodeAttrHelper helper(node);
  auto params = shl_ep::GetShlParams<csinn_diso_params>(session_, node);

  params->base.name = const_cast<char*>(node.Name().c_str());

  std::string l_input = node.InputDefs()[0]->Name();
  auto l_input_tensor = shl_tensor_map.at(l_input);
  std::string r_input = node.InputDefs()[1]->Name();
  auto r_input_tensor = shl_tensor_map.at(r_input);
  std::string output = node.OutputDefs()[0]->Name();
  auto output_tensor = shl_tensor_map.at(output);
  CheckQuantDtype<csinn_tensor>(l_input_tensor, r_input_tensor, output_tensor);
  csinn_add_init(l_input_tensor, r_input_tensor, output_tensor, params);
  csinn_add(l_input_tensor, r_input_tensor, output_tensor, params);
}

void OnnxToShlConverter::Sub(const onnxruntime::Node& node) {
  NodeAttrHelper helper(node);
  auto params = shl_ep::GetShlParams<csinn_diso_params>(session_, node);

  params->base.name = const_cast<char*>(node.Name().c_str());

  std::string l_input = node.InputDefs()[0]->Name();
  auto l_input_tensor = shl_tensor_map.at(l_input);
  std::string r_input = node.InputDefs()[1]->Name();
  auto r_input_tensor = shl_tensor_map.at(r_input);
  std::string output = node.OutputDefs()[0]->Name();
  auto output_tensor = shl_tensor_map.at(output);
  CheckQuantDtype<csinn_tensor>(l_input_tensor, r_input_tensor, output_tensor);
  csinn_sub_init(l_input_tensor, r_input_tensor, output_tensor, params);
  csinn_sub(l_input_tensor, r_input_tensor, output_tensor, params);
}

void OnnxToShlConverter::Mul(const onnxruntime::Node& node) {
  NodeAttrHelper helper(node);
  auto params = shl_ep::GetShlParams<csinn_diso_params>(session_, node);

  params->base.name = const_cast<char*>(node.Name().c_str());

  std::string l_input = node.InputDefs()[0]->Name();
  auto l_input_tensor = shl_tensor_map.at(l_input);
  std::string r_input = node.InputDefs()[1]->Name();
  auto r_input_tensor = shl_tensor_map.at(r_input);
  std::string output = node.OutputDefs()[0]->Name();
  auto output_tensor = shl_tensor_map.at(output);
  CheckQuantDtype<csinn_tensor>(l_input_tensor, r_input_tensor, output_tensor);
  csinn_mul_init(l_input_tensor, r_input_tensor, output_tensor, params);
  csinn_mul(l_input_tensor, r_input_tensor, output_tensor, params);
}

void OnnxToShlConverter::Div(const onnxruntime::Node& node) {
  NodeAttrHelper helper(node);
  auto params = shl_ep::GetShlParams<csinn_diso_params>(session_, node);

  params->base.name = const_cast<char*>(node.Name().c_str());

  std::string l_input = node.InputDefs()[0]->Name();
  auto l_input_tensor = shl_tensor_map.at(l_input);
  std::string r_input = node.InputDefs()[1]->Name();
  auto r_input_tensor = shl_tensor_map.at(r_input);
  std::string output = node.OutputDefs()[0]->Name();
  auto output_tensor = shl_tensor_map.at(output);
  CheckQuantDtype<csinn_tensor>(l_input_tensor, r_input_tensor, output_tensor);
  csinn_div_init(l_input_tensor, r_input_tensor, output_tensor, params);
  csinn_div(l_input_tensor, r_input_tensor, output_tensor, params);
}

void OnnxToShlConverter::Flatten(const onnxruntime::Node& node) {
  NodeAttrHelper helper(node);
  auto params = shl_ep::GetShlParams<csinn_flatten_params>(session_, node);

  params->base.name = const_cast<char*>(node.Name().c_str());
  params->axis = helper.Get("axis", 1);

  std::string input = node.InputDefs()[0]->Name();
  auto input_tensor = shl_tensor_map.at(input);
  if (params->axis < 0) {
    params->axis += input_tensor->dim_count;
  }
  std::string output = node.OutputDefs()[0]->Name();
  auto output_tensor = shl_tensor_map.at(output);
  CheckQuantDtype<csinn_tensor>(input_tensor, output_tensor);
  csinn_flatten_init(input_tensor, output_tensor, params);
  csinn_flatten(input_tensor, output_tensor, params);
}

void OnnxToShlConverter::Gather(const onnxruntime::Node& node) {
  NodeAttrHelper helper(node);
  auto params = shl_ep::GetShlParams<csinn_gather_params>(session_, node);

  params->base.name = const_cast<char*>(node.Name().c_str());
  params->axis = helper.Get("axis", 0);

  std::string input = node.InputDefs()[0]->Name();
  auto input_tensor = shl_tensor_map.at(input);
  std::string indices = node.InputDefs()[1]->Name();
  auto indices_tensor = shl_tensor_map.at(indices);
  if (params->axis < 0) {
    params->axis += input_tensor->dim_count;
  }
  std::string output = node.OutputDefs()[0]->Name();
  auto output_tensor = shl_tensor_map.at(output);
  CheckQuantDtype<csinn_tensor>(input_tensor, indices_tensor, output_tensor);
  csinn_gather_init(input_tensor, indices_tensor, output_tensor, params);
  csinn_gather(input_tensor, indices_tensor, output_tensor, params);
}

void OnnxToShlConverter::Sigmoid(const onnxruntime::Node& node) {
  NodeAttrHelper helper(node);
  auto params = shl_ep::GetShlParams<csinn_sigmoid_params>(session_, node);

  params->base.name = const_cast<char*>(node.Name().c_str());

  std::string input = node.InputDefs()[0]->Name();
  auto input_tensor = shl_tensor_map.at(input);
  std::string output = node.OutputDefs()[0]->Name();
  auto output_tensor = shl_tensor_map.at(output);
  CheckQuantDtype<csinn_tensor>(input_tensor, output_tensor);
  csinn_sigmoid_init(input_tensor, output_tensor, params);
  csinn_sigmoid(input_tensor, output_tensor, params);
}

void OnnxToShlConverter::Concat(const onnxruntime::Node& node) {
  NodeAttrHelper helper(node);
  auto params = shl_ep::GetShlParams<csinn_concat_params>(session_, node);

  params->base.name = const_cast<char*>(node.Name().c_str());
  params->axis = helper.Get("axis", -1);
  params->inputs_count = node.InputDefs().size();

  csinn_tensor* inputs[params->inputs_count];

  for (int32_t i = 0; i < params->inputs_count; i++) {
    std::string input = node.InputDefs()[i]->Name();
    inputs[i] = shl_tensor_map.at(input);
  }

  if (params->axis < 0) {
    params->axis += inputs[0]->dim_count;
  }

  std::string output = node.OutputDefs()[0]->Name();
  auto output_tensor = shl_tensor_map.at(output);
  CheckQuantDtype(output_tensor, inputs, params->inputs_count);
  csinn_concat_init(inputs, output_tensor, params);
  csinn_concat(inputs, output_tensor, params);
}

void OnnxToShlConverter::Resize(const onnxruntime::Node& node) {
  NodeAttrHelper helper(node);
  auto params = shl_ep::GetShlParams<csinn_resize_params>(session_, node);

  params->base.name = const_cast<char*>(node.Name().c_str());
  const auto coordinate_transformation_mode = helper.Get("coordinate_transformation_mode", "half_pixel");
  params->align_corners = coordinate_transformation_mode == "align_corners";
  const auto mode = helper.Get("mode", "nearest");

  if (mode == "bilinear") {
    params->resize_mode = CSINN_RESIZE_BILINEAR;
  } else if (mode == "linear") {
    params->resize_mode = CSINN_RESIZE_BILINEAR;
  } else if (mode == "nearest") {
    params->resize_mode = CSINN_RESIZE_NEAREST_NEIGHBOR;
  } else {
    throw std::invalid_argument("Unsupported dims of AvgPool.");
  }

  std::string input = node.InputDefs()[0]->Name();
  auto input_tensor = shl_tensor_map.at(input);
  std::string output = node.OutputDefs()[0]->Name();
  auto output_tensor = shl_tensor_map.at(output);
  CheckQuantDtype<csinn_tensor>(input_tensor, output_tensor);
  csinn_resize_init(input_tensor, output_tensor, params);
  csinn_resize(input_tensor, output_tensor, params);
}

void OnnxToShlConverter::Reshape(const onnxruntime::Node& node) {
  NodeAttrHelper helper(node);
  auto params = shl_ep::GetShlParams<csinn_reshape_params>(session_, node);

  params->base.name = const_cast<char*>(node.Name().c_str());

  std::string input = node.InputDefs()[0]->Name();
  auto input_tensor = shl_tensor_map.at(input);
  std::string output = node.OutputDefs()[0]->Name();
  auto output_tensor = shl_tensor_map.at(output);

  params->shape = reinterpret_cast<int32_t*>(malloc(sizeof(int32_t) * 8));

  for (int i = 0; i < output_tensor->dim_count; i++) {
    params->shape[i] = output_tensor->dim[i];
  }
  params->shape_num = output_tensor->dim_count;
  CheckQuantDtype<csinn_tensor>(input_tensor, output_tensor);
  csinn_reshape_init(input_tensor, output_tensor, params);
  csinn_reshape(input_tensor, output_tensor, params);
}

void OnnxToShlConverter::Transpose(const onnxruntime::Node& node) {
  NodeAttrHelper helper(node);
  auto params = shl_ep::GetShlParams<csinn_transpose_params>(session_, node);
  const auto perm = helper.Get("perm", vector<int>{0, 1, 2});

  params->base.name = const_cast<char*>(node.Name().c_str());
  params->permute_num = perm.size();
  params->permute = reinterpret_cast<int32_t*>(malloc(sizeof(int32_t) * perm.size()));

  for (uint i = 0; i < perm.size(); i++) {
    params->permute[i] = perm[i];
  }

  std::string input = node.InputDefs()[0]->Name();
  auto input_tensor = shl_tensor_map.at(input);
  std::string output = node.OutputDefs()[0]->Name();
  auto output_tensor = shl_tensor_map.at(output);

  CheckQuantDtype<csinn_tensor>(input_tensor, output_tensor);
  csinn_transpose_init(input_tensor, output_tensor, params);
  csinn_transpose(input_tensor, output_tensor, params);
}

void OnnxToShlConverter::Split(const onnxruntime::Node& node) {
  NodeAttrHelper helper(node);
  auto params = shl_ep::GetShlParams<csinn_split_params>(session_, node);
  params->base.name = const_cast<char*>(node.Name().c_str());
  params->output_num = node.OutputDefs().size();
  if (node.InputDefs().size() == 2) {
    std::string split = node.InputDefs()[1]->Name();
    csinn_tensor* split_tensor = shl_tensor_map.at(split);
    int split_length = split_tensor->dim[0] - 1;
    params->split_index = reinterpret_cast<int32_t*>(malloc(split_length * sizeof(int32_t)));
    int64_t* index = reinterpret_cast<int64_t*>(split_tensor->data);
    for (int i = 0; i < split_length; i++) {
      if (i == 0) {
        params->split_index[i] = index[i];
      } else {
        params->split_index[i] = params->split_index[i - 1] + index[i];
      }
    }
  }

  std::string input = node.InputDefs()[0]->Name();
  auto input_tensor = shl_tensor_map.at(input);
  csinn_tensor* output_tensors[params->output_num];

  for (int i = 0; i < params->output_num; i++) {
    std::string output = node.OutputDefs()[i]->Name();
    output_tensors[i] = shl_tensor_map.at(output);
  }
  auto axis = helper.Get("axis", 0);
  params->axis = axis < 0 ? axis + input_tensor->dim_count : axis;

  CheckQuantDtype(input_tensor, output_tensors, params->output_num);
  csinn_split_init(input_tensor, output_tensors, params);
  csinn_split(input_tensor, output_tensors, params);
}

void OnnxToShlConverter::Pow(const onnxruntime::Node& node) {
  NodeAttrHelper helper(node);
  auto params = shl_ep::GetShlParams<csinn_diso_params>(session_, node);
  params->base.name = const_cast<char*>(node.Name().c_str());

  std::string l_input = node.InputDefs()[0]->Name();
  auto l_input_tensor = shl_tensor_map.at(l_input);
  std::string r_input = node.InputDefs()[1]->Name();
  auto r_input_tensor = shl_tensor_map.at(r_input);
  std::string output = node.OutputDefs()[0]->Name();
  auto output_tensor = shl_tensor_map.at(output);

  CheckQuantDtype<csinn_tensor>(l_input_tensor, r_input_tensor, output_tensor);
  csinn_power_init(l_input_tensor, r_input_tensor, output_tensor, params);
  csinn_power(l_input_tensor, r_input_tensor, output_tensor, params);
}

void OnnxToShlConverter::Clip(const onnxruntime::Node& node) {
  NodeAttrHelper helper(node);
  auto params = shl_ep::GetShlParams<csinn_clip_params>(session_, node);
  params->base.name = const_cast<char*>(node.Name().c_str());
  auto max_value = helper.Get("max", std::numeric_limits<float>::max());
  auto min_value = helper.Get("min", std::numeric_limits<float>::min());
  params->max_value = max_value;
  params->min_value = min_value;
  
  std::string input = node.InputDefs()[0]->Name();
  auto input_tensor = shl_tensor_map.at(input);
  auto in_dtype = node.InputDefs()[0]->TypeAsProto()->tensor_type();
  int32_t in_dtype_elem;
  if (in_dtype.has_elem_type()) {
    in_dtype_elem = in_dtype.elem_type();
  } else {
    throw std::invalid_argument("Shl clip get error dtype.");
  }

  if (in_dtype_elem == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    if (node.InputDefs().size() >= 2) {
      std::string min_name = node.InputDefs()[1]->Name();
      auto min_tensor = shl_tensor_map.at(min_name);
      float* min_value = reinterpret_cast<float*>(min_tensor->data);
      params->min_value = min_value[0];
    }
    if (node.InputDefs().size() == 3) {
      std::string max_name = node.InputDefs()[2]->Name();
      auto max_tensor = shl_tensor_map.at(max_name);
      float* max_value = reinterpret_cast<float*>(max_tensor->data);
      params->max_value = max_value[0];
    }
  } else if (in_dtype_elem == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
    if (node.InputDefs().size() >= 2) {
      std::string min_name = node.InputDefs()[1]->Name();
      auto min_tensor = shl_tensor_map.at(min_name);
      int16_t* min_value = reinterpret_cast<int16_t*>(min_tensor->data);
      params->min_value = float16_to_float32(min_value[0]);
    }
    if (node.InputDefs().size() == 3) {
      std::string max_name = node.InputDefs()[2]->Name();
      auto max_tensor = shl_tensor_map.at(max_name);
      int16_t* max_value = reinterpret_cast<int16_t*>(max_tensor->data);
      params->max_value = float16_to_float32(max_value[0]);
    }
  }

  std::string output = node.OutputDefs()[0]->Name();
  auto output_tensor = shl_tensor_map.at(output);
  CheckQuantDtype<csinn_tensor>(input_tensor, output_tensor);

  if (params->min_value == 0) {
    if (params->max_value == std::numeric_limits<float>::max()) {
      auto relu_params = shl_ep::GetShlParams<csinn_relu_params>(session_, node);
      relu_params->base.name = params->base.name;
      csinn_relu_init(input_tensor, output_tensor, relu_params);
      csinn_relu(input_tensor, output_tensor, relu_params);
      return;
    }
    if (params->max_value == 6) {
      auto relu_params = shl_ep::GetShlParams<csinn_relu_params>(session_, node);
      relu_params->base.name = params->base.name;
      csinn_relu6_init(input_tensor, output_tensor, relu_params);
      csinn_relu6(input_tensor, output_tensor, relu_params);
      return;
    }
  }

  csinn_clip_init(input_tensor, output_tensor, params);
  csinn_clip(input_tensor, output_tensor, params);
}

void OnnxToShlConverter::FuseLayerNormalization(std::unordered_map<std::string, const Node*> nodes_map) {
  auto mul_node = nodes_map["mul"];
  auto add1_node = nodes_map["add1"];
  auto add2_node = nodes_map["add2"];
  auto reduce_mean_node = nodes_map["key_node"];
  auto first_node = nodes_map["first_node"];

  // get axis
  const onnxruntime::NodeAttributes& attributes = reduce_mean_node->GetAttributes();
  std::vector<int64_t> axes_values;
  if (attributes.find("axes") != attributes.end()) {
    axes_values = onnx::RetrieveValues<int64_t>(attributes.at("axes"));
  } else if (reduce_mean_node->InputDefs().size() == 2) {
    auto axes_name = reduce_mean_node->InputDefs()[1]->Name();
    auto axes_tensor = shl_tensor_map.at(axes_name);
    int64_t* axes = reinterpret_cast<int64_t*>(axes_tensor->data);
    if (axes_tensor->dim_count == 1) {
      axes_values.insert(axes_values.end(), axes, axes + axes_tensor->dim[0] * sizeof(int64_t));
    } else {
      axes_values.push_back(-1);
    }
  }

  // get eps
  float eps;
  auto add1_input0 = add1_node->InputDefs()[0]->Name();
  auto add1_input1 = add1_node->InputDefs()[1]->Name();
  auto add1_input0_tensor = shl_tensor_map.at(add1_input0);
  auto add1_input1_tensor = shl_tensor_map.at(add1_input1);
  if (add1_input1_tensor->is_const) {
    eps = reinterpret_cast<float*>(add1_input1_tensor->data)[0];
  } else if (add1_input0_tensor->is_const) {
    eps = reinterpret_cast<float*>(add1_input0_tensor->data)[0];
  } else {
    throw std::invalid_argument("When fuse laynorm eps must be a const value.");
  }

  // get gamma
  csinn_tensor* gamma_tensor;
  auto mul_input0 = mul_node->InputDefs()[0]->Name();
  auto mul_input1 = mul_node->InputDefs()[1]->Name();
  auto mul_input0_tensor = shl_tensor_map.at(mul_input0);
  auto mul_input1_tensor = shl_tensor_map.at(mul_input1);
  if (mul_input1_tensor->is_const) {
    gamma_tensor = mul_input1_tensor;
  } else {
    gamma_tensor = mul_input0_tensor;
  }

  // get beta
  csinn_tensor* beta_tensor;
  auto add2_input0 = add2_node->InputDefs()[0]->Name();
  auto add2_input1 = add2_node->InputDefs()[1]->Name();
  auto add2_input0_tensor = shl_tensor_map.at(add2_input0);
  auto add2_input1_tensor = shl_tensor_map.at(add2_input1);
  if (add2_input1_tensor->is_const) {
    beta_tensor = add2_input1_tensor;
  } else {
    beta_tensor = add2_input0_tensor;
  }

  // input
  std::string input_name = first_node->InputDefs()[0]->Name();
  auto input_tensor = shl_tensor_map.at(input_name);

  auto params = shl_ep::GetShlParams<csinn_layer_norm_params>(session_);
  params->base.name = const_cast<char*>(reduce_mean_node->Name().c_str());

  params->axis = axes_values[0] >= 0 ? axes_values[0] : axes_values[0] + input_tensor->dim_count;
  params->epsilon = eps;
  params->center = true;
  params->scale = true;
  std::string output = add2_node->OutputDefs()[0]->Name();
  auto output_tensor = shl_tensor_map.at(output);
  CheckQuantDtype<csinn_tensor>(input_tensor, output_tensor, gamma_tensor, beta_tensor);
  csinn_layer_norm_init(input_tensor, output_tensor, gamma_tensor, beta_tensor, params);
  csinn_layer_norm(input_tensor, output_tensor, gamma_tensor, beta_tensor, params);
}

csinn_quant_info* OnnxToShlConverter::GetQuantInfo(const Node* node) {
  NodeAttrHelper helper(*node);
  auto has_axis = helper.HasAttr("axis");
  const auto& scale_name = node->InputDefs()[1]->Name();
  const auto& scale_tensor = shl_tensor_map.at(scale_name);
  const int scale_size = csinn_tensor_size(scale_tensor);

  if (has_axis) {
    // per axis
    const int axis = helper.Get("axis", 1);
    if (!is_one_of(axis, 0, 1)) {
      throw std::invalid_argument("Per channel quant unsupported axis != 0 or 1, now axis=" + std::to_string(axis));
    }
  }

  csinn_quant_info* q_infos = new csinn_quant_info[scale_size];
  if (node->InputDefs().size() == 3) {
    const auto& zero_point_name = node->InputDefs()[2]->Name();
    const auto& zero_point_tensor = shl_tensor_map.at(zero_point_name);
    const int zp_size = csinn_tensor_size(zero_point_tensor);
    if (scale_size != zp_size) {
      throw std::invalid_argument(
          "Per channel quant scale size must equle to zero point size, but scale size=" +
          std::to_string(scale_size) + ", zero point size=" + std::to_string(zp_size));
    }
    for (int i = 0; i < scale_size; i++) {
      q_infos[i].scale = (reinterpret_cast<float*>(scale_tensor->data))[i];
      if (zero_point_tensor->dtype == CSINN_DTYPE_UINT8) {
        q_infos[i].zero_point = (reinterpret_cast<uint8_t*>(zero_point_tensor->data))[i];
      } else if (zero_point_tensor->dtype == CSINN_DTYPE_INT8) {
        q_infos[i].zero_point = (reinterpret_cast<int8_t*>(zero_point_tensor->data))[i];
      } else if (zero_point_tensor->dtype == CSINN_DTYPE_INT32) {
        q_infos[i].zero_point = (reinterpret_cast<int32_t*>(zero_point_tensor->data))[i];
      } else {
        throw std::invalid_argument("Quant supported dtype {uint8, int8}, but get dtype=" + std::to_string(zero_point_tensor->dtype));
      }
    }
  } else {
    for (int i = 0; i < scale_size; i++) {
      q_infos[i].scale = (reinterpret_cast<float*>(scale_tensor->data))[i];
      q_infos[i].zero_point = 0;
    }
  }

  return q_infos;
}

void OnnxToShlConverter::ProcessDQDNodes(const Node* q_node, const Node* dq_node) {
  auto q_input_name = q_node->InputDefs()[0]->Name();
  auto q_input_tensor = shl_tensor_map.at(q_input_name);
  auto dq_output_name = dq_node->OutputDefs()[0]->Name();
  auto dq_output_tensor = shl_tensor_map.at(dq_output_name);

  if (q_input_tensor == dq_output_tensor) {
    throw std::invalid_argument("q_input_tensor == dq_output_tensor.");
  }

  // since q_node is before dq_node as we check,
  // q_in == q_out == dq_in, so just set dq_out as q_in
  shl_tensor_map[dq_output_name] = q_input_tensor;
}

void OnnxToShlConverter::ProcessDQNodes(const Node* dq_node) {
  csinn_quant_info* q_infos = GetQuantInfo(dq_node);
  if (dq_node->InputDefs().size() != 3 || dq_node->OutputDefs().size() != 1) {
    throw std::invalid_argument("dq_node input size must be 3 and output size must be 1.");
  }
  auto dq_in_name = dq_node->InputDefs()[0]->Name();
  auto dq_out_name = dq_node->OutputDefs()[0]->Name();
  auto dq_in_tensor = shl_tensor_map.at(dq_in_name);
  auto dq_out_tensor = shl_tensor_map.at(dq_out_name);

  // e.g.: dq is uint8 to float, so set input to output
  dq_in_tensor->qinfo = q_infos;
  shl_tensor_map[dq_out_name] = dq_in_tensor;
  dq_out_tensor->name = const_cast<char*>("dirty data");

  // csinn_free_tensor(dq_out_tensor);
}

void OnnxToShlConverter::ProcessQNodes(const Node* q_node) {
  csinn_quant_info* q_infos = GetQuantInfo(q_node);
  if (q_node->InputDefs().size() != 3 || q_node->OutputDefs().size() != 1) {
    throw std::invalid_argument("q_node input size must be 3 and output size must be 1.");
  }
  auto q_in_name = q_node->InputDefs()[0]->Name();
  auto q_out_name = q_node->OutputDefs()[0]->Name();
  auto q_in_tensor = shl_tensor_map.at(q_in_name);
  auto q_out_tensor = shl_tensor_map.at(q_out_name);

  // e.g.: q is float to uint8, so set output to input
  q_out_tensor->qinfo = q_infos;
  q_out_tensor->name = q_in_tensor->name;
  shl_tensor_map[q_in_name] = q_out_tensor;
  q_in_tensor->name = const_cast<char*>("dirty data");
  // csinn_free_tensor( q_in_tensor);
}

void OnnxToShlConverter::FuseDQDNodes(std::unordered_map<std::string, const Node*> nodes_map) {
  auto key_node = nodes_map["key_node"];
  const auto& op_type = key_node->OpType();
  if (shl_ops_map.find(op_type) == shl_ops_map.end()) {
    throw std::invalid_argument("Unsupported operator " + op_type);
  }

  (this->*(shl_ops_map.at(op_type)))(*key_node);
}

}  // namespace shl_ep
}  // namespace onnxruntime
