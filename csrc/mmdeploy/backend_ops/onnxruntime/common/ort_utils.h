// Copyright (c) OpenMMLab. All rights reserved.
#ifndef ORT_MMCV_UTILS_H
#define ORT_MMCV_UTILS_H
#include <onnxruntime_cxx_api.h>

#include <unordered_map>
#include <vector>

namespace mmdeploy {

typedef std::unordered_map<std::string, std::vector<OrtCustomOp*>> CustomOpsTable;

// struct OrtTensorDimensions : std::vector<int64_t> {
//   OrtTensorDimensions(const OrtApi& ort, const OrtValue* value) {
//     // OrtTensorTypeAndShapeInfo* info = ort.GetTensorTypeAndShape(value);
//     // std::vector<int64_t>::operator=(ort.GetTensorShape(info));
//     // ort.ReleaseTensorTypeAndShapeInfo(info);

//     OrtTensorTypeAndShapeInfo *info;
//     ort.GetTensorTypeAndShape(value, &info);
//     int64_t* dim_values;
//     size_t dim_values_length;
//     ort.GetDimensionsCount(info, &dim_values_length);
//     ort.GetDimensions(info, dim_values, dim_values_length);
//     std::vector<int64_t>::assign(dim_values, dim_values + dim_values_length);
//     ort.ReleaseTensorTypeAndShapeInfo(info);
//   }
// };

CustomOpsTable& get_mmdeploy_custom_ops();

template <char const* domain, typename T>
class OrtOpsRegistry {
 public:
  OrtOpsRegistry() { get_mmdeploy_custom_ops()[domain].push_back(&instance); }

 private:
  T instance{};
};

#define REGISTER_ONNXRUNTIME_OPS(domain, name)     \
  static char __domain_##domain##name[] = #domain; \
  static OrtOpsRegistry<__domain_##domain##name, name> ort_ops_registry_##domain##name {}

}  // namespace mmdeploy
#endif  // ORT_MMCV_UTILS_H
