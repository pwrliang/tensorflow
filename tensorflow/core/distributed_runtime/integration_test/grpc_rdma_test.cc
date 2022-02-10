/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "gflags/gflags.h"
#include "tensorflow/c/c_api_experimental.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/c_api_internal.h"
#include "tensorflow/c/eager/c_api_test_util.h"
#include "tensorflow/c/eager/tfe_context_internal.h"
#include "tensorflow/c/eager/tfe_tensorhandle_internal.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/platform/blocking_counter.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/profiler/utils/time_utils.h"
#include "tensorflow/core/protobuf/cluster.pb.h"
#include "tensorflow/core/protobuf/coordination_config.pb.h"
#include "tensorflow/core/protobuf/tensorflow_server.pb.h"

DEFINE_string(hosts, "localhost,localhost", "ip list separated by ,");
DEFINE_int32(scale, 4, "Tensor size in byte");
DEFINE_int32(task_index, 0, "");
DEFINE_int32(repeat, 10, "");

namespace {

void StartCustomWorkers(
    const std::vector<std::string>& hosts, int worker_id,
    std::function<void(TFE_Context* ctx, TF_Status* status, int worker_id,
                       int cluster_size, int repeat)>
        fn) {
  tensorflow::ServerDef server_def =
      GetCustomMultiClientServerDef("worker", hosts);
  // Enable coordination service for propagating remote device attributess
  auto* config = server_def.mutable_default_session_config()
                     ->mutable_experimental()
                     ->mutable_coordination_config();
  config->set_service_type("standalone");
  config->set_service_leader("/job:worker/replica:0/task:0");

  auto cluster_size = hosts.size();
  // The blocking counter makes sure that worker/0 thread (leader that starts
  // the coordination service) does not exit early while other workers are still
  // interacting with the coordination service.
  // By default, server_def has task index set to 0.
  server_def.set_task_index(worker_id);
  std::string serialized = server_def.SerializeAsString();

  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_ContextOptionsSetAsync(opts, static_cast<unsigned char>(/*enable=*/true));
  TFE_ContextOptionsSetDevicePlacementPolicy(opts, TFE_DEVICE_PLACEMENT_SILENT);

  tensorflow::SessionOptions options;
  options.config = server_def.default_session_config();
  opts->session_options.options = options;

  TFE_Context* ctx = TFE_NewContext(opts, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteContextOptions(opts);

  TFE_EnableCollectiveOps(ctx, serialized.data(), serialized.size(), status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  TFE_Executor* executor = TFE_ContextGetExecutorForThread(ctx);

  for (int i = 0; i < FLAGS_repeat; i++) {
    fn(ctx, status, worker_id, cluster_size, i);
  }

  // Since we created an async EagerContext, wait for all pending operations
  // to finish before deleting the context.
  TFE_ExecutorWaitForAllPendingNodes(executor, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteExecutor(executor);

  TFE_DeleteContext(ctx);

  TF_DeleteStatus(status);
}

std::vector<std::string> split(const std::string& text, char delim) {
  std::string line;
  std::vector<std::string> vec;
  std::stringstream ss(text);
  while (std::getline(ss, line, delim)) {
    vec.push_back(line);
  }
  return vec;
}

TEST(CAPI, RDMACollectiveOps) {
  auto scale = FLAGS_scale;
  auto hosts = split(FLAGS_hosts, ',');
  auto task_index = FLAGS_task_index;

  auto fn = [scale](TFE_Context* ctx, TF_Status* status, int worker_id,
                    int cluster_size, int repeat) {
    int n_elem = (float)scale / sizeof(int) * 1024 * 1024;
    int* data = new int[n_elem];
    int* result = new int[n_elem];
    int64_t dims[] = {n_elem};

    for (int i = 0; i < n_elem; i++) {
      data[i] = i;
    }

    TFE_TensorHandle* in = TestTensorHandleWithDimsInt(
        ctx, data, dims, sizeof(dims) / sizeof(int64_t));
    TFE_Op* allreduce = AllReduceOp(ctx, in, cluster_size);
    TFE_TensorHandle* retvals[1];
    int num_retvals = 1;

    auto begin = tensorflow::profiler::GetCurrentTimeNanos();
    TFE_Execute(allreduce, &retvals[0], &num_retvals, status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    TF_Tensor* t = TFE_TensorHandleResolve(retvals[0], status);
    ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    EXPECT_EQ(n_elem * sizeof(int), TF_TensorByteSize(t));
    auto end = tensorflow::profiler::GetCurrentTimeNanos();

    LOG(INFO) << "Worker: " << worker_id << " Repeat: " << repeat
              << " Time: " << (end - begin) / 1000 / 1000 << " ms Tensor size: "
              << (float)(sizeof(int) * n_elem) / 1024 / 1024 << " MB";

    memcpy(&result[0], TF_TensorData(t), TF_TensorByteSize(t));
    TF_DeleteTensor(t);

    for (int i = 0; i < n_elem; i++) {
      EXPECT_EQ(i * cluster_size, result[i]);
    }

    TFE_DeleteTensorHandle(in);
    TFE_DeleteTensorHandle(retvals[0]);
    TFE_DeleteOp(allreduce);
    delete[] result;
    delete[] data;
  };
  StartCustomWorkers(hosts, task_index, fn);
}

}  // namespace

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  google::ParseCommandLineFlags(&argc, &argv, true);
  return RUN_ALL_TESTS();
}