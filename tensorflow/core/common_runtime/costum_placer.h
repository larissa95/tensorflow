/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
 
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

#ifndef TENSORFLOW_COMMON_RUNTIME_COSTUM_PLACER_H_
#define TENSORFLOW_COMMON_RUNTIME_COSTUM_PLACER_H_

#include <string>
#include <unordered_map>

#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/graph/costmodel.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
    
    // A costum placement algorithm that assigns the nodes of the given Graph to
    // devices the given DeviceSet
    
    class CostumPlacer {
    public:
        
        // Creates an instance of the SimplePlacer algorithm for the given
        // Graph "graph" (nodes in which may or may not be assigned) on the
        // given DeviceSet "devices".
        //
        // The "graph", and "devices" pointer arguments
        // are borrowed by this CostumPlacer, and must outlive it.
        CostumPlacer(Graph* graph, const DeviceSet* devices,
                     const SessionOptions* options, const CostModel* cost_model);
        
        CostumPlacer(Graph* graph, const DeviceSet* devices);
        
        ~CostumPlacer();
        
        // Assigns each node in this SimplePlacer's graph to a device in its
        // set of devices.
        //
        // This method is not thread-safe.
        // Run() may be invoked at most once.
        Status Run();
        
    private:
        // Returns true if the device type of 'candidate_device_name' is
        // found in 'devices'.
        bool CanAssignToDevice(const string& candidate_device_name,
                               const std::vector<Device*> devices) const;
        
        // Assigns 'node's devices to 'assigned_device', and logs the
        // placement if the SessionOptions entry in 'options_' requests it.
        void AssignAndLog(const string& assigned_device, Node* node) const;
        
        Graph* const graph_;                           // Not owned.
        const DeviceSet* const devices_;               // Not owned.
        const SessionOptions* options_;                // Not owned.
        const CostModel* cost_model_;                  // Not owned.
        
        TF_DISALLOW_COPY_AND_ASSIGN(CostumPlacer);
    };
    
}  // namespace tensorflow

#endif  // TENSORFLOW_COMMON_RUNTIME_COSTUM_PLACER_H_
