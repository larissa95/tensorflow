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

#include "tensorflow/core/common_runtime/costum_placer.h"

#include <memory>
#include <set>
#include <utility>
#include <vector>
#include <iostream>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"

namespace tensorflow {
    
    namespace {
        
        // Returns a list of devices sorted by preferred type and then name
        // from 'devices' whose type is in 'supported_device_types'.  This
        // function searches the device types in 'supported_device_types' and
        // returns the subset of devices that match.
        std::vector<Device*> FilterSupportedDevices(
                                                    const std::vector<Device*>& devices,
                                                    const DeviceTypeVector& supported_device_types) {
            std::vector<Device*> filtered_devices;
            for (const DeviceType& d : supported_device_types) {
                for (Device* device : devices) {
                    if (DeviceType(device->attributes().device_type()) == d) {
                        filtered_devices.emplace_back(device);
                    }
                }
            }
            
            auto device_sort = [](const Device* a, const Device* b) {
                // First sort by prioritized device type and then by device name.
                return std::make_pair(
                                      DeviceSet::DeviceTypeOrder(DeviceType(a->device_type())),
                                      StringPiece(a->name())) <
                std::make_pair(
                               DeviceSet::DeviceTypeOrder(DeviceType(b->device_type())),
                               StringPiece(b->name()));
            };
            std::sort(filtered_devices.begin(), filtered_devices.end(), device_sort);
            return filtered_devices;
        }
    } //namespace
    
    CostumPlacer::CostumPlacer(Graph* graph, const DeviceSet* devices,
                               const SessionOptions* options,const CostModel* cost_model)
    : graph_(graph), devices_(devices), options_(options),cost_model_(cost_model) {}
    
    CostumPlacer::CostumPlacer(Graph* graph, const DeviceSet* devices)
    : graph_(graph), devices_(devices) {
        options_ = nullptr;
        cost_model_ = nullptr;
    }
    
    CostumPlacer::~CostumPlacer() {}
    
    Status CostumPlacer::Run() {
        if (devices_->devices().empty()) {
            return errors::FailedPrecondition("No devices are registered");
        }
        const std::vector<Device*>& devices = devices_->devices();
        std::vector<string> device_names;
        for(std::vector<Device*>::const_iterator it = devices.begin(); it != devices.end(); ++it){
            std::cout << (*it)->DebugString() << std::endl;
            device_names.push_back((*it)->name());
        }
        
        std::cout << "Anzahl Knoten" <<graph_->num_nodes() << std::endl;
        int hash = 0;
        for(Node* node: graph_->nodes()){
            if(!node->IsOp()){
                std::cout << "no operation" << std::endl;
            }else{
                int costId = node->cost_id();
                std::cout << "assigned to first device: " << node->DebugString() << std::endl << "CostId" << (char) costId << std::endl;
                //node->in_edges(); node->out_edges(),
                //bringt nicht weil nie durchgelaufen
                //std::cout << "Size estimate: " << cost_model_->SizeEstimate(node, 0) << std::endl;
                
                hash = (hash + 1) % device_names.size();
                std::cout << "Hash:" << hash << std::endl;
                string assigned_device = device_names.at(0);
                if (CanAssignToDevice(assigned_device, devices)) {
                    AssignAndLog(assigned_device, node);
                }else{
                    std::cout << "Deivce could not be assigned" << std::endl;
                }
            }
            
        }
        /*
         status = colocation_graph.GetDevicesForNode(node, &devices);
         devices = FilterSupportedDevices(
         device_set_->devices(), members_[node_root].supported_device_types);
         */
        return Status::OK();
    }
    
    bool CostumPlacer::CanAssignToDevice(const string& candidate_device_name,
                                         const std::vector<Device*> devices) const {
        if (!candidate_device_name.empty()) {
            // Can we assign to the same device?  Check by validating that
            // the device type of 'candidate_device_name' is present
            // in 'devices'.
            const Device* other_device =
            devices_->FindDeviceByName(candidate_device_name);
            if (std::any_of(devices.begin(), devices.end(), [other_device](Device* d) {
                return d->device_type() == other_device->device_type();
            })) {
                return true;
            }
        }
        
        return false;
    }
    
    void CostumPlacer::AssignAndLog(const string& assigned_device,
                                    Node* node) const {
        node->set_assigned_device_name(assigned_device);
        // Log placement if log_device_placement is set.
        if (options_ && options_->config.log_device_placement()) {
            printf("%s: %s\n", node->name().c_str(),
                   node->assigned_device_name().c_str());
            LOG(INFO) << node->name() << ": " << node->assigned_device_name();
        }
    }
}  // namespace tensorflow
