#pragma once

#include <dirent.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

class NumaDetector {
   public:
    NumaDetector() { detectNumaNodes(); }

    ~NumaDetector() {}

    int getNumaNodes() const { return numaNodes; }

    std::vector<int> getCoresForNode(int node) const {
        auto it = node2Cores.find(node);
        if (it != node2Cores.end()) {
            return it->second;
        } else {
            throw std::out_of_range("NUMA node not found");
        }
    }

    std::vector<int> getPhyCoresForNode(int node) const {
        auto it = node2PhyCores.find(node);
        if (it != node2PhyCores.end()) {
            return it->second;
        } else {
            throw std::out_of_range("NUMA node not found");
        }
    }

   private:
    void detectNumaNodes() {
        {
            DIR* dir = opendir("/sys/devices/system/node");
            if (!dir) {
                throw std::runtime_error("Failed to open /sys/devices/system/node");
            }

            numaNodes = 0;
            std::regex nodeRegex("^node[0-9]+$");
            while (auto entry = readdir(dir)) {
                if (entry->d_type == DT_DIR && std::regex_match(entry->d_name, nodeRegex)) {
                    ++numaNodes;
                }
            }
            closedir(dir);
        }

        // Map cores to NUMA nodes
        for (int node = 0; node < numaNodes; ++node) {
            std::ostringstream path;
            path << "/sys/devices/system/node/node" << node << "/cpulist";
            std::ifstream cpuFile(path.str());
            if (!cpuFile.is_open()) {
                throw std::runtime_error("Failed to open " + path.str());
            }

            std::string line;
            if (std::getline(cpuFile, line)) {
                std::istringstream iss(line);
                std::string range;
                while (std::getline(iss, range, ',')) {
                    size_t dashPos = range.find('-');
                    if (dashPos != std::string::npos) {
                        int start = std::stoi(range.substr(0, dashPos));
                        int end = std::stoi(range.substr(dashPos + 1));
                        for (int core = start; core <= end; ++core) {
                            node2Cores[node].push_back(core);
                        }
                    } else {
                        node2Cores[node].push_back(std::stoi(range));
                    }
                }
            }
            cpuFile.close();
        }

        // Map physical cores to NUMA nodes
        for (const auto& [node, cores] : node2Cores) {
            for (int core : cores) {
                std::ostringstream path;
                path << "/sys/devices/system/cpu/cpu" << core << "/topology/thread_siblings_list";
                std::ifstream siblingFile(path.str());
                if (!siblingFile.is_open()) {
                    throw std::runtime_error("Failed to open " + path.str());
                }

                std::string siblingLine;
                if (std::getline(siblingFile, siblingLine)) {
                    size_t commaPos = siblingLine.find(',');
                    int physicalCore = std::stoi(siblingLine.substr(0, commaPos));
                    if (physicalCore == core) {
                        node2PhyCores[node].push_back(core);
                    }
                }
                siblingFile.close();
            }
        }
    }

   private:
    int numaNodes;
    std::unordered_map<int, std::vector<int>> node2Cores;
    std::unordered_map<int, std::vector<int>> node2PhyCores;
};