#include "numa_detector.h"


// int main() {
//     try {
//         NumaDetector detector;
//         std::cout << "Number of NUMA nodes: " << detector.getNumaNodes() << std::endl;

//         for (int node = 0; node < detector.getNumaNodes(); ++node) {
//             std::cout << "Cores for NUMA node " << node << ": ";
//             for (int core : detector.getPhyCoresForNode(node)) {
//                 std::cout << core << " ";
//             }
//             std::cout << std::endl;
//         }
//     } catch (const std::exception &e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//     }

//     return 0;
// }