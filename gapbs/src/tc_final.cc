// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

// g++ --std=c++17 -O3 -Wall -fopenmp -w tc_final.cc -lpoplar -lpopops -lpoplin -lpoputil -o tc

// Encourage use of gcc's parallel algorithms (for sort for relabeling)
#ifdef _OPENMP
  #define _GLIBCXX_PARALLEL
#endif

#include <algorithm>
#include <cinttypes>
#include <iostream>
#include <vector>
#include <chrono>

#ifdef __IPU__
#include <poplar/HalfFloat.hpp>
#else
#include <Eigen/Dense>
using half = Eigen::half;
#endif

#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "pvector.h"

// Graphcore Poplar headers
#include <poplar/OptionFlags.hpp>
#include <poplar/IPUModel.hpp>
#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poplar/CycleCount.hpp>
#include <poputil/TileMapping.hpp>
#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>
#include <poplin/MatMul.hpp>
#include <poplin/codelets.hpp>
#include <popops/codelets.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Reduce.hpp>
#include <popops/Fill.hpp>
#include <pva/pva.hpp>

inline
half nextHalfUp(half h) {
  static_assert(sizeof(std::uint16_t) == sizeof(half), "Eigen half badly sized.");
  std::uint16_t bits;
  std::memcpy(&bits, &h, sizeof(bits));
  bits += 1;
  half result;
  std::memcpy(&result, &bits, sizeof(bits));
  return result;
}

inline
half roundToHalfNotSmaller(float f) {
  half h = (half)f;
  float ff = (float)h;
  if (ff < f) {
    h = nextHalfUp(h);
  }
  return h;
}

// To avoid confusion between Poplar Graph and GAP's input Graph
typedef CSRGraph<NodeID> Network;
/*
GAP Benchmark Suite
Kernel: Triangle Counting (TC)
Author: Scott Beamer

Will count the number of triangles (cliques of size 3)

Requires input graph:
  - to be undirected
  - no duplicate edges (or else will be counted as multiple triangles)
  - neighborhoods are sorted by vertex identifiers

Other than symmetrizing, the rest of the requirements are done by SquishCSR
during graph building.

This implementation reduces the search space by counting each triangle only
once. A naive implementation will count the same triangle six times because
each of the three vertices (u, v, w) will count it in both ways. To count
a triangle only once, this implementation only counts a triangle if u > v > w.
Once the remaining unexamined neighbors identifiers get too big, it can break
out of the loop, but this requires that the neighbors to be sorted.

Another optimization this implementation has is to relabel the vertices by
degree. This is beneficial if the average degree is high enough and if the
degree distribution is sufficiently non-uniform. To decide whether or not
to relabel the graph, we use the heuristic in WorthRelabelling.
*/


using namespace std;
using namespace poplar;
using namespace poplar::program;
using namespace popops;


// Utility code to acquire a real IPU device
poplar::Device getIpuHwDevice(std::size_t numIpus) {
  auto dm = poplar::DeviceManager::createDeviceManager();
  auto hwDevices = dm.getDevices(poplar::TargetType::IPU, numIpus);
  
  auto it =
      std::find_if(hwDevices.begin(), hwDevices.end(),
                   [](poplar::Device &device) { return device.attach(); });
  if (it != hwDevices.end()) {
    return std::move(*it);
  }
  throw std::runtime_error("No IPU hardware available.");
}

size_t countTriangles(const Network &g) {

    const unsigned NUM_IPUS = 4;
    Device device = getIpuHwDevice(NUM_IPUS);
    Target target = device.getTarget();
    poplar::Graph graph(target);
    popops::addCodelets(graph);
    poplin::addCodelets(graph);

    long unsigned int N = g.num_nodes();
    long unsigned int blocksize = 1024;//std::ceil((float)N / NUM_IPUS);
    N = (N % blocksize == 0) ? N : blocksize * (N / blocksize + 1);  // Padding to ensure compatibility with arbitrary input size
    long unsigned int blocksPerRow = N / blocksize;
    long unsigned int numBlocks = blocksPerRow * blocksPerRow;
    unsigned tilesPerIPU = 1472;
    auto hMatrix = std::vector<float>(N * N, (float)0.0f);
    auto hLMatrix = std::vector<float>(N * N, (float)0.0f);
    auto hUMatrix = std::vector<float>(N * N, (float)0.0f);
    auto result = std::vector<float>(1, (float)0.0f);
    auto mask = std::vector<int>(numBlocks, 0);
    auto hostBlockA = std::vector<std::vector<float>>(numBlocks, std::vector<float>(blocksize * blocksize));
    auto hostBlockL = std::vector<std::vector<float>>(numBlocks, std::vector<float>(blocksize * blocksize));
    auto hostBlockU = std::vector<std::vector<float>>(numBlocks, std::vector<float>(blocksize * blocksize));

    std::cout << "Input Size: " << g.num_nodes() << std::endl;
    std::cout << "Matrix: " << N << std::endl;
    std::cout << "IPUs: " << NUM_IPUS << std::endl;
    std::cout << "blocksize: " << blocksize << std::endl;
    std::cout << "numblocks: " << numBlocks << std::endl;

    // Initialize Adjacency from the network
    for (int64_t i = 0; i < g.num_nodes(); i++) {
      for (auto j : g.out_neigh(i)) {
        hMatrix[i * N + j] = (float)1.0f;
        if (i < j)
          hUMatrix[i * N + j] = (float)1.0f;
        else
          hLMatrix[i * N + j] = (float)1.0f;
      }
    }
    
    std::cout << "Host adjacency, lower, and upper initialized" << std::endl;

    // Tile up the Adjacency into 'numBlocks'-many blocks of size blocksize X blocksize
    int count = 0;
    for (int64_t b = 0; b < numBlocks; b++) {
      size_t sum = 0;
      int64_t idx = 0; 
      
      for (int64_t i = ((int)(b / (N / blocksize))) * blocksize; i < ((int)(b / (N / blocksize))) * blocksize + blocksize; i++) {
          for (int64_t j = ((int)(b % (N / blocksize))) * blocksize; j < (((int)(b % (N / blocksize))) * blocksize) + blocksize; j++) {
              
            hostBlockA[b][idx] = (hMatrix[i * N + j]);
            hostBlockL[b][idx] = (hLMatrix[i * N + j]);
            hostBlockU[b][idx] = (hUMatrix[i * N + j]);
            sum += (hMatrix[i * N + j]);
            idx++;
              
          }
      }
      if (sum > 0)
        mask[b] = 1; 
        count ++;
    }
    
    std::cout << "Decomposition into blocks on host complete" << std::endl;
    std::cout << "Number of blocks ignored: " << (numBlocks - count) << "/" << numBlocks << std::endl;

    Sequence mul, eleMul;
    Sequence copyIn, copyOut;

    std::vector<std::vector<Tensor>> matLHS(numBlocks);
    std::vector<std::vector<Tensor>> matRHS(numBlocks);
    std::vector<Tensor> matB(numBlocks);
    std::vector<Tensor> matA(numBlocks);
    std::vector<Tensor> reduction;

    std::vector<std::vector<DataStream>> inStreamLHS(numBlocks);
    std::vector<std::vector<DataStream>> inStreamRHS(numBlocks);
    std::vector<DataStream> inStreamA(numBlocks);
    std::vector<DataStream> outStream(numBlocks);
    std::vector<poplar::Graph> vg(NUM_IPUS);

    std::vector<int> weight(numBlocks, 1);
    std::vector<int> mapToIPU(numBlocks, NUM_IPUS - 1);

    // Load balancing logic
    // for (int64_t i = 0; i < blocksPerRow; i++) {
      
    //   for (int64_t j = 0; j < blocksPerRow; j++) {
        
    //     for (int64_t k = 0; k < blocksPerRow; k++) {

    //       if ((i >= k) && (k <= j)) {
    //         weight[i * blocksPerRow + j] += 1;
    //       }

    //     }
      
    //   }

    // }

    for (int64_t i = 1; i < numBlocks; i++) {
      weight[i] += weight[i - 1];
    }

    std::cout << "Workload per block calculated" << std::endl;

    int ipu = 0;
    int max_weight = weight[numBlocks - 1];
    int weightPerIPU = std::ceil((float)max_weight / NUM_IPUS);
  
    
    // Load balancing logic
    for (int64_t i = 0; i < numBlocks; i++) {      
    
      if (weight[i] <= (ipu + 1) * weightPerIPU) {
        mapToIPU[i] = ipu;
      }
      else {
        ipu++;
        mapToIPU[i] = ipu;
      }        

    }

    // for (int64_t i = 0; i < numBlocks; i++) {
    //   std::cout << weight[i] << " ";
    // }
    // std::cout << std::endl;

    // for (int64_t i = 0; i < numBlocks; i++) {
    //   std::cout << mapToIPU[i] << " ";
    // }
    // std::cout << std::endl;



    std::cout << "Load balance map created" << std::endl;

    for (int64_t i = 0; i < NUM_IPUS; i++) {

      // Virtual graph for every IPU
    
      unsigned startTile = (i) * tilesPerIPU;
      unsigned endTile = startTile + tilesPerIPU;

      vg[i] = graph.createVirtualGraph(startTile, endTile);
    }

    std::cout << "Virtual graphs initialized" << std::endl;

    // placeholder for LHS, RHS, and Adjacency blocks
    bool copyAdj = false;
    
    // #pragma omp parallel 
    for (int64_t i = 0; i < blocksPerRow; i++) {
      
      for (int64_t j = 0; j < blocksPerRow; j++) {
        copyAdj = false;
        
        // && (mask[k * blocksPerRow + j])
        for (int64_t k = 0; k < blocksPerRow; k++) {
          if ((i >= k) && (k <= j) && (mask[i * blocksPerRow + k]) && (mask[k * blocksPerRow + j])) {
            
            
            copyAdj = true;
            //LHS
            inStreamLHS[i * blocksPerRow + j].push_back(vg[mapToIPU[i * blocksPerRow + j]].addHostToDeviceFIFO("LHS_" + std::to_string(i * blocksPerRow + j) + "_" +std::to_string(k), FLOAT, blocksize * blocksize));
            matLHS[i * blocksPerRow + j].push_back(poplin::createMatMulInputLHS(vg[mapToIPU[i * blocksPerRow + j]], FLOAT, {blocksize, blocksize}, {blocksize, blocksize}, "LHS_" + std::to_string(i * blocksPerRow + j) + "_" +std::to_string(k)));
            

            //RHS
            inStreamRHS[i * blocksPerRow + j].push_back(vg[mapToIPU[i * blocksPerRow + j]].addHostToDeviceFIFO("RHS_" + std::to_string(i * blocksPerRow + j) + "_" +std::to_string(k), FLOAT, blocksize * blocksize));
            matRHS[i * blocksPerRow + j].push_back(poplin::createMatMulInputRHS(vg[mapToIPU[i * blocksPerRow + j]], FLOAT, {blocksize, blocksize}, {blocksize, blocksize}, "RHS_" + std::to_string(i * blocksPerRow + j) + "_" +std::to_string(k)));
            

            //matmul result
            matB[i * blocksPerRow + j] = poplin::createMatMulInputLHS(vg[mapToIPU[i * blocksPerRow + j]], FLOAT, {blocksize, blocksize}, {blocksize, blocksize}, "B_" + std::to_string(i * blocksPerRow + j));
            
            // Initialize the matmul results to zero
            popops::fill(vg[mapToIPU[i * blocksPerRow + j]], matB[i * blocksPerRow + j], mul, 0.0f);

            // Replaced the matmul initialization with popops::fill instead of copying a zero tensor
            // Tensor z = vg[mapToIPU[i * blocksPerRow + j]].addConstant<float>(FLOAT, {blocksize, blocksize}, 0.0f);
            // vg[mapToIPU[i * blocksPerRow + j]].setTileMapping(z, vg[mapToIPU[i * blocksPerRow + j]].getTileMapping(matB[i * blocksPerRow + j]));
            // copyIn.add(Copy(z, matB[i * blocksPerRow + j]));
          }
        }
        
        // only copy the adjacency where needed
        if (copyAdj) {
          
          inStreamA[i * blocksPerRow + j] = vg[mapToIPU[i * blocksPerRow + j]].addHostToDeviceFIFO("A_" + std::to_string(i * blocksPerRow + j), FLOAT, blocksize * blocksize);
          matA[i * blocksPerRow + j] = poplin::createMatMulInputLHS(vg[mapToIPU[i * blocksPerRow + j]], FLOAT, {blocksize, blocksize}, {blocksize, blocksize}, "A_" + std::to_string(i * blocksPerRow + j));
          outStream[i * blocksPerRow + j] = vg[mapToIPU[i * blocksPerRow + j]].addDeviceToHostFIFO("Out_" + std::to_string(i * blocksPerRow + j), FLOAT, 1);
        
        }
      }
    }

    // auto outStream = graph.addDeviceToHostFIFO("out", FLOAT, 1);

    std::cout << "Streams and device Tensors created" << std::endl;

    // Copy in the tile data from host to IPU through streams
    for (int64_t i = 0; i < blocksPerRow; i++) {
      for (int64_t j = 0; j < blocksPerRow; j++) {

        copyAdj = false;
        for (int64_t k = 0; k < inStreamLHS[i * blocksPerRow + j].size(); k++) {
          
          copyAdj = true;
          copyIn.add(Copy(inStreamLHS[i * blocksPerRow + j][k], matLHS[i * blocksPerRow + j][k])); 
          copyIn.add(Copy(inStreamRHS[i * blocksPerRow + j][k], matRHS[i * blocksPerRow + j][k]));

        }

        if (copyAdj) {
          copyIn.add(Copy(inStreamA[i * blocksPerRow + j], matA[i * blocksPerRow + j]));
        }

      }
    }

    std::cout << "Host to device copy done" << std::endl;
    
    for (int64_t i = 0; i < numBlocks; i++) {
      
      // matmul
      for (int64_t j = 0; j < matLHS[i].size(); j++) {

        popops::addInPlace(vg[mapToIPU[i]], matB[i], poplin::matMul(vg[mapToIPU[i]], matLHS[i][j], matRHS[i][j], mul, FLOAT), mul, "ProgMatmul");
        
      }    
      
    }


    std::cout << "Matmul complete" << std::endl;

    for (int64_t i = 0; i < blocksPerRow; i++) {

      for (int64_t j = 0; j < blocksPerRow; j++) {

        copyAdj = false;
        for (int64_t k = 0; k < blocksPerRow; k++) {

          if ((i >= k) && (k <= j) && (mask[i * blocksPerRow + k]) && (mask[k * blocksPerRow + j])) {
            copyAdj = true;
          }

        }

        if (copyAdj) {
          
          popops::mulInPlace(vg[mapToIPU[i * blocksPerRow + j]], matA[i * blocksPerRow + j], matB[i * blocksPerRow + j], eleMul, "ProgElemul");
          reduction.push_back(popops::reduce(vg[mapToIPU[i * blocksPerRow + j]], matA[i * blocksPerRow + j], FLOAT, {0,1}, popops::Operation::ADD, eleMul, "ProgReduction"));
          
        }
        
      }    

    }
    
    std::cout << "Hadamard and partial reduce complete" << std::endl;

    // for (int64_t i = 1; i < reduction.size(); i++) {
    //   popops::addInPlace(graph, reduction[0], reduction[i], eleMul, "ProgFinalReduction");
    // }

    // for (int64_t i = 0; i < reduction.size(); i++) {
    //   vg[mapToIPU[i]].createHostRead("Reduction" + std::to_string(i), reduction[i]);
    // }

    for (int64_t i = 0; i < reduction.size(); i++) {
      copyOut.add(Copy(reduction[i], outStream[i]));
    }
    
    


    // copyOut.add(Copy(reduction[0], outStream));

    std::cout << "Device to host copy done" << std::endl;

    Engine engine(graph, {copyIn, mul, eleMul, copyOut});
    engine.load(device);

    std::cout << "Engine loaded" << std::endl;
    auto reduceResult = std::vector<std::vector<float>>(numBlocks, std::vector<float>(1, (float)0.0f));

    // Connect the input streams
    for (int64_t i = 0; i < blocksPerRow; i++) {
      for (int64_t j = 0; j < blocksPerRow; j++) {
        
        copyAdj = false;
        for (int64_t k = 0; k < blocksPerRow; k++) {
          if ((i >= k) && (k <= j) && (mask[i * blocksPerRow + k]) && (mask[k * blocksPerRow + j])) {

            copyAdj = true;
            engine.connectStream("LHS_" + std::to_string(i * blocksPerRow + j) + "_" +std::to_string(k), hostBlockL[i * blocksPerRow + k].data());
            engine.connectStream("RHS_" + std::to_string(i * blocksPerRow + j) + "_" +std::to_string(k), hostBlockU[k * blocksPerRow + j].data());

          }          

        }

        if (copyAdj) {
          engine.connectStream("A_" + std::to_string(i * blocksPerRow + j), hostBlockA[i * blocksPerRow + j].data());
          engine.connectStream("Out_" + std::to_string(i * blocksPerRow + j), reduceResult[i * blocksPerRow + j].data());
        }

      }
    }
    
    // Connect the output stream
    // engine.connectStream("out", result.data());
    
    
    std::cout << "Datastreams connected to engine" << std::endl;
    
    engine.run(0); // CopyIn
    engine.run(1); // Mul 
    engine.run(2); // EleMul   
    engine.run(3); // CopyOut
    
    std::cout << "Kernel executed" << std::endl;

    // auto reduceResult = std::vector<std::vector<float>>(reduction.size(), std::vector<float>(1));
    

    for (int64_t i = 0; i < numBlocks; i++) {
      // engine.readTensor("Reduction" + std::to_string(i), &reduceResult, &reduceResult + 1);
      result[0] += reduceResult[i][0];
    }

    return ((int)result[0]) / 2;

}

void PrintTriangleStats(const Network &g, size_t total_triangles) {
  cout << total_triangles << " triangles" << endl;
}


// Compares with simple serial implementation that uses std::set_intersection
bool TCVerifier(const Network &g, size_t test_total) {
  float total = 0.0f;
  vector<NodeID> intersection;
  intersection.reserve(g.num_nodes());
  for (NodeID u : g.vertices()) {
    for (NodeID v : g.out_neigh(u)) {
      auto new_end = set_intersection(g.out_neigh(u).begin(),
                                      g.out_neigh(u).end(),
                                      g.out_neigh(v).begin(),
                                      g.out_neigh(v).end(),
                                      intersection.begin());
      intersection.resize(new_end - intersection.begin());
      total += intersection.size();
    }
  }
  total = total / 6.0f;  // each triangle was counted 6 times
  if (total != test_total)
    cout << total << " != " << test_total << endl;
  return total == test_total;
}


int main(int argc, char* argv[]) {
  CLApp cli(argc, argv, "triangle count");
  if (!cli.ParseArgs())
    return -1;
  Builder b(cli);
  Network g = b.MakeGraph();
  if (g.directed()) {
    cout << "Input graph is directed but tc requires undirected" << endl;
    return -2;
  }

  BenchmarkKernel(cli, g, countTriangles, PrintTriangleStats, TCVerifier);


  return 0;
}