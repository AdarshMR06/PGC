# Viva Questions – Theme 5

1. **What is a CUDA thread?** The smallest execution unit in a CUDA kernel.
2. **What is a thread block?** A group of threads that can cooperate and share shared memory.
3. **What is a grid?** All blocks launched by one kernel invocation.
4. **What is a warp?** A group of 32 threads scheduled together on NVIDIA GPUs.
5. **What is an SM?** A Streaming Multiprocessor that executes thread blocks and schedules warps.
6. **What is SIMT?** Single Instruction, Multiple Threads execution.
7. **What is occupancy?** Active warps divided by the maximum supported active warps per SM.
8. **What is a register?** Fast per-thread storage held in the SM register file.
9. **Why can registers reduce occupancy?** High register use per thread leaves resources for fewer resident warps or blocks.
10. **What is shared memory?** Fast on-chip memory shared by threads in a block.
11. **Why can shared memory reduce occupancy?** Each resident block reserves its requested shared-memory allocation.
12. **What is block-size tuning?** Testing different threads-per-block values to find an efficient configuration.
13. **What does CUDA event timing measure?** Elapsed GPU time between recorded events.
14. **Why warm up a kernel?** To reduce first-use effects such as initialization and clock ramp-up.
15. **Why repeat measurements?** To reduce noise and observe stable timing.
16. **What does cudaOccupancyMaxActiveBlocksPerMultiprocessor do?** Estimates active blocks per SM for a kernel, block size and dynamic shared-memory size.
17. **What does cudaOccupancyMaxPotentialBlockSize do?** Suggests a block size and minimum grid size for occupancy.
18. **What does ptxas report?** Compiler resource information including registers and shared memory.
19. **Does maximum occupancy guarantee maximum speed?** No; memory bandwidth, instruction mix and latency also matter.
20. **What is latency hiding?** Scheduling other ready warps while one warp waits on a long-latency operation.
21. **What is SAXPY?** A vector operation: output[i] = a × x[i] + y[i].
22. **Why validate against a CPU reference?** To confirm GPU computation correctness.
23. **What is dynamic shared memory?** Per-block shared memory size specified at kernel launch.
24. **What is theoretical occupancy?** A resource-based estimate, not a direct measurement of achieved occupancy.
25. **How will you select the best block size?** Compare validated repeated timings on the actual GPU and explain the result using resource data.
