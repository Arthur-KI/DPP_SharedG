Dynamically Path-Gated Perceptron with Shared Gating (DPP_SharedG)
1. Introduction & Motivation

This project introduces a novel neural network layer architecture: the Dynamically Path-Gated Perceptron (DPP) with a shared gating path (DPP_SharedG). The goal was to design an architecture capable of learning complex, state-dependent, and procedural logic tasks using very few parameters—tasks that are typically solved with classic algorithms or very large neural networks. The DPP_SharedG model demonstrates that even a single specialized layer within an otherwise simple network can be remarkably powerful.
2. Architecture and Functionality

The core model (DPPModelBase) consists of three main components:

    A single DPPLayer_SharedG (the key innovation)
    A ReLU activation function
    A linear output layer

The DPPLayer_SharedG combines two parallel processing paths ("A" and "B") for each output unit. A gating mechanism—comprising a shared and a unit-specific part—computes an alpha value for each unit, determining how much the result from path A or B is weighted. This allows the model to dynamically select different computational routes depending on the input.

At each timestep, the model receives a vector containing current data, context bits, and past states. The architecture is designed to model internal states, branching, and even simple program logic.
3. Key Test Results and Highlights (from Test7.py onwards)

The model was evaluated on a series of increasingly complex tasks, including:

    Test 7: Conditional accumulation—the model learns to use a control bit to choose between addition and simply carrying over the previous value. Achieved 100% accuracy with minimal parameters.
    Test 8: State-dependent operation—output depends on whether the previous output was 0 or 1 (conditional XOR/identity operation). Also achieved 100% test accuracy.
    Test 9: Counter with reset and mode selection—the model manages an internal modulo-4 counter, can reset it, and uses it to select between two operations. Again, perfect accuracy.
    Test 11: Mini-interpreter for 8 instructions (with two registers)—the model correctly interprets a simple machine instruction set and manages register states, demonstrating early CPU-like capabilities.
    Test 12: Stack machine—with subroutine stack, conditional jumps, and memory access. The model handles demanding procedural logic while maintaining very low parameter counts.
    Test 13 / FPLA V1: Simulation of a microcontroller core with multiple registers, memory, stack, flags, and 18 instructions. The model shows it can handle this complexity with just one layer and about 18,000 parameters.
    Test 24: 8-bit CPU—the model successfully simulates a simple 8-bit CPU with complex behavior (registers, memory, ALU, jumps).
    Test 26: 32-bit CPU—the model can also correctly simulate arithmetic and logical operations as well as memory access with 32-bit wide data paths.

It is especially noteworthy that the model converges very quickly in all tests, often reaching perfect accuracy in just a few epochs.
4. Properties and Advantages

    Parameter Efficiency: Even highly complex tasks are mastered with very few parameters.
    Fast Learning: The model often achieves high accuracy after only a few training epochs.
    Flexible Logic: The gating mechanism allows dynamic switching between different computational routes.
    Robustness & Scalability: The architecture works for tasks ranging from simple flip-flops to complex CPU cores and can be extended to wider data paths.
    Algorithmic Learning: The model learns not just input-output patterns, but actual logical rules and processes.

5. Application Possibilities

The DPP_SharedG model is particularly well-suited for:

    Simulation and emulation of classical logic (e.g., CPUs, automata, controllers)
    Research in "Neural Algorithmic Reasoning" and differentiable programming
    Educational purposes: demonstrating how neural networks can learn complex logic and even small programs
    Deployment in hardware-near or resource-constrained systems where parameter efficiency is crucial
    Serving as a building block for AI systems that require explicit, learnable logic components

Conclusion:
DPP_SharedG offers a highly efficient and powerful architecture that demonstrates how complex, state-dependent, and programmatic logic tasks can be solved with small neural networks. Its open license invites further research, adaptation, and experimentation for a wide range of new applications.
