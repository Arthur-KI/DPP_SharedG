# Dynamic Path-Controlled Perceptron with Shared Gating (DPP_SharedG) for Algorithmic Logic

**Author/Discoverer:** Arthur Heyde
**Last Updated:** May 24, 2025
**License:** MIT License

## 1. Introduction & Motivation

This project explored a novel neural layer architecture, the **"Dynamic Path-controlled Perceptron" (DPP)**, with a focus on the variant with a partially shared gating path (**DPP_SharedG**). The goal was to develop a neuron layer with internal, learnable, and data-driven logic that exhibits high parameter efficiency and robustness for specific problem classes. The motivation was to investigate whether such architectures can better handle complex, conditional, and state-dependent tasks—especially those requiring programmatic or algorithmic logic—than standard MLP structures of comparable size.

The tests have shown that the `DPP_SharedG` model, even as a single specialized layer within an otherwise simple neural network, is capable of solving tasks that demand a high degree of programmatic logic, state management, and conditional execution. This was often achieved with a remarkably small number of parameters and very rapid convergence. The exploration was successfully extended from simple logic gates and complex 1-bit CPU cores (as demonstrated in FPLA V1 / Test20) to the simulation of simplified 8-bit (Test24) and 32-bit CPU logic (Test26), underscoring the architecture's scalability in terms of data width and instruction logic complexity.

## 2. Model Architecture: `DPPModelBase` with `DPPLayer_SharedG`

The core model (`DPPModelBase`) used in the successful sequential logic tests typically consists of:

1.  A single `DPPLayer_SharedG`.
2.  A ReLU activation function.
3.  A single linear output layer (`nn.Linear`).

The output of this layer is logits, which are then, for example, further processed by a sigmoid function for a binary cross-entropy loss function (BCEWithLogitsLoss).

### 2.1. The `DPPLayer_SharedG` Layer

This layer is the heart of the model and implements dynamic path control. For a given input vector $x \in \mathbb{R}^{D_{in}}$, each of the $H$ (e.g., `dpp_units`) parallel units in the layer calculates its output.

* Input: $x \in \mathbb{R}^{D_{in}}$
* Output of the layer: $z_{out} \in \mathbb{R}^{H}$ (before ReLU and the final linear layer)

Internal paths per unit $j$ (conceptually, as weights are defined for the entire layer):

* Path A (linear):
    $z_{Aj} = W_{Aj}x + b_{Aj}$
    Where $W_A \in \mathbb{R}^{H \times D_{in}}$ and $b_A \in \mathbb{R}^{H}$.
* Path B (linear):
    $z_{Bj} = W_{Bj}x + b_{Bj}$
    Where $W_B \in \mathbb{R}^{H \times D_{in}}$ and $b_B \in \mathbb{R}^{H}$.
* Gating Path G (with a shared core):
    * Shared Context Extraction: A linear transformation, common to all $H$ units in the layer, projects the input $x$ to a lower dimension (`shared_g_dim`, $D_{gsh}$). $x_{shared\_g} = W_{g\_shared}x + b_{g\_shared}$ Where $W_{g\_shared} \in \mathbb{R}^{D_{gsh} \times D_{in}}$ and $b_{g\_shared} \in \mathbb{R}^{D_{gsh}}$.
    * Unit-Specific Gating Logits: For each of the $H$ units, an individual gating logit $g_j$ is computed from $x_{shared\_g}$. $g_j = W_{g\_unit_j}x_{shared\_g} + b_{g\_unit_j}$ Where $W_{g\_unit} \in \mathbb{R}^{H \times D_{gsh}}$ and $b_{g\_unit} \in \mathbb{R}^{H}$.
    * Mixing Coefficient $\alpha$: $\alpha_j = \text{sigmoid}(g_j)$
* Final Output of DPP Unit $j$ (before layer-wide ReLU):
    $z_{final_j} = \alpha_j \cdot z_{Aj} + (1-\alpha_j) \cdot z_{Bj}$

In the PyTorch implementation, these operations are performed in parallel for all $H$ units using matrix multiplications at the layer level.

### 2.2. Overall Model `DPPModelBase` (Diagram)
Input (BatchSize, Din)
  |
  V
+-------------------------------------------------------------------+
| DPPLayer_SharedG                                                  |
|   - input_features: Din                                           |
|   - output_features (H): dpp_units                                |
|   - shared_g_dim: Dg_sh                                           |
|   Output: z_final (BatchSize, H)                                  |
+-------------------------------------------------------------------+
  |
  V
+-------------------------------------------------------------------+
| ReLU Activation                                                   |
|   Output: relu_out (BatchSize, H)                                 |
+-------------------------------------------------------------------+
  |
  V
+-------------------------------------------------------------------+
| Linear Output Layer (fc_out)                                      |
|   - in_features: H                                                |
|   - out_features: Dout (e.g., 1 for binary classification)        |
|   Output: logits (BatchSize, Dout)                                |
+-------------------------------------------------------------------+
  |
  V
Output (Logits, which then, for example, go through Sigmoid for BCEWithLogitsLoss)

## 3. Experimental History & Key Achievements

The model was tested on a series of increasingly complex sequential logic tasks that required "light memory" (feeding back the previous output $y_{t-1}$ as input) and processing of contextual information.

* Test 7: Conditional Accumulation
    * Task: $y_t = (y_{t-1} + x_t)\%2$ if $c_t=0$, else $y_t = y_{t-1}$. Input: $(x_t, c_t, y_{t-1})$.
    * Model: 129 parameters (Input=3, Units=8, SharedG=4).
    * Result: 100% test accuracy, target (99%) reached in 4 epochs.
    * Significance: Demonstrated the ability to react to an external context bit $c_t$ and change the internal operation accordingly.
* Test 8 (Log as Test9.py): State-Controlled Operation
    * Task: $y_t = (x_t \oplus 1)$ if $y_{t-1}=0$, else $y_t = x_t$. Input: $(x_t, y_{t-1})$.
    * Model: 87 parameters (Input=2, Units=8, SharedG=2).
    * Result: 100% test accuracy, target (99%) reached in 2 epochs.
    * Significance: Showed that the model can use its own previous output as the primary context for selecting the operation.
* Test 9 (Log as Test9.py, but more complex task): State-Controlled Counter with Reset and Operation
    * Task: Counter $z_t (\text{mod } 4)$ influences $y_t=f(x_t, y_{t-1})$ (XOR vs. AND), $c_t$ resets $z_t$. Input: $(x_t, c_t, y_{t-1}, \text{one-hot}(z_{t-1}))$.
    * Model: 385 parameters (Input=7, Units=16, SharedG=4).
    * Result: 100% test accuracy, target (98%) reached in 2 epochs.
    * Significance: Successfully handled a multi-value internal state ($z_{t-1}$) and an external control signal ($c_t$) to control the operation.
* Test 11 (Log as Test10.py / Your Test11.py): Simple Instruction Interpreter
    * Task: Execution of 8 instructions (LOAD, XOR, AND, NOT, OUT etc.) with 2 registers. Input: $(x_t, \text{instr}_{oh}, R0_{t-1}, R1_{t-1}, y_{t-1})$.
    * Model: 1167 parameters (Input=12, Units=32, SharedG=6).
    * Result: 100% test accuracy, target (90%) reached in epoch 1.
    * Significance: Ability to "understand" an instruction set and correctly manipulate register states. Began to show processor-like characteristics.
* Test 12 (Log as Test16.py / Your Test14.py or Test16.py): Stack Machine V1 (Codebase was more like Test15.py design)
    * Task: 13 instructions, 2 data registers, 1 address register, 4 memory cells, 1 flag, stack for subroutines, conditional jump. Input: $(x1_t, \text{instr}_{oh}, R0, R1, RA_{oh}, F0, \text{DataMem}, SP_{oh}, y_{t-1})$.
    * Model: 4909 parameters (Input=27, Units=64, SharedG=13).
    * Result: 100% test accuracy, target (65%) reached in epoch 1 (tested for 5 epochs).
    * Significance: Mastered memory access via address register, stack operations for simulated subroutines, and conditional control flow. Clear approximation of "Mini-CPU" functionality.
* Test 13 (Log as Test17.py / Your Test19.py): FPLA V1 / CPU Core V0.1
    * Task: 16 instructions, 4 data registers, address register, 8 memory cells, 2 flags, stack, more complex ALU and jump logic. Input: $(x1..x6, \text{instr}_{oh}, R0-3, AR_{oh}, ZF, EQF, \text{DataMem}, SP_{oh}, y_{t-1})$.
    * Model: 8801 parameters (Input=33, Units=96, SharedG=16).
    * Result: 100% test accuracy, target (55%) reached in epoch 1 (tested for 5 epochs).
    * Significance: The model demonstrated the ability to handle an even more complex instruction set, more registers, larger memory, multiple flags, and more complex addressing and jump logic. This is an impressive step towards a learnable, programmable logic unit.
* **Test20 ("CPU Core V0.1" / "FPLA V1"): Simulation of a Programmable 1-Bit CPU Core**
    * Task: Simulation of a rudimentary microcontroller core with 1-bit registers (R0-R3, AR), memory (8x1 bit), flags (ZF, EQF), stack (4 levels for 6-bit PC values), and an instruction set of 18 instructions (including load/store, ALU ops, conditional jumps, CALL/RET).
    * Input: 51 features (control bits, instruction OH, registers, AR OH, flags, memory, stack pointer OH, $y_{t-1}$).
    * Model: 18069 parameters (Input=51, DPP Units=128, SharedG=25).
    * Result: 100% test accuracy, target (50%) reached in epoch 1. Successful execution of a specific test program (based on Test21/Test22 logic).
    * Significance: Demonstrated the ability to learn extremely complex, program-like logic with interacting components and an extensive instruction set.
* **Test24: Simulation of a Simplified 8-Bit CPU**
    * Task: Simulation of a CPU with 8-bit data width (2 general-purpose registers, 4x8-bit memory cells), 2 flags (Z, C), and a reduced instruction set of 8 instructions.
    * Input: 75 features.
    * Model: ~23k parameters (DPP Units=64, SharedG=18).
    * Result: 100% accuracy in training and testing on random data. Successful execution of a test program with 100% accuracy.
    * Significance: Showcased the architecture's scalability to 8-bit data width.
* **Test26 (based on Test25 setup): Simulation of a Simplified 32-Bit CPU**
    * Task: Similar to the 8-bit CPU but with 32-bit data width for registers, memory, and ALU operations (2 general-purpose registers, 4x32-bit memory cells, 2 flags (Z, C), 8 instructions).
    * Input: 187 features.
    * Model (as per user's successful log for Test26): `dpp_units=96`, `shared_g_dim=18` (~41.4k parameters).
    * Result: 100% accuracy in training and testing on random data (after 4 epochs of training, early stopping after 12 out of 100 epochs). Successful execution of an 11-step test program with 100% accuracy.
    * Significance: This success is a crucial milestone, demonstrating that the architecture can handle not only complex 1-bit logic (FPLA V1) or 8-bit logic but also scales effectively to a 32-bit data width, learning to simulate the corresponding operations precisely, at least in a simplified configuration.

## 4. Summary of Key Properties (based on tests)

* **High Parameter Efficiency for Complex Logic:** The model can learn sophisticated, rule-based, and state-dependent tasks with a remarkably low number of parameters.
* **Fast Convergence:** For the tested logic tasks, the model learns extremely quickly, often achieving very high or perfect accuracy within a few epochs.
* **Effective Contextualization and State Management:** The `DPPLayer_SharedG` can effectively use inputs (containing data, context bits, and previous states) to dynamically adapt its internal processing.
* **Capability for "Algorithmic Inference":** The model learns not just patterns but the underlying rules and procedures of a task, allowing it to execute quasi-programs.
* **Robustness of the Gating Mechanism:** The alpha values indicate that the gating mechanism actively responds to different inputs and internal states to control processing correctly.
* **Scalability with Data Width:** Successful simulations of 1-bit, 8-bit, and simplified 32-bit CPUs demonstrate that the fundamental architecture can handle wider data paths and their corresponding operations.
* **Ability to Simulate Highly Complex Procedural and Programmatic Logic:** Crowned by the success in the "FPLA V1" test (1-bit CPU core) and further extended by the successful simulation of a simplified 32-bit CPU (Test26), where the model mastered the processing of 32-bit wide data paths involving arithmetic, logical, and memory operations.

## 5. Conclusion and Future Outlook

The `DPPModelBase` with the `DPPLayer_SharedG` has proven to be an exceptionally powerful and efficient architecture for learning and executing complex, state-dependent, and program-like logic tasks. It often exceeds expectations for models with a comparably low parameter count in these domains. The successful handling of the "FPLA V1" task (1-bit CPU) as well as the simulations of simplified 8-bit and 32-bit CPUs (especially Test26) shows that the limits of a single `DPPLayer_SharedG` for algorithmic tasks are remarkably high. The architecture is capable of learning the essence of small, programmable processors with noteworthy efficiency.

Potential next steps could include:

* **Verification of the 32-bit simulation through program execution:** Implement a `CPUSimulator_32Bit_Simplified` and run specific test programs on the model trained in Test26 (as demonstrated for previous CPU versions and confirmed successful by the user for Test26).
* Systematic investigation of scalability:
    * How does the model perform with a *more complete* 32-bit CPU (e.g., larger memory, more registers, more complex addressing modes, a richer instruction set including stack operations)?
    * Testing the limits with even wider data paths (e.g., 64-bit).
* Testing generalization capabilities on slightly modified, unseen instructions or program structures.
* Analyzing the impact of different `shared_g_dim` sizes and `dpp_units` counts on learning capability for such extreme tasks.
* Comparison with specialized architectures for program synthesis or algorithmic learning.
* Exploring the use of stacked `DPPLayer_SharedG` layers to potentially learn hierarchical abstractions of program logic or even more complex algorithms if a single layer reaches its limits.
* Deployment as a modular component in larger AI systems that require an explicit, learnable logic component.

The results to date strongly advocate for the potential of architectures with dynamic, internal path control for algorithmic learning.

## Code and Execution

The source code for the model implementations (`DPPLayer_SharedG`, `DPPModelBase`), data generators for the various CPU simulations, and training scripts are available in this repository.
* The core components `DPPLayer_SharedG` and `DPPModelBase` are defined within the respective `TestXX.py` files.
* Examples of CPU simulations can be found in `Test20.py` (1-Bit FPLA), the logic of `Test24.py` (8-Bit CPU), and `Test26.py` (simplified 32-Bit CPU).

To run a test:
```bash
python TestXX.py