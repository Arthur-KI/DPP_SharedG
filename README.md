Dynamisch Pfad-gesteuertes Perzeptron mit Geteiltem Gating (DPP_SharedG)
English:

The Dynamic Path-controlled Perceptron with Shared Gating (DPP_SharedG) is a novel neural network layer architecture designed for high parameter efficiency and robustness in tackling complex, conditional, and stateful tasks. The core idea is to equip the layer with an internal, learnable, data-driven logic that allows it to dynamically choose or combine different processing paths for incoming data.

Architecture and Functionality:
The DPPLayer_SharedG typically consists of:

    Two independent linear processing paths (Path A and Path B): Each path applies a linear transformation to the input.
    A Gating Path with a partially shared mechanism:
        A shared linear transformation projects the input to a lower-dimensional shared context representation (xshared_g​).
        Unit-specific linear transformations then process this shared representation to compute individual gating logits for each output unit of the layer.
        These logits are passed through a sigmoid function to produce gating coefficients (α) between 0 and 1.
    Weighted Combination: The final output of each unit in the DPPLayer_SharedG is a dynamic blend of the outputs from Path A and Path B, weighted by the gating coefficients: zfinal​=α⋅zA​+(1−α)⋅zB​.

The overall model (DPPModelBase) typically uses a single DPPLayer_SharedG followed by a ReLU activation and a final linear output layer.

Learning Performance and Speed:
The DPP_SharedG architecture has demonstrated exceptional performance and efficiency in learning and executing complex, state-dependent, and program-like logic tasks.

    High Parameter Efficiency: The model can solve sophisticated, rule-based, and stateful tasks with a remarkably small number of parameters compared to what might be expected from traditional architectures. For instance, it successfully simulated a 1-bit CPU core ("FPLA V1") with ~18k parameters, an 8-bit CPU with ~23k parameters, and a simplified 32-bit CPU with ~28.7k parameters.
    Rapid Convergence: For the logic tasks tested, the model learns extremely quickly, often achieving very high or perfect accuracy within just a few epochs. For example, the FPLA V1 task (1-bit CPU) reached its target accuracy in the first epoch, and the 32-bit CPU simulation also converged rapidly (target accuracy in 4 epochs).
    Effective Contextualization and State Management: The DPPLayer_SharedG effectively utilizes inputs (containing data, context bits, and previous states) to dynamically adjust its internal processing.
    Algorithmic Inference: The model demonstrates an ability to learn underlying rules and procedures, allowing it to execute small "programs". This was particularly evident in its successful simulation of various CPU architectures, including handling instruction sets, register states, memory access, flags, and control flow (jumps, subroutines).
    Robust Gating Mechanism: The alpha gating values actively respond to different inputs and internal states, guiding the correct processing path.
    Scalability with Data Width: The successful simulation of 8-bit and 32-bit CPUs (Tests 24 & 26) indicates that the architecture can handle wider data paths effectively.

In summary, the DPP_SharedG has proven to be a powerful and efficient architecture for tasks requiring learned procedural logic and stateful computation, often outperforming expectations for models with a comparable parameter count.


Deutsch:

Das Dynamisch Pfad-gesteuerte Perzeptron mit Geteiltem Gating (DPP_SharedG) ist eine neuartige Neuronenschicht-Architektur, die für hohe Parametereffizienz und Robustheit bei der Bewältigung komplexer, bedingter und zustandsbehafteter Aufgaben entwickelt wurde. Die Kernidee besteht darin, die Schicht mit einer internen, erlernbaren und datengesteuerten Logik auszustatten, die es ihr ermöglicht, dynamisch verschiedene Verarbeitungspfade für eingehende Daten auszuwählen oder zu kombinieren.

Architektur und Funktionsweise:
Der DPPLayer_SharedG besteht typischerweise aus:

    Zwei unabhängigen linearen Verarbeitungspfaden (Pfad A und Pfad B): Jeder Pfad wendet eine lineare Transformation auf die Eingabe an.
    Einem Gating-Pfad mit einem partiell geteilten Mechanismus:
        Eine geteilte lineare Transformation projiziert die Eingabe auf eine niedrigdimensionale, geteilte Kontextrepräsentation (xshared_g​).
        Einheitenspezifische lineare Transformationen verarbeiten dann diese geteilte Repräsentation, um individuelle Gating-Logits für jede Ausgabeeinheit der Schicht zu berechnen.
        Diese Logits werden durch eine Sigmoid-Funktion geleitet, um Gating-Koeffizienten (α) zwischen 0 und 1 zu erzeugen.
    Gewichtete Kombination: Die endgültige Ausgabe jeder Einheit im DPPLayer_SharedG ist eine dynamische Mischung der Ausgaben von Pfad A und Pfad B, gewichtet durch die Gating-Koeffizienten: zfinal​=α⋅zA​+(1−α)⋅zB​.

Das Gesamtmodell (DPPModelBase) verwendet typischerweise einen einzelnen DPPLayer_SharedG, gefolgt von einer ReLU-Aktivierung und einer finalen linearen Ausgabeschicht.

Lernleistung und Geschwindigkeit:
Die DPP_SharedG-Architektur hat eine außergewöhnliche Leistungsfähigkeit und Effizienz beim Erlernen und Ausführen komplexer, zustandsabhängiger und programmartiger Logikaufgaben gezeigt.

    Hohe Parametereffizienz: Das Modell kann anspruchsvolle, regelbasierte und zustandsabhängige Aufgaben mit einer bemerkenswert geringen Anzahl von Parametern lösen, verglichen mit dem, was man von traditionellen Architekturen erwarten würde. Beispielsweise simulierte es erfolgreich einen 1-Bit-CPU-Kern ("FPLA V1") mit ca. 18k Parametern, eine 8-Bit-CPU mit ca. 23k Parametern und eine vereinfachte 32-Bit-CPU mit ca. 28.7k Parametern.
    Schnelle Konvergenz: Bei den getesteten Logikaufgaben lernt das Modell extrem schnell und erreicht oft schon nach wenigen Epochen eine sehr hohe oder perfekte Genauigkeit. Die "FPLA V1"-Aufgabe (1-Bit-CPU) erreichte beispielsweise ihre Zielgenauigkeit in der ersten Epoche, und die 32-Bit-CPU-Simulation konvergierte ebenfalls sehr schnell (Zielgenauigkeit in 4 Epochen).
    Effektive Kontextualisierung und Zustandsverwaltung: Der DPPLayer_SharedG nutzt Eingaben (die Daten, Kontextbits und vorherige Zustände enthalten) effektiv, um seine interne Verarbeitung dynamisch anzupassen.
    Algorithmische Inferenz: Das Modell demonstriert die Fähigkeit, zugrundeliegende Regeln und Prozeduren zu lernen, was ihm erlaubt, quasi kleine "Programme" auszuführen. Dies wurde besonders deutlich bei der erfolgreichen Simulation verschiedener CPU-Architekturen, einschließlich der Handhabung von Befehlssätzen, Registerzuständen, Speicherzugriff, Flags und Kontrollfluss (Sprüngen, Subroutinen).
    Robuster Gating-Mechanismus: Die Alpha-Gating-Werte reagieren aktiv auf unterschiedliche Eingaben und interne Zustände und steuern so den korrekten Verarbeitungspfad.
    Skalierbarkeit bezüglich der Datenbreite: Die erfolgreiche Simulation von 8-Bit- und 32-Bit-CPUs (Tests 24 & 26) zeigt, dass die Architektur auch mit breiteren Datenwegen effektiv umgehen kann.

Zusammenfassend hat sich DPP_SharedG als eine leistungsstarke und effiziente Architektur für Aufgaben erwiesen, die erlernte prozedurale Logik und zustandsbehaftete Berechnungen erfordern, und übertrifft dabei oft die Erwartungen an Modelle mit vergleichbarer Parameteranzahl.
