Dynamisch Pfad-gesteuertes Perzeptron mit Geteiltem Gating (DPP_SharedG)
1. Einleitung & Motivation

Dieses Projekt stellt eine neuartige neuronale Schichtarchitektur vor: das Dynamisch Pfad-gesteuerte Perzeptron (DPP) mit geteiltem Gating-Pfad (DPP_SharedG). Ziel war es, eine Architektur zu entwickeln, die mit sehr wenigen Parametern komplexe, zustandsabhängige und prozedurale Logikaufgaben lernen kann – Aufgaben, die klassischerweise mit klassischen Algorithmen oder sehr großen neuronalen Netzen gelöst werden. Das DPP_SharedG-Modell zeigt, dass schon eine einzige spezialisierte Schicht in einem ansonsten einfachen Netzwerk erstaunlich leistungsfähig sein kann.
2. Architektur und Funktionsweise

Das Kernmodell (DPPModelBase) besteht aus drei Hauptkomponenten:

    Einem einzelnen DPPLayer_SharedG (Hauptinnovationspunkt)
    Einer ReLU-Aktivierungsfunktion
    Einer linearen Ausgabeschicht

Der DPPLayer_SharedG kombiniert zwei parallele Verarbeitungspfade ("A" und "B") für jede Ausgabeeinheit. Ein Gating-Mechanismus – bestehend aus einem gemeinsam genutzten und einem unitspezifischen Teil – berechnet für jede Einheit ein Alpha, das steuert, wie stark das Ergebnis von Pfad A bzw. B gewichtet wird. So lernt das Modell dynamisch, je nach Eingabe, unterschiedliche Rechenwege zu nutzen.

Das Modell erhält pro Zeitschritt einen Vektor, der aktuelle Daten, Kontextbits und vergangene Zustände enthalten kann. Die Architektur ist darauf ausgelegt, interne Zustände, Verzweigungen und sogar einfache Programm-Logik zu modellieren.
3. Zentrale Testergebnisse und Highlights (ab Test7.py)

Das Modell wurde auf einer Reihe von zunehmend komplexen Aufgaben getestet, darunter:

    Test 7: Bedingtes Akkumulieren – Das Modell lernt, anhand eines Kontrollbits zwischen Addition und Übernahme des Vorgängerwertes zu wählen. 100% Genauigkeit bei minimalen Parametern.
    Test 8: Zustandsgesteuerte Operation – Die Ausgabe hängt davon ab, ob der vorherige Output 0 oder 1 war (konditionale XOR/Identitäts-Operation). Ebenfalls 100% Testgenauigkeit.
    Test 9: Zähler mit Reset und Moduswahl – Das Modell verwaltet einen internen Zähler (modulo 4), kann diesen zurücksetzen und nutzt ihn, um zwischen zwei Operationen zu wählen. Auch hier wird perfekte Genauigkeit erreicht.
    Test 11: Mini-Interpreter für 8 Instruktionen (mit zwei Registern) – Das Modell kann einen einfachen Maschinenbefehlssatz korrekt interpretieren und Registerzustände verwalten, was erste CPU-ähnliche Fähigkeiten demonstriert.
    Test 12: Stackmaschine – Mit Stack für Subroutinen, bedingtem Sprung und Speicherzugriff. Das Modell meistert anspruchsvolle prozedurale Logik mit weiterhin sehr wenigen Parametern.
    Test 13 / FPLA V1: Simulation eines Mikrocontroller-Kerns mit mehreren Registern, Speicher, Stack, Flags und 18 Instruktionen. Das Modell zeigt, dass es auch diese hohe Komplexität mit nur einer Schicht und ca. 18.000 Parametern vollständig abbilden kann.
    Test 24: 8-Bit CPU – Das Modell simuliert erfolgreich eine einfache 8-Bit-CPU mit komplexem Verhalten (Register, Speicher, ALU, Sprünge).
    Test 26: 32-Bit CPU – Auch mit 32-Bit breiten Datenwegen kann das Modell erfolgreich arithmetische und logische Operationen sowie Speicherzugriffe abbilden.

Bemerkenswert ist, dass das Modell in allen Tests eine sehr schnelle Konvergenz zeigt und oft schon nach wenigen Epochen perfekte Genauigkeit erreicht.
4. Eigenschaften und Vorteile

    Parametereffizienz: Selbst sehr komplexe Aufgaben werden mit wenigen Parametern gemeistert.
    Schnelles Lernen: Hohe Genauigkeit wird oft schon nach wenigen Trainings-Epochen erreicht.
    Flexible Logik: Über den Gating-Mechanismus kann das Modell dynamisch zwischen verschiedenen Rechenwegen wählen.
    Robustheit & Skalierbarkeit: Die Architektur funktioniert für Aufgaben von einfachen Flip-Flops bis hin zu komplexen CPU-Kernen und lässt sich auf größere Datenbreiten übertragen.
    Algorithmisches Lernen: Das Modell lernt nicht nur Input-Output-Muster, sondern tatsächlich logische Regeln und Prozesse.

5. Anwendungsmöglichkeiten

Das DPP_SharedG-Modell eignet sich besonders für:

    Simulation und Nachbildung klassischer Logik (z. B. CPUs, Automaten, Steuerungen)
    Forschung zu "Neural Algorithmic Reasoning" und differenzierbarem Programmieren
    Didaktische Zwecke: Zeigen, wie neuronale Netze komplexe Logik und sogar kleine Programme lernen können
    Embedding in Hardware-nahe oder ressourcenbeschränkte Systeme, wo Parametereffizienz zählt
    Grundbaustein für KI-Systeme, die explizite, lernbare Logik-Komponenten benötigen

Fazit:
Mit DPP_SharedG steht eine hocheffiziente und mächtige Architektur zur Verfügung, die zeigt, dass komplexe, zustandsabhängige und programmatische Logikaufgaben auch mit kleinen neuronalen Netzen lösbar sind. Die offene Lizenz lädt dazu ein, die Architektur weiter zu erforschen, zu adaptieren und für neue Aufgabenbereiche auszuprobieren.
