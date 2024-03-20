# **Einfache Lineare Regression mit Gradientenabstieg von Grund auf**
Dieses Projekt zeigt die Umsetzung der einfachen linearen Regression mit dem Gradientenabstiegverfahren in Python. Die Implementierung erfolgt eigenständig mit eigenen Funktionen.

### **Übersicht**
In diesem Projekt wird der iterative Prozess des Gradientenabstiegs vollständig auf Codeebene erläutert und angewendet. Das Hauptziel besteht darin, das Konzept des Gradientenabstiegs zu verstehen und umzusetzen.

### **Kostenfunktion MSE**
Die Kostenfunktion berechnet den mittleren quadratischen Fehler (MSE) für eine Reihe von Beobachtungen. Diese Funktion wird verwendet, um den Fehler zwischen den geschätzten und den tatsächlichen Werten zu quantifizieren.

### **Gewichtsaktualisierung**
Die Funktion zur Gewichtsaktualisierung verwendet die Ableitungen der Kostenfunktion, um die Gewichte iterativ anzupassen. Dies geschieht durch Multiplikation mit einer Lernrate, um den Gradienten zu modifizieren.

### **Trainingsfunktion**
Die Trainingsfunktion führt den gesamten Prozess durch, indem sie die Kostenfunktion und die Gewichtsaktualisierung für eine bestimmte Anzahl von Iterationen anwendet.

### **Verwendung**
Um das Modell zu trainieren, müssen die Hyperparameter wie Lernrate, Startgewichte und Iterationsanzahl festgelegt werden. Anschließend kann die Trainingsfunktion aufgerufen werden, um den Lernprozess zu starten.

### **Hinweis**
Diese Implementierung betrifft nur die einfache lineare Regression mit einer unabhängigen Variablen. Bei Anwendungen mit mehreren Variablen kann eine Anpassung der Implementierung erforderlich sein.
