# Bilaga B - Övningsuppgift: Dense-lager (del I)
Ni ska bygga vidare på er `ml`-kodbas från **L06-L07** genom att implementera en skarp underklass `Dense`, som ersätter den tillfälliga stubbklassen `ml::dense_layer::Stub`. Se [bilaga A](./a_architecture.md) för en genomgång av dense-lagrets arkitektur innan ni börjar.

---

### 1. Enumerationsklassen `ActFunc`
I `ml/types.h`, definiera en enumerationsklass döpt `ActFunc`. Denna enumerationsklass ska kunna användas för att välja aktiveringsfunktion för ett givet dense-lager. Följande enumeratorer ska läggas till:
* `Relu`: För ReLU (`Rectified Linear Unit`), som för en given input x returnerar x om x > 0 annars 0.
* `Tanh`: För hyperbolisk tangent, som ger utdata i intervallet [-1, 1].

---

### 2. Skapa `dense.h`
Skapa en ny fil döpt `dense.h` i `include/ml/dense_layer`. Filsökvägen ska alltså vara `include/ml/dense_layer/dense.h` och filen ska kunna inkluderas såsom visas nedan:

```cpp
#include "ml/dense_layer/dense.h"
```

---

### 3. Skapa `dense.cpp`
Skapa en ny fil döpt `dense.cpp` i `source/dense_layer`. Filsökvägen ska alltså vara `source/dense_layer/dense.cpp`.

**OBS! Glöm inte att lägga till denna filen i din makefil, se nedan!**

```makefile
# Application target.
TARGET := app

# Source files.
SOURCE_FILES := source/dense_layer/dense.cpp # Här läggs filen till i bygget.
                source/dense_layer/stub.cpp\
                source/main.cpp\
				source/neural_network/shallow.cpp\

# Include directory.
INCLUDE_DIR := include

# Compiler flags.
COMPILER_FLAGS := -Wall -Werror -std=c++17

# Build and run the application as default.
default: build run

# Build the application.
build:
	@g++ $(SOURCE_FILES) -o $(TARGET) -I$(INCLUDE_DIR) $(COMPILER_FLAGS)

# Run the application.
run:
	@./$(TARGET)

# Clean the application.
clean:
	@rm -f $(TARGET)
```

---

### 4. Klassen `Dense` - deklaration
I headerfilen `ml/dense_layer/dense.h`, lägg till klass döpt `Dense`, som ärver motsvarande interface, se filen `include/ml/dense_layer/interface.h`:
* Använd publikt arv och markera klassen `final` så att den inte kan ärvas vidare.
* Överlagra samtliga metoder från interfacet, inklusive destruktorn.

**Tips**: Kopiera allt innehåll i interfacet och klistra in i den nya filen. Anpassa sedan koden för de nya underklassen `Dense` (inga `virtual` eller `= 0`, använd `override` med mera).

---

### 5. Privata medlemsvariabler
Lägg till följande privata medlemsvariabler i klassen:
* `myOutput`: Vektor innehållande nodernas output (flyttal). Ska ha utrymme för ett värde per nod i lagret.
* `myPreActivationOutput`: Vektor innehållande nodernas viktade summa innan aktiveringsfunktionen
  har applicerats (flyttal). Ska ha utrymme för ett värde per nod i lagret. Används av `backpropagate()`
  (se **L09**) för att beräkna aktiveringsfunktionens derivata korrekt.
* `myError`: Vektor innehållande nodernas fel (flyttal). Ska ha utrymme för ett värde per nod i lagret.
* `myBias`: Vektorn innehållande nodernas bias-värden (flyttal). Ska ha utrymme för ett värde per nod i lagret.
* `myWeights`: Tvådimensionell vektor innehållande nodernas vikter. Ska ha utrymme för en vikt per nod och ingång, dvs. antalet noder x antalet vikter per nod.
* `myActFunc`: Lagrets aktiveringsfunktion (av typen `ActFunc`).

Medlemsvariablerna läggs till före konstruktorn, så att ni vet exakt vad konstruktorn ska initiera.

---

### 6. Konstruktor
Skapa en konstruktor som ska kunna användas för att skapa ett dense-lager med godtyckliga dimensioner samt en valbar aktiveringsfunktion.
* **Tar emot:**
    * `nodeCount`: Antalet noder i lagret (osignerat heltal).
    * `weightCount`: Antalet vikter per nod i lagret (osignerat heltal).
    * `actFunc`: Aktiveringsfunktionen som ska användas (av typen `ActFunc`). Som default ska `ReLU` användas.
* Ska markeras `explicit` samt `noexcept`.
* Om `nodeCount` eller `weightCount` är lika med 0 ska ett felmeddelande skrivas ut och programmet avslutas genom att anropa `std::terminate()`.

---

### 7. Borttagna konstruktorer och operatorer
Radera följande:
* Defaultkonstruktorn.
* Kopieringskonstruktorn.
* Förflyttningskonstruktorn.
* Kopieringsoperatorn.
* Förflyttningsoperatorn.

---

### 8. Definition av metoder
Definiera samtliga metoder, konstruktorer med mera som ej är markerade `delete` eller `default` i filen `source/dense_layer/dense.cpp`:

**Konstruktor:**
* I konstruktorn ska samtliga medlemsvariabler initieras:
    * `myOutput`, `myPreActivationOutput` samt `myError` ska innehålla `nodeCount` flyttal som är lika med 0.0 vid start.
    * `myBias` ska innehålla `nodeCount` flyttal randomiserade mellan 0.0 - 1.0.
    * `myWeights` ska innehålla `nodeCount` x `weightCount` flyttal randomiserade mellan 0.0 - 1.0.
    * `myActFunc` ska tilldelas angiven aktiveringsfunktion.
* Om `nodeCount` eller `weightCount` är lika med 0 ska felmeddelandet `Invalid dense layer parameters: nodeCount and weightCount must be greater than 0!` skrivas ut till standard error, varpå programmet ska avslutas genom att anropa `std::terminate()`, såsom visas nedan:

```cpp
#include <cstdio>
#include <exception>

if ((0U == nodeCount) || (0U == weightCount))
{
    std::fprintf(stderr, "Invalid dense layer parameters: "
        "nodeCount and weightCount must be greater than 0!\n");
    std::terminate();
}
```

**Övriga metoder:**
* Implementera samtliga metoder från interfacet förutom `feedforward()`, `backpropagate()` samt `optimize()`.
* Följ beskrivningarna i `include/ml/dense_layer/interface.h`:
    * Metoder som `nodeCount()`, `weightCount()` ska returnera antalet noder respektive vikter per nod i lagret.
    * Getter-metoder såsom `output()` samt `error()` ska returnera referenser till motsvarande medlemsvariabler.
    * De tre ej implementerade metoderna returnerar `bool` (se **L06**), så en tom funktionskropp kompilerar inte. Ge var och en av dem en platshållarkropp bestående av `return false;` tills ni implementerar den under **L09**. `false` i stället för `true`, så att en metod ni glömmer att färdigställa rapporterar att den misslyckades i stället för att tyst påstå att allt gick bra.

---
