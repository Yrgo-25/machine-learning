# Bilaga B - Övningsuppgift: Dense-lager (del I)
Ni ska bygga vidare på er `ml`-kodbas från **L06–L07** genom att implementera en skarp underklass `Dense`, som ersätter den tillfälliga stubbklassen `ml::dense_layer::Stub`. Se [bilaga A](./a_architecture.md) för en genomgång av dense-lagrets arkitektur innan ni börjar.

---

### 1. Kom igång
Har ni inte redan hämtat testramverket under en tidigare lektion, gör det först. Kör följande
kommando en gång, i repots rotkatalog:

```bash
git submodule update --init --recursive
```

Katalogen [`exercises`](../exercises) innehåller lösningsförslaget från **L06–L07**, samt en
testsvit i `exercises/test` (se avsnitt 11):
* `include/ml/dense_layer/`: dense-lagrets interface samt stubbklassen `Stub` (**L06**).
* `include/ml/neural_network/` samt `source/ml/neural_network/`: nätverkets interface samt klassen
  `Shallow` (**L07**).
* `include/ml/types.h`, `source/main.cpp` samt `Makefile`.

Där ligger även två tomma filer märkta `@todo`, som ni fyller i under lektionen:
* `include/ml/dense_layer/dense.h` (avsnitt 5–8).
* `source/ml/dense_layer/dense.cpp` (avsnitt 9).

Utöver dessa utökar ni `include/ml/types.h` med enumerationsklassen `ActFunc` (avsnitt 2).

Skriv er kod där, eller i er befintliga `ml`-kodbas. Är er egen kod från **L06–L07** inte färdig,
utgå från lösningsförslaget, så att ni kan lägga hela lektionen på dense-lagret.

---

### 2. Enumerationsklassen `ActFunc`
I `ml/types.h`, definiera en enumerationsklass döpt `ActFunc` i namnrymden `ml`, dvs. bredvid matrisaliasen, så att den kan refereras till som `ml::ActFunc`. Denna enumerationsklass ska kunna användas för att välja aktiveringsfunktion för ett givet dense-lager. Följande enumeratorer ska läggas till:
* `Relu`: För ReLU (`Rectified Linear Unit`), som för en given input x returnerar x om x > 0, annars 0.
* `Tanh`: För hyperbolisk tangent, som ger utdata i intervallet (-1, 1).
* `None`: Ingen aktiveringsfunktion, dvs. nodens utdata är den viktade summan oförändrad (linjär aktivering).

---

### 3. Headerfilen `dense.h`
Klassen `Dense` deklareras i `include/ml/dense_layer/dense.h`, som redan finns i `exercises`,
märkt `@todo`. Arbetar ni i er egen `ml`-kodbas, skapa filen där. Filen ska kunna inkluderas såsom
visas nedan:

```cpp
#include "ml/dense_layer/dense.h"
```

---

### 4. Källkodsfilen `dense.cpp`
Klassens metoder definieras i `source/ml/dense_layer/dense.cpp`, som redan finns i `exercises`,
märkt `@todo`. Makefilen i `exercises` kompilerar den redan.

**OBS!** Arbetar ni i er egen `ml`-kodbas, skapa filen där och glöm inte att lägga till den i er
makefil:

```makefile
# Source files.
SRC_FILES := source/main.cpp \
             source/ml/dense_layer/dense.cpp \
             source/ml/neural_network/shallow.cpp
```

---

### 5. Klassen `Dense` - deklaration
I headerfilen `ml/dense_layer/dense.h`, lägg till en klass döpt `Dense`, som ärver motsvarande interface, se filen `include/ml/dense_layer/interface.h`:
* Använd publikt arv och markera klassen `final` så att den inte kan ärvas vidare.
* Överlagra samtliga metoder från interfacet, inklusive destruktorn.

**Tips**: Kopiera allt innehåll i interfacet och klistra in i den nya filen. Anpassa sedan koden för den nya underklassen `Dense` (inga `virtual` eller `= 0`, använd `override` med mera).

---

### 6. Privata medlemsvariabler
Lägg till följande privata medlemsvariabler i klassen:
* `myOutput`: Vektor innehållande nodernas output (flyttal). Ska ha utrymme för ett värde per nod i lagret.
* `myPreActOutput`: Vektor innehållande nodernas viktade summa innan aktiveringsfunktionen
  har applicerats (flyttal). Ska ha utrymme för ett värde per nod i lagret. Används av `backpropagate()`
  (se **L09**) för att beräkna aktiveringsfunktionens derivata korrekt.
* `myError`: Vektor innehållande nodernas fel (flyttal). Ska ha utrymme för ett värde per nod i lagret.
* `myBias`: Vektor innehållande nodernas bias-värden (flyttal). Ska ha utrymme för ett värde per nod i lagret.
* `myWeights`: Tvådimensionell vektor innehållande nodernas vikter. Ska ha utrymme för en vikt per nod och ingång, dvs. antalet noder x antalet vikter per nod.
* `myActFunc`: Lagrets aktiveringsfunktion (av typen `ActFunc`).

Medlemsvariablerna läggs till före konstruktorn, så att ni vet exakt vad konstruktorn ska initiera.

---

### 7. Konstruktor
Skapa en konstruktor som ska kunna användas för att skapa ett dense-lager med godtyckliga dimensioner samt en valbar aktiveringsfunktion.
* **Tar emot:**
    * `nodeCount`: Antalet noder i lagret, osignerat heltal (`std::size_t`).
    * `weightCount`: Antalet vikter per nod i lagret, osignerat heltal (`std::size_t`).
    * `actFunc`: Aktiveringsfunktionen som ska användas (av typen `ActFunc`). Som default ska aktiveringsfunktionen ReLU (`ActFunc::Relu` i implementationen) användas.
* Ska markeras `explicit` samt `noexcept`.
* Om `nodeCount` eller `weightCount` är lika med 0 ska ett felmeddelande skrivas ut och programmet avslutas genom att anropa `std::terminate()` (se avsnitt 9).

---

### 8. Borttagna konstruktorer och operatorer
Radera följande:
* Defaultkonstruktorn.
* Kopieringskonstruktorn.
* Förflyttningskonstruktorn.
* Kopieringsoperatorn.
* Förflyttningsoperatorn.

---

### 9. Definition av metoder
Definiera samtliga metoder, konstruktorer med mera som ej är markerade `delete` eller `default` i filen `source/ml/dense_layer/dense.cpp`:

**Konstruktor:**
* I konstruktorn ska samtliga medlemsvariabler initieras:
    * `myOutput`, `myPreActOutput` samt `myError` ska innehålla `nodeCount` flyttal som är lika med 0.0 vid start.
    * `myBias` ska innehålla `nodeCount` flyttal, och `myWeights` `nodeCount` x `weightCount`
      flyttal, samtliga lika med 0.0 tills vidare. Under **L09** randomiseras de i stället mellan
      0.0 och 1.0, när hjälpfunktionerna för detta finns på plats.
    * `myActFunc` ska tilldelas angiven aktiveringsfunktion.
* Om `nodeCount` är lika med 0 ska felmeddelandet `Node count cannot be 0!` skrivas ut till
  standard error, och om `weightCount` är lika med 0 felmeddelandet `Weight count cannot be 0!`.
  Programmet ska i båda fallen avslutas genom att anropa `std::terminate()`, såsom visas nedan:

```cpp
#include <cstdio>
#include <exception>

if (0U == nodeCount)
{
    std::fprintf(stderr, "Node count cannot be 0!\n");
    std::terminate();
}
if (0U == weightCount)
{
    std::fprintf(stderr, "Weight count cannot be 0!\n");
    std::terminate();
}
```

**Tips:** Stubbens konstruktor i `include/ml/dense_layer/stub.h` innehåller exakt dessa kontroller,
så kopiera gärna dem och upplägget därifrån: medlemsinitieringslistan lämnar vektorerna tomma,
kontrollerna genomförs först, och först därefter tilldelas vektorerna sina storlekar via `resize()`.
Ett lager som skapas med 0 noder avslutar då programmet utan att ha allokerat någonting.

**Getters:**
* Implementera samtliga getters från interfacet. Följ beskrivningarna i `include/ml/dense_layer/interface.h`:
    * `nodeCount()` samt `weightCount()` ska returnera antalet noder respektive vikter per nod i lagret.
    * Getter-metoder såsom `output()`, `error()` samt `weights()` ska returnera referenser till motsvarande medlemsvariabler.

**Beräkningsmetoder:**
Själva beräkningarna i `feedforward()`, `backpropagate()` (två varianter) samt `optimize()`
implementeras under **L09**. Tills vidare ska de fyra metoderna enbart kontrollera sina argument.
Vid ett ogiltigt argument skrivs ett felmeddelande ut till standard error, varpå metoden returnerar
`false`; annars returneras `true` utan att någonting beräknas:
* `feedforward(input)`: Kontrollera att `input.size() == weightCount()`. Annars: skriv ut
  `"Input dimension mismatch: expected X, actual: Y!"`.
* `backpropagate(reference)`: Kontrollera att `reference.size() == nodeCount()`. Annars: skriv ut
  `"Output dimension mismatch: expected X, actual: Y!"`.
* `backpropagate(nextLayer)`: Kontrollera att `nextLayer.weightCount() == nodeCount()`. Annars:
  skriv ut `"Layer dimension mismatch: expected X, actual: Y!"`.
* `optimize(input, learningRate)`:
    1. Kontrollera att lärhastigheten ligger inom intervallet `(0.0, 1.0)`, dvs.
       `(0.0 < learningRate) && (1.0 > learningRate)`. Annars: skriv ut
       `"Invalid learning rate X!"`.
    2. Kontrollera att `input.size() == weightCount()`. Annars: skriv ut
       `"Input dimension mismatch: expected X, actual: Y!"`.

`X` är det förväntade värdet och `Y` det faktiska; i `"Invalid learning rate X!"` är `X` den givna
lärhastigheten. Så här kan `feedforward()` se ut:

```cpp
bool Dense::feedforward(const Matrix1d& input) noexcept
{
    if (input.size() != weightCount())
    {
        std::fprintf(stderr, "Input dimension mismatch: expected %zu, actual: %zu!\n",
                     weightCount(), input.size());
        return false;
    }
    return true;
}
```

Villkoren är desamma som stubbens i `include/ml/dense_layer/stub.h`, så kopiera gärna dem
därifrån, men utan feedforward-räknaren, som enbart finns för att göra nätverket testbart. Under
**L09** behålls kontrollerna oförändrade, och beräkningarna läggs till efter dem.

---

### 10. Kontrollpunkt: testprogrammet
Testprogrammet i `source/main.cpp` är redan skrivet. Det gör samma sak som i **L07**, men med två
dense-lager i stället för två stubbar:
1. Skapar det dolda lagret med 3 noder och 2 vikter per nod, med ReLU som aktiveringsfunktion
   (default), samt utgångslagret med 1 nod och 3 vikter per nod, med `ActFunc::Tanh`.
2. Skapar ett `ml::neural_network::Shallow`-nätverk av de två lagren, med ett 2-bitars
   XOR-mönster som träningsdata.
3. Predikterar för varje träningsuppsättnings indata, tränar nätverket i 100 epoker med
   lärhastigheten `0.01`, och predikterar sedan igen.

Kompilera och testkör programmet via kommandot `make`. Utskriften ska se ut så här:

```
--------------------------------------------------------------------------------
Predictions before training:
Input: 0 0, predicted output: 0
Input: 0 1, predicted output: 0
Input: 1 0, predicted output: 0
Input: 1 1, predicted output: 0
--------------------------------------------------------------------------------
Predictions after training:
Input: 0 0, predicted output: 0
Input: 0 1, predicted output: 0
Input: 1 0, predicted output: 0
Input: 1 1, predicted output: 0
--------------------------------------------------------------------------------
```

Notera följande i utskriften:
* **Samtliga prediktioner är 0, både före och efter träningen.** Detta är förväntat, eftersom
  lagren inte beräknar någonting ännu: `feedforward()` kontrollerar enbart sin indata, så
  utdatan behåller sitt startvärde 0.0. Prediktionerna börjar ändras först under **L09**, när
  beräkningarna implementeras och bias samt vikter randomiseras.
* **Utskriften saknar felmeddelanden och `Training failed!`.** Samtliga argumentkontroller har
  alltså godkänt samtliga anrop under träningens 400 steg, vilket visar att lagren är rätt
  kopplade till varandra och till träningsdatan.

Skriver programmet ut `Training failed!`, har någon av argumentkontrollerna avvisat ett korrekt
anrop. Felmeddelandet som skrivs ut precis innan anger vilken kontroll det gäller.

---

### 11. Enhetstester
Kontrollera er implementation mot testsviten i `exercises/test`. Se testsvitens
[README](../exercises/test/README.md) för detaljer. Testsviten innehåller även testerna från
**L06–L07**, så det räcker att köra denna.

1. Bygg och kör testsviten via kommandot `make` i katalogen `exercises/test`. Testramverket måste
   ha hämtats först, se avsnitt 1.
    * Skriver ni er kod i er egen `ml`-kodbas i stället för i `exercises`, ange sökvägen till den
      via `make ML_DIR=<sökväg till er ml-katalog>`.
2. Åtgärda eventuella fel och kör testsviten igen, tills samtliga testfall går igenom.

Testsviten kompilerar inte förrän `ml::ActFunc` finns i `ml/types.h` (avsnitt 2), och klassen
`Dense` är deklarerad i `dense.h` samt definierad i `dense.cpp` med samtliga metoder som testerna
anropar. Innan dess är det första kompileringsfelet
`‘ActFunc’ in namespace ‘ml’ does not name a type`. Läs alltid det första kompileringsfelet; det
anger oftast vilken metod som saknas eller har fel signatur.

Argumentkontrollerna i `feedforward()`, `backpropagate()` och `optimize()` testas redan nu, medan
själva beräkningarna testas först under **L09**. Felmeddelanden som `Input dimension mismatch` och
`Invalid learning rate` skrivs ut av de testfall som medvetet anropar lagret med felaktiga
argument. De betyder alltså inte att något test har misslyckats. Konstruktorns anrop till `std::terminate()` testas inte heller,
eftersom det avslutar hela testprogrammet. Kontrollera det för hand genom att skapa ett lager med
0 noder.

---
