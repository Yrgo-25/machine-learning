# Hemtentamen 2 - Konvolutionella neurala nätverk och testning

## Information
Denna hemtentamen är gemensam för kurserna **Maskininlärning** och **Mjuk- och hårdvarutestning**:
Varje uppgift är märkt med vilken kurs den examinerar samt med **G** eller **VG**; betyget sätts
separat per kurs, se *Betygsättning* nedan.

### Hjälpmedel
* Kurslitteratur, egna anteckningar, föreläsningsmaterial samt internet.
* AI-verktyg får användas. Koden ska dock skrivas manuellt, även om ni hämtar kodsnuttar från ett
  AI-verktyg; skriv alltså in koden själva i stället för att klistra in den rakt av.
* Ni ska kunna förklara varje testfall ni lämnar in samt varje rättning ni gör.

### Betygsättning
Betyget avgörs helt av vilka uppgifter ni genomför:

| Kurs | **G** | **VG** |
|---|---|---|
| Maskininlärning | Uppgift 3 | Uppgift 3 och 5 |
| Mjuk- och hårdvarutestning | Uppgift 1 och 2 | Uppgift 1, 2 och 4 |

Ni väljer alltså själva nivå: gör G-uppgifterna för ett G, gör samtliga för ett VG. En uppgift
räknas som genomförd när den är klar och fungerar, dvs. testfallen går igenom, rättningarna är på
plats, och ni kan förklara dem vid redovisningen. Halvfärdiga uppgifter ger inget betyg; välj nivå
och gör den klar.

---

## Uppgiften i korthet
Katalogen [code](./code) innehåller ett komplett konvolutionellt neuralt nätverk: de
konvolutionella lagren `Conv` och `MaxPool`, flatten-lagret `Flatten`, dense-lagret `Dense`,
orkestreringen `Cnn` samt stubbar och en factory. Koden kompilerar, men **åtta buggar har medvetet
planterats**:
* **Två i `Conv`**, i [source/ml/conv_layer/conv.cpp](./code/source/ml/conv_layer/conv.cpp).
* **Två i `MaxPool`**, i
  [source/ml/conv_layer/max_pool.cpp](./code/source/ml/conv_layer/max_pool.cpp).
* **Två i `Flatten`**, i
  [source/ml/flatten_layer/flatten.cpp](./code/source/ml/flatten_layer/flatten.cpp).
* **Två i `Cnn`**, i [source/ml/cnn/cnn.cpp](./code/source/ml/cnn/cnn.cpp).

Er uppgift är att skriva tester för koden, hitta buggarna via de testfall som misslyckas, och rätta
dem:

1. Skriv vanliga testfall för lagrens beteende, ett per beräkning. Ett test av `feedforward()` i
   `Conv` räknar exempelvis ut den förväntade utdatan ur lagrets egna `kernel()` och `bias()`, och
   jämför mot det lagret faktiskt gav.
2. Kör testerna. De som misslyckas pekar ut var buggarna sitter.
3. Rätta buggen.
4. Kör testerna igen; nu ska de gå igenom.

Ni skriver alltså inga testfall som är särskilt konstruerade för en viss bugg, utan helt vanliga
testfall som kontrollerar att koden gör det den ska.

---

## Kom igång
Hämta testramverket, en gång, i repots rotkatalog:

```bash
git submodule update --init --recursive
```

Bygg och kör programmet i katalogen [code](./code):

```bash
make
```

Programmet tränar nätverket på fyra små bildmönster och predikterar sedan med vart och ett av dem.
**Just nu misslyckas träningen** och programmet skriver ut `Training failed!`; det är det första
av flera symptom, och tar er inte hela vägen. Bygg och kör testsviten i katalogen
[code/test](./code/test):

```bash
make
```

Testsviten innehåller åtta exempeltester som samtliga går igenom. De visar hur ramverket används
och avslöjar ingen av buggarna.

---

## Så ska lagren fungera
Buggarna går bara att hitta om ni vet vad koden borde göra. Följande gäller i denna kodbas:
* **`Conv`** använder zero-padding, så utdatan har samma storlek som indatan. Varje utdatavärde är
  `bias` plus summan av kernelns värden multiplicerade med motsvarande värden i den paddade
  indatan, varefter aktiveringsfunktionen appliceras.
* **`MaxPool`** använder icke-överlappande pooling, dvs. steglängden är lika med poolstorleken.
  Varje utdatavärde är det största värdet i sitt block. Vid backpropagation går gradienten enbart
  till den position i blocket som bidrog med maxvärdet.
* **`Flatten`** läser indatan radvis: element `[i][j]` hamnar på index `i * storlek + j` i
  utdatavektorn. Backpropagation gör samma sak baklänges.
* **`Cnn`** kedjar lagren: conv, pooling, flatten, dense. Kontrakten för `train()`, `inputSize()`
  och `outputSize()` står i [cnn.h](./code/include/ml/cnn/cnn.h).

### Viktigt
* **Ändra enbart det som är trasigt.** Följande är korrekt och ska lämnas orört: samtliga
  interface, samtliga stubbar, `Dense`, `factory`, `utils`, `random`, `main.cpp` samt makefilerna.
  Samtliga åtta buggar ligger i de fyra filer som räknas upp ovan.
* **`Conv` exponerar sina parametrar** via `kernel()` och `bias()`, så att ett förväntat värde kan
  räknas ut direkt ur lagret. `MaxPool` och `Flatten` har inga parametrar alls och är därmed helt
  förutsägbara.
* **Stubbarna är till för `Cnn`.** Med `factory::Stub` byggs nätverket av stubblager med känt
  beteende, så att ett misslyckat testfall bara kan bero på `Cnn` självt.
* Felmeddelanden som `Dimension mismatch` skrivs ut av de testfall som medvetet anropar med
  felaktiga argument. De betyder alltså inte att ett test har misslyckats.
* Koden behöver ej kommenteras.

---

## Uppgifter

### **1.** Unit-tester för lagren: beräkningarna (Testning, G)
Skriv testfall som kontrollerar att lagren räknar rätt, i katalogen [code/test](./code/test).
Räkna ut det förväntade värdet i testet, och jämför med vad lagret gav:
* **`Conv`:** `feedforward()` mot `bias()` plus kernelns bidrag, samt `backpropagate()` och
  `optimize()`, exempelvis genom att läsa av `bias()` före och efter optimeringen.
* **`MaxPool`:** `feedforward()` mot blockens maxvärden, och `backpropagate()` mot
  gradienterna på maxvärdenas positioner.
* **`Flatten`:** `feedforward()` och `backpropagate()` mot den radvisa ordningen.

Använd `act_func::Type::None` i `Conv` när ni vill jämföra mot den viktade summan utan filtrering,
och en indata där värdena skiljer sig åt, exempelvis en stigande talföljd. Symmetrisk indata döljer
flera av felen.

---

### **2.** Unit-tester för lagren: dimensionskontroller (Testning, G)
Samtliga lager ska avvisa indata och gradienter av fel storlek och returnera `false`, och returnera
`true` för rätt storlek. Exempeltesterna `Conv.FeedforwardChecksInputSize` samt
`Flatten.FeedforwardChecksInputSize` visar mönstret. Skriv motsvarande testfall för `backpropagate()`
i samtliga tre lager.

---

### **3.** Rätta buggarna i lagren (Maskininlärning, G)
Rätta de sex buggarna i lagren, så att testfallen från uppgift 1 och 2 går igenom.

---

### **4.** Komponenttester för `Cnn` (Testning, VG)
Skriv komponenttester för `ml::cnn::Cnn` med `factory::Stub`, så att testerna är oberoende av
lagrens matematik. Jämför mot kontraktet i [cnn.h](./code/include/ml/cnn/cnn.h), och täck bland
annat `train()` med ogiltiga argument samt `inputSize()` och `outputSize()`, även efter att ett
extra dense-lager har lagts till med `addDenseLayer()`.

---

### **5.** Rätta buggarna i `Cnn` (Maskininlärning, VG)
Rätta båda buggarna, så att testfallen från uppgift 4 går igenom.

---

## Redovisning
Lämna in:
* Testfallen i [code/test](./code/test).
* Den rättade koden i `conv.cpp`, `max_pool.cpp`, `flatten.cpp` samt `cnn.cpp`.

När samtliga åtta buggar är rättade ska testsviten gå igenom, och demoprogrammet ska träna utan
felmeddelande och prediktera rätt mönster.

### Muntlig genomgång
Uppgiften redovisas därefter muntligt, cirka fem minuter per student. Ni visar er egen kod och går
igenom den. Genomgången är ett villkor för betyg, inte en egen uppgift: betyget sätts på det ni
lämnat in, men rapporteras först när ni har redovisat.

Jag väljer ut någon eller några av buggarna, och ni får förklara:
* Vad koden gjorde fel, och vad er rättning ändrar.
* Vilket av era testfall som fångar buggen, och varför det gör det.
* Varför ett testfall är uppställt som det är, exempelvis varför `ActFunc::None` används i `Conv`,
  varför indatan är osymmetrisk, eller varför `Cnn` testas med stubbar i stället för riktiga lager.
* Vad som händer om en av rättningarna tas bort och testsviten körs igen.

AI-verktyg är tillåtna under arbetet, men koden ska vara skriven av er, och ni ska kunna redogöra
för varje rad ni lämnar in. Den som inte kan förklara sin egen kod får komplettera och redovisa på
nytt.

---
