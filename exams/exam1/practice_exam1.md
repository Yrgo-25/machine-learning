# Övningstentamen 1 - Neurala nätverk och testning

## Information
Denna tentamen är gemensam för kurserna **Maskininlärning** och **Mjuk- och hårdvarutestning**:
* Uppgift 3 och 5 examinerar maskininlärning.
* Uppgift 1, 2 och 4 examinerar testning.

Betyg sätts separat per kurs enligt poänggränserna nedan.

### Hjälpmedel
* En A4 med anteckningar.
* Dator med textredigerare (t.ex. Visual Studio Code med IntelliSense).
* Kodkomplettering, AI-verktyg och internetåtkomst är inte tillåtna.

### Poänggränser och betygsnivåer

**Maskininlärning (uppgift 3 och 5):**
Totalt: 30 poäng.

Betygsgränser:
* **G:** Minst 14 poäng.
* **VG:** Minst 23 poäng.

Bidrag till kursens slutpoäng:
* Betyget **G** ger 1 poäng till kurssammanställningen.
* Betyget **VG** ger 2 poäng till kurssammanställningen.

**Mjuk- och hårdvarutestning (uppgift 1, 2 och 4):**
Totalt: 20 poäng.

Betygsgränser:
* **G:** Minst 10 poäng.
* **VG:** Minst 16 poäng.

Bidrag till kursens slutpoäng: se separat information för kursen Mjuk- och hårdvarutestning.

---

## Uppgiften i korthet
Katalogen [code](./code) innehåller ett komplett neuralt nätverk av samma typ som ni byggde under
**L06-L09**: klassen `Dense`, nätverket `Shallow`, stubben `Stub` samt hjälpfunktionerna i
`ml/helpers.h`. Koden kompilerar och kör, men **sex buggar har medvetet planterats**:
* **Fyra i `Dense`**, i [source/ml/dense_layer/dense.cpp](./code/source/ml/dense_layer/dense.cpp).
* **Två i `Shallow`**, i
  [source/ml/neural_network/shallow.cpp](./code/source/ml/neural_network/shallow.cpp).

Er uppgift är att skriva tester för koden, hitta buggarna via de testfall som misslyckas, och rätta
dem:

1. Skriv vanliga testfall för lagrets och nätverkets beteende, ett per beräkning. Ett test av
   `feedforward()` räknar exempelvis ut den förväntade utdatan ur lagrets egna `bias()` och
   `weights()`, och jämför mot det lagret faktiskt gav.
2. Kör testerna. De som misslyckas pekar ut var buggarna sitter.
3. Rätta buggen.
4. Kör testerna igen; nu ska de gå igenom.

Ni skriver alltså inga testfall som är särskilt konstruerade för en viss bugg, utan helt vanliga
testfall som kontrollerar att koden gör det den ska. Det räcker med ett testfall per beräkning.

### Viktigt
* Övriga filer är korrekta och ska inte ändras: `interface.h` (båda), `stub.h`, `types.h`,
  `helpers.h`, `helpers.cpp`, `main.cpp` samt makefilerna. Deklarationerna i
  [dense.h](./code/include/ml/dense_layer/dense.h) är också korrekta; samtliga fyra buggar i
  `Dense` ligger i implementationsfilen.
* Hjälpfunktionerna `ml::actFuncOutput()` och `ml::actFuncDelta()` är korrekta och stödjer
  `ActFunc::Relu`, `ActFunc::Tanh` samt `ActFunc::None`.
* Bygg och kör demoprogrammet med `make` i katalogen [code](./code). Programmet tränar ett
  2-10-1-nätverk på XOR-mönstret och skriver ut prediktionerna. Lägg till `-DSTUB` i makefilens
  `CXX_FLAGS` för att bygga demot med stubbar i stället.
* Bygg och kör testerna med `make` i katalogen [code/test](./code/test). Där finns sex
  exempeltester som samtliga går igenom; de visar mönstret ni bygger vidare på och avslöjar inte
  någon av buggarna.
* Felmeddelanden som `Feedforward failed due to dimension mismatch` skrivs ut av de testfall som
  medvetet anropar med felaktiga argument. De betyder alltså inte att ett test har misslyckats.
* Koden behöver ej kommenteras.

### Två tips på vägen
* **Lagrets parametrar går att läsa av.** `bias()` ger biasvärdena och `weights()` vikterna, så ett
  förväntat värde kan räknas ut direkt ur lagret. Aktiveringsfunktionen `ActFunc::None` släpper
  dessutom igenom den viktade summan orörd, vilket gör utdatan enkel att jämföra mot. Testfallet
  `Dense.FeedforwardComputesWeightedSum` visar hela mönstret.
* **Demoprogrammet räcker inte som facit.** Startvärdena slumpas, så en körning kan misslyckas
  även med helt korrekt kod, och lyckas trots en bugg. Använd programmet för att se symptom, och
  tester för att avgöra vad som faktiskt är fel.

---

## Uppgifter

### **1.** Unit-tester för `Dense`: beräkningarna (8p · Testning)
Skriv ett testfall per beräkning i `Dense`, alltså fyra stycken (2p styck), i
[test/ml/dense_layer/test_dense.cpp](./code/test/ml/dense_layer/test_dense.cpp) med ramverket
`yrgo::test` (se [libs/test](../../libs/test/README.md)). Räkna i varje testfall ut det förväntade
värdet ur lagrets egna parametrar, som `bias()` och `weights()` ger, och jämför med vad lagret
faktiskt gav:
* **`feedforward()`:** utdatan ska vara aktiveringsfunktionen tillämpad på
  `bias() + summan av weights() * input`, nod för nod.
* **`backpropagate()` (utgångslager):** felet ska vara referensvärdet minus utdatan, skalat med
  aktiveringsfunktionens derivata.
* **`backpropagate()` (dolt lager):** felet ska vara summan av nästa lagers fel gånger vikterna som
  förbinder noderna, skalat med aktiveringsfunktionens derivata.
* **`optimize()`:** biasvärdet ska ha ökat med felet gånger lärhastigheten, och varje vikt med
  samma tal gånger sin egen insignal. Läs av `bias()` och `weights()` både före och efter anropet.

Två saker påverkar om ett testfall faktiskt fångar det det ska:
* Välj aktiveringsfunktion med omsorg. `ActFunc::None` ger derivatan `1.0`, vilket döljer om
  derivatan saknas i beräkningen.
* Välj indata med olika värden, exempelvis `{2.0, -1.0}`. Med `{1.0, 1.0}` blir flera felaktiga
  formler omöjliga att skilja från de korrekta.

---

### **2.** Unit-tester för `Dense`: argumentkontroller (6p · Testning)
Varje beräkningsmetod i `Dense` ska avvisa ogiltiga argument och returnera `false`, och returnera
`true` när argumenten är giltiga. Exempeltestet `Dense.FeedforwardChecksInputSize` visar mönstret
för `feedforward()`. Skriv motsvarande testfall för de övriga metoderna:
* **`backpropagate()`, båda överlagringarna (3p):** fel antal referensvärden respektive ett nästa
  lager vars viktantal inte matchar detta lagers nodantal ska ge `false`, och rätt dimensioner
  `true`.
* **`optimize()` (3p):** en lärhastighet utanför `(0.0, 1.0)` samt fel dimension på indatan ska ge
  `false`, och giltiga argument `true`.

Kontrollerna i `Dense` är korrekt implementerade i den utlämnade koden, så dessa testfall ska gå
igenom direkt. De examinerar att ni kan testa en metods kontrakt, inte att ni hittar buggar.

---

### **3.** Rätta buggarna i `Dense` (20p · Maskininlärning)
Buggarna ligger en i vardera `feedforward()`, `backpropagate()` (utgångslager), `backpropagate()`
(dolt lager) samt `optimize()`. Rätta samtliga (5p styck), så att testfallen från uppgift 1 går
igenom.

---

### **4.** Komponenttester för `Shallow` (6p · Testning)
Skriv två testfall (3p styck) i
[test/ml/neural_network/test_shallow.cpp](./code/test/ml/neural_network/test_shallow.cpp), med två
`ml::dense_layer::Stub` som dolt lager respektive utgångslager. En stubb har känd utdata, som ni
sätter själva via `setOutput()`, medan ett `Dense`-lager har slumpade startvärden:
* **`predict()`:** ska returnera exakt den utdata som utgångslagrets stubb har satts till, oavsett
  indata.
* **`train()`:** ska följa kontraktet i
  [shallow.h](./code/include/ml/neural_network/shallow.h), dvs. returnera `false` vid ogiltiga
  argument, som noll epoker eller en lärhastighet utanför `(0.0, 1.0)`, och `true` vid giltiga.

---

### **5.** Rätta buggarna i `Shallow` (10p · Maskininlärning)
Rätta båda buggarna (5p styck), så att testfallen från uppgift 4 går igenom.

**Ledtråd:** båda buggarna syns i testfallen från uppgift 4, utan en enda `Dense`-instans.

---

## Redovisning
Lämna in:
* Testfallen i [code/test](./code/test).
* Den rättade koden i `dense.cpp` samt `shallow.cpp`.

När samtliga sex buggar är rättade ska testerna gå igenom, och demoprogrammet ska vid de flesta
körningar prediktera XOR-mönstret, dvs. `{0.0}`, `{1.0}`, `{1.0}` och `{0.0}`.

---

## Självkontroll
Hur vet ni att ni har hittat alla sex? Nedan följer sex frågor att ställa till koden, en per bugg,
i samma ordning som buggarna ligger i uppgift 3 och 5. Frågorna säger inte vad som är fel, utan
vilken egenskap som ska gälla; svarar er kod `ja` på samtliga, och testerna som visar det går
igenom, har ni hittat samtliga buggar.

1. Ger två noder i samma lager olika utdata för samma indata, när deras vikter skiljer sig åt?
2. Blir felet i ett utgångslager positivt när prediktionen är för låg, och negativt när den är för
   hög? Använd `ActFunc::None` eller `ActFunc::Tanh`, vars derivata alltid är positiv.
3. Blir felet noll för en nod i ett dolt lager vars ReLU-utdata är noll, även när nästa lager har
   ett fel skilt från noll?
4. Lämnas en vikt vars insignal var noll orörd av `optimize()`, medan vikten bredvid, med
   insignalen 2.0, ändras dubbelt så mycket som biasvärdet?
5. Innehåller prediktionen från `predict()` lika många värden som utgångslagret har noder?
6. Avvisar `train()` en lärhastighet på exakt `1.0`?

Utöver detta finns två kontroller av helheten:
* Kör `make check` i katalogen [code](./code). Demoprogrammet körs då fem gånger med en sekunds
  paus emellan, så att startvärdena hinner slumpas om. Med samtliga buggar rättade ska
  prediktionerna bestå av ett enda värde och följa XOR-mönstret vid de flesta körningarna.
* En korrekt implementation sänker felet för en och samma träningsuppsättning när den tränas om
  och om igen. Ett litet testfall som tränar ett nätverk i några hundra epoker och jämför felet
  före och efter är ofta det som fäller den sista kvarvarande buggen.

---

## Lösningsförslag
Lösningsförslag finns [här](./practice_exam1_solution.md), med samtliga sex buggar, rättningarna
samt det testfall som fångar respektive bugg. **OBS!** Läs det först efter ert eget försök.

---
