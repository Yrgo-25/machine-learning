# Övningstentamen 2 - Konvolutionella neurala nätverk och testning

## Information
Denna tentamen är gemensam för kurserna **Maskininlärning** och **Mjuk- och hårdvarutestning**:
* Uppgift 1–5, 7–9 examinerar maskininlärning.
* Uppgift 6, 10–11 examinerar testning.

Uppgifterna är ordnade i den ordning ni förväntas lösa dem, inte efter ämne; testuppgift 6 kommer före implementationsuppgifterna 7–9 eftersom ni behöver de testerna för att verifiera era implementationer.

Betyg sätts separat per kurs enligt poänggränserna nedan.

### Hjälpmedel
* En A4 med anteckningar.
* Dator med textredigerare (t.ex. Visual Studio Code med IntelliSense).
* Kodkomplettering, AI-verktyg och internetåtkomst är inte tillåtna.

### Poänggränser och betygsnivåer

**Maskininlärning (uppgift 1–5, 7–9):**
Totalt: 28 poäng.

Betygsgränser:
* **G:** Minst 14 poäng.
* **VG:** Minst 23 poäng.

Bidrag till kursens slutpoäng:
* Betyget **G** ger 1 poäng till kurssammanställningen.
* Betyget **VG** ger 2 poäng till kurssammanställningen.

**Mjuk- och hårdvarutestning (uppgift 6, 10–11):**
Totalt: 20 poäng.

Betygsgränser:
* **G:** Minst 10 poäng.
* **VG:** Minst 16 poäng.

Bidrag till kursens slutpoäng: se separat information för kursen Mjuk- och hårdvarutestning.

### Viktigt
* Bekanta er med kodbasen i [code](./code) innan ni börjar: läs igenom [main.cpp](./code/source/main.cpp), [ml/types.h](./code/include/ml/types.h), [ml/utils.h](./code/include/ml/utils.h) samt [ml/cnn/cnn.h](./code/include/ml/cnn/cnn.h) för att se hur nätverket är uppbyggt och vilka hjälpfunktioner som redan finns tillgängliga. Bygg och kör gärna programmet med stubbarna (standardläget) för att se att allt fungerar innan ni börjar. Prediktionerna stämmer inte än, men programmet ska bygga och köra utan fel.
* I katalogen [code](./code) finns ett konvolutionellt neuralt nätverk (CNN) med utbytbara stubklasser. `ml::dense_layer::Dense` samt orkestreringen i `ml::cnn::Cnn` (feedforward, backpropagation, optimering och träning över samtliga lager) är redan implementerade. Er uppgift är att implementera de tre konvolutionella lagerklasserna (uppgift 7–9) så att nätverket fungerar som tänkt.
* `ml::conv_layer::Conv`, `ml::conv_layer::MaxPool` samt `ml::flatten_layer::Flatten` ska implementeras utifrån respektive interface samt tillhörande stubklass (`ConvStub`, `MaxPoolStub` respektive `flatten_layer::Stub`) i [ml/conv_layer/stub.h](./code/include/ml/conv_layer/stub.h) och [ml/flatten_layer/stub.h](./code/include/ml/flatten_layer/stub.h). Stubbarna visar den dimensionskontroll (`matchDimensions()`, `isMatrixSquare()`) som era riktiga implementationer också ska hantera, samt hur ogiltiga konstruktorargument ska hanteras: skriv ut ett felmeddelande och avsluta programmet (`std::terminate()`). Undantag (`throw`/`try`/`catch`) ska inte användas någonstans i kodbasen, och samtliga funktioner ska vara markerade `noexcept`.
* Det är okej att modifiera interfacet något, t.ex. genom att låta lärhastigheten utgöra en medlemsvariabel i `Conv` i stället för att passera denna vid `optimize()`.
* När ett lager är implementerat, uppdatera [ml/factory/factory.cpp](./code/source/ml/factory/factory.cpp) så att er implementation används i stället för stubben. Testkör nätverket genom att kommentera ut flaggan `-DSTUB` under `COMPILER_FLAGS` i [makefilen](./code/makefile):

```bash
COMPILER_FLAGS := -Wall -Werror -std=c++17 #-DSTUB
```

* Uppgift 6 samt 10–11 (testning) använder testramverket `yrgo::test` (se [libs/test](../../libs/test/README.md) för `TEST()`, `EXPECT_*` med mera). Testerna ligger i katalogen [test](./code/test), som har en egen makefil; bygg och kör dem genom att köra `make` i den katalogen.
* **OBS!** Den konvolution ni räknar för hand i uppgift 3 använder ingen padding (utdatan krymper), medan `Conv` i uppgift 7 ska använda zero-padding så att utdatan behåller samma storlek som indatan. Detta är en medveten skillnad, inte en motsägelse: räkneuppgiften illustrerar själva konvolutionsoperationen, medan koden följer den spatiala bevarande-konventionen som resten av nätverket förutsätter. På samma sätt använder poolingexemplet i uppgift 5 ett överlappande poolingfönster (stride = 1) för att illustrera principen på en feature map vars storlek inte är jämnt delbar, medan `MaxPool` i uppgift 8 ska använda icke-överlappande pooling (stride = pool-storlek).

---

## Uppgifter

### **1.** CNN:ar för bildklassificering (2p · Maskininlärning, G)
Förklara kortfattat varför konvolutionella neurala nätverk (CNN-nätverk) används i samband med bildklassificering. Varför används inte enbart dense-lager såsom i traditionella neurala nätverk?

---

### **2.** Kernels (2p · Maskininlärning, G)
Mellan konvolutionella lager i CNN-nätverk implementeras oftast så kallade kernels.
* Förklara vad dessa kernels har för funktion samt ange inom vilket storleksintervall sådana kernels brukar ligga.
* Vilka för- och nackdelar finns gällande användning av mindre respektive större kernels?

---

### **3.** Feature map för hand (4p · Maskininlärning, G)
Nedan visas en 5x5-bild följt av en 3x3-kernel. Använd aktiveringsfunktionen ReLU och **ingen padding** (utdatan blir alltså mindre än indatan).

Beräkna erhållen feature map som resulteras från operationen. Visa genomförda beräkningar.

**Bild:**
```
 1   0   2   3   1
 9   2  -3   2   6
 4   5   3   4   8
 4   5   6   7   8
-2   0   4  -4   1
```

**Kernel:**
```
0  4  1
6  4  4
2  3  5
```

---

### **4.** Pooling-lager (2p · Maskininlärning, G)
Mellan konvolutionella lager används så kallade pooling-lager. Förklara vad dessa pooling-lager har för funktion samt hur de fungerar, både för max pooling samt average pooling.

---

### **5.** Pooling för hand (4p · Maskininlärning, G)
Utgå från feature map:en ni räknade fram i uppgift 3. Använd ett 2x2 poolingfönster med **stride 1** (dvs. fönstret överlappar mellan varje position).
* Beräkna erhållen feature map efter max pooling.
* Använd average pooling i stället och jämför resultatet. Vilken pooling-metod anser ni vara lämpligast här, och varför?

---

### **6.** Unit- och komponenttester (8p · Testning)
Skriv unit-tester för `ml::conv_layer::Conv`, `ml::conv_layer::MaxPool` samt `ml::flatten_layer::Flatten` med `yrgo::test`-ramverket (se [libs/test](../../libs/test/README.md)). Exempel finns redan i [test/ml/dense_layer/test_dense.cpp](./code/test/ml/dense_layer/test_dense.cpp). Lägg era tester i motsvarande kataloger, t.ex. `test/ml/conv_layer/test_conv.cpp`.

Skriv testerna innan (eller i samband med) att ni implementerar respektive lager i uppgift 7–9, och använd dem för att verifiera era implementationer. Verifiera bland annat:
* Att konstruktorn sätter korrekt `inputSize()` och `outputSize()`.
* Att `feedforward()`, `backpropagate()` samt `optimize()` returnerar `false` vid ogiltig indata (felaktig dimension, icke-kvadratisk matris, ogiltig lärhastighet) och `true` vid giltig indata, i linje med stubbarnas kontrakt.
* Att `MaxPool::feedforward()` faktiskt väljer maxvärdet i respektive pool (inte bara att den returnerar `true`).
* Att `Flatten::feedforward()` samt `Flatten::backpropagate()` korrekt konverterar mellan 2D och 1D, elementen i samma ordning.

---

### **7.** Implementera `Conv` (6p · Maskininlärning, VG)
Implementera klassen `ml::conv_layer::Conv` genom att utgå från både interfacet och stubklassen `ConvStub`. Din implementation ska korrekt hantera feedforward, backpropagation samt optimering enligt de algoritmer som gäller för konvolutionella lager. Se till att din kod är tydlig, väldokumenterad och följer projektets struktur.

Implementationen ska göras i följande filer:
* [ml/conv_layer/conv.h](./code/include/ml/conv_layer/conv.h)
* [ml/conv_layer/conv.cpp](./code/source/ml/conv_layer/conv.cpp)

Indatan ska paddas med nollor (zero-padding) så att utdata får samma spatiala dimensioner som indatan. Stride ska sättas till 1, vilket innebär att konvolutionen förflyttas ett pixelsteg i taget i både x- och y-led.

När du är klar, uppdatera [ml/factory/factory.cpp](./code/source/ml/factory/factory.cpp) så att din riktiga implementation används i stället för stubben, och testkör nätverket enligt beskrivningen i "Viktigt".

---

### **8.** Implementera `MaxPool` (4p · Maskininlärning, VG)
Implementera klassen `ml::conv_layer::MaxPool` genom att utgå från både interfacet och stubklassen `MaxPoolStub`. Din implementation ska korrekt hantera feedforward samt backpropagation enligt de algoritmer som gäller för maxpooling-lager. Se till att din kod är tydlig, väldokumenterad och följer projektets struktur.

Implementationen ska göras i följande filer:
* [ml/conv_layer/max_pool.h](./code/include/ml/conv_layer/max_pool.h)
* [ml/conv_layer/max_pool.cpp](./code/source/ml/conv_layer/max_pool.cpp)

Poolerna ska inte vara överlappande, dvs. stride ska sättas till pool-storleken. Vid backpropagation ska gradienten endast sättas för positionen för maxvärdet i respektive pool. Övriga positioner ska få gradient 0.

När du är klar, uppdatera [ml/factory/factory.cpp](./code/source/ml/factory/factory.cpp) så att din riktiga implementation används i stället för stubben, och testkör nätverket enligt beskrivningen i "Viktigt".

---

### **9.** Implementera `Flatten` (4p · Maskininlärning, VG)
Implementera klassen `ml::flatten_layer::Flatten` genom att utgå från både interfacet och stubklassen `flatten_layer::Stub`. Din implementation ska korrekt hantera feedforward samt backpropagation för att omvandla matriser från 2D till 1D och tvärtom. Se till att din kod är tydlig, väldokumenterad och följer projektets struktur.

Implementationen ska göras i följande filer:
* [ml/flatten_layer/flatten.h](./code/include/ml/flatten_layer/flatten.h)
* [ml/flatten_layer/flatten.cpp](./code/source/ml/flatten_layer/flatten.cpp)

När du är klar, uppdatera [ml/factory/factory.cpp](./code/source/ml/factory/factory.cpp) så att din riktiga implementation används i stället för stubben. Testkör nätverket; utskriften bör nu stämma överens med prediktionerna i [README](./code/README.md).

---

### **10.** Komponenttester för `Cnn` (8p · Testning)
Ingen av de befintliga stubklasserna (`ConvStub`, `MaxPoolStub`, `dense_layer::Stub`, `flatten_layer::Stub`) tillåter er att manuellt styra ett lagers utdata. Komplettera **en** av dessa stubbar, t.ex. `dense_layer::Stub`, med:
* En metod som lagrar en given utdata internt i stubben.
* En metod som tar bort denna override.
* En överlagring av `feedforward()` som, när en utdata har satts, returnerar `true` utan att skriva över den satta utdatan.

Skriv därefter komponenttester för `ml::cnn::Cnn` med `yrgo::test`-ramverket, där ni bygger nätverket via en `ml::factory::Interface` sammansatt av stubbar (se `ml::factory::Stub` i [ml/factory/stub.h](./code/include/ml/factory/stub.h)). Ett exempel utan denna override finns redan i [test/ml/cnn/test_cnn.cpp](./code/test/ml/cnn/test_cnn.cpp). Verifiera bland annat:
* Att `predict()` returnerar exakt den utdata som er kompletterade stubb är satt till, oavsett given indata.
* Lämpliga edge cases, t.ex. ogiltiga parametrar till `train()` (epok-antal, lärhastighet, saknad träningsdata).
* Skriv även minst ett test som tränar ett `Cnn` med **riktiga** lager (inte stubbar, och alltså först meningsfullt sedan ni implementerat uppgift 7–9) på de fyra bilderna i [main.cpp](./code/source/main.cpp), och verifierar att nätverket lär sig prediktera rätt siffra för respektive bild.

---

### **11.** Testteori (4p · Testning)
Besvara följande kortfattat:
* Vad är skillnaden mellan ett unit-test och ett komponenttest, och varför används lagerstubbar i komponenttesterna för `Cnn` i stället för riktiga implementationer?
* Varför är det viktigt att `Conv`, `MaxPool` och `Flatten` returnerar `false` i stället för att avsluta programmet vid ogiltig indata till `feedforward()`, `backpropagate()` eller `optimize()` (till skillnad från konstruktorn, som redan skriver ut ett felmeddelande och anropar `std::terminate()` vid ogiltiga konstruktorargument i stubbarna), med tanke på hur `yrgo::test` rapporterar testfel?
* I uppgift 10 skriver ni ett test som tränar `Cnn` på riktiga bilder och kontrollerar slutresultatet, snarare än att bara kontrollera returvärden efter varje enskilt anrop. Vilken typ av bugg kan ett sådant test hitta som varken unit-testerna i uppgift 6 eller enkla sant/falskt-kontroller av `train()` kan hitta?

---
