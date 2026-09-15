# Bilaga B - Övningsuppgift: Enkelt neuralt nätverk i C++ (del I)
Ni ska bygga vidare på er `ml`-kodbas med ett interface för dense-lager samt en stubbklass som 
implementerar detta interface. Nätverket som använder lagren byggs under **L07**.

---

### 1. Katalogstruktur
Katalogen [`exercises`](../exercises) innehåller en färdig katalogstruktur med tomma headerfiler 
för interfacet och stubbklassen, samt en testsvit i `exercises/test` (se avsnitt 5). Skriv er kod 
där, eller bygg ut katalogstrukturen i er befintliga `ml`-kodbas enligt nedan:

```
ml/
├── include/
│   └── ml/
│       ├── dense_layer/
│       │   ├── interface.h
│       │   └── stub.h
│       └── types.h
├── source/
│   └── main.cpp
└── Makefile
```

Både interfacet och stubbklassen implementeras direkt i sina headerfiler, så makefilen behöver 
inte uppdateras under denna lektion.

---

### 2. Dense-lagrets interface
Det dolda lagret samt utgångslagret representeras av interfacet `ml::dense_layer::Interface`. 
En skarp implementation skapas först under **L08–L09**. Fram tills dess implementerar ni en enkel 
stubbklass `ml::dense_layer::Stub` som placeholder, se avsnitt 3 nedan.

I headerfilen `ml/dense_layer/interface.h`, lägg till namnrymden `ml::dense_layer`. I denna 
namnrymd, implementera ett interface döpt `Interface`. Samtliga metoder (förutom destruktorn) ska 
deklareras som rent virtuella (`= 0`).

* **`~Interface()`:** Ska sättas till `default` samt markeras `virtual` och `noexcept`.

Getters, samtliga `const`, `noexcept` samt `[[nodiscard]]`:

| Metod | Returnerar |
|---|---|
| `nodeCount()` | Antal noder i lagret (`std::size_t`). |
| `weightCount()` | Antal vikter per nod (`std::size_t`). |
| `output()` | Referens till lagrets utdata (skrivskyddad flyttalsvektor). |
| `error()` | Referens till lagrets fel (skrivskyddad flyttalsvektor). |
| `weights()` | Referens till lagrets vikter (skrivskyddad, tvådimensionell flyttalsvektor). |

Beräkningsmetoder, samtliga `noexcept` och med returtypen `bool`. Var och en returnerar `false` 
vid ogiltig indata (felaktig dimension eller ogiltig lärhastighet) och `true` annars, så att 
anroparen själv kan avgöra vad som ska göras åt felet. Som i **L02** och **L04** är 
`std::terminate()` reserverad för konstruktorn, som inte kan returnera någon felkod till 
anroparen:
* **`feedforward(input)`:** Genomför feedforward.
    * `input`: skrivskyddad flyttalsvektor med indata.
* **`backpropagate(output)`** (utgångslager): Beräknar fel utifrån referensvärden.
    * `output`: skrivskyddad flyttalsvektor med referensvärden.
* **`backpropagate(nextLayer)`** (dolt lager): Beräknar fel utifrån nästa lager.
    * `nextLayer`: referens till nästa lager (`const Interface&`).
* **`optimize(input, learningRate)`:** Uppdaterar bias och vikter.
    * `input`: skrivskyddad flyttalsvektor.
    * `learningRate`: flyttal.

---

### 3. Stubbklassen
I headerfilen `ml/dense_layer/stub.h`, lägg till namnrymden `ml::dense_layer`. I denna namnrymd, 
implementera en underklass döpt `Stub` som ärver `Interface` via publikt arv. Klassen ska markeras 
`final`. Stubben genomför ingen riktig beräkning; den finns enbart för att annan kod ska gå att 
kompilera, testköra och enhetstesta mot ett riktigt `dense_layer::Interface` innan en skarp 
`Dense`-implementation finns (se **L08–L09**). Nätverket ni bygger under **L07** testas i sin 
helhet mot denna stubb, så det är värt att få den rätt.

Lägg till privata medlemsvariabler för utdatan, felet, vikterna samt feedforward-räknaren.

Antalet noder och antalet vikter per nod behöver inga egna medlemsvariabler. Båda går att läsa ur 
de vektorer ni redan har: `nodeCount()` är utdatavektorns storlek, och `weightCount()` är 
viktmatrisens bredd.

Bias behöver ingen alls. Den ingår inte i `Interface`, och denna stubb optimerar aldrig 
någonting, så det skulle inte finnas något att lägga i den och inget sätt att läsa ut den igen.

**OBS!** Att utelämna den är en förenkling som bara gäller stubben. Det skarpa `Dense`-lagret ni 
skriver i **L08–L09** behöver en biasvektor: den är en träningsbar parameter som justeras 
tillsammans med vikterna vid varje anrop till `optimize()`.

Klassen ska inneha följande publika metoder:
* **`Stub()`:**
    * Klassens enda implementerade konstruktor.
    * Ska ha följande ingående argument:
        * `nodeCount`: antal noder i lagret (osignerat heltal).
        * `weightCount`: antal vikter per nod (osignerat heltal).
        * `outputValue`: värdet som varje element i utdatavektorn sätts till (flyttal). 
          Defaultvärde: `0.5`.
    * Initierar varje element i utdatavektorn till `outputValue`, samt fel- och viktvektorerna 
      till nollor.
    * Ska skriva ut ett felmeddelande och anropa `std::terminate()` om `nodeCount` eller 
      `weightCount` är 0:
        * Som i **L02** och **L04** är konstruktorn den enda plats som avslutar programmet, 
          eftersom den inte kan returnera någon felkod.
        * Det garanterar också minst en nod och en vikt, vilket är det som gör det säkert för 
          `weightCount()` att läsa bredden på viktmatrisens första rad.
    * `outputValue` är ett argument i stället för en hårdkodad konstant för att två stubbar ska 
      gå att skilja åt. Ett nätverk vars samtliga lager rapporterar samma utdata kan inte visa 
      om en prediktion kom från utgångslagret eller från det dolda lagret; ger man varje lager 
      sitt eget värde blir det synligt.
    * Ska markeras `explicit` samt `noexcept`.
* **`~Stub()`:**
    * Destruktor som överlagrar interfacets destruktor.
    * Ska markeras `default`, `noexcept` samt `override`.
* **Getters** (`nodeCount()`, `weightCount()`, `output()`, `error()`, `weights()`):
    * Överlagrar motsvarande metoder i interfacet.
    * Ska markeras `override` (behåller interfacets `const` och `noexcept`, men **inte** 
      `[[nodiscard]]`).
* **`feedforward()`**, båda varianterna av **`backpropagate()`** samt **`optimize()`**:
    * Genomför endast range-checkar, och returnerar `true` om samtliga villkor nedan är 
      uppfyllda, annars `false`:

      | Metod | Villkor |
      |---|---|
      | `feedforward(input)` | `input.size() == weightCount()` |
      | `backpropagate(output)` | `output.size() == nodeCount()` |
      | `backpropagate(nextLayer)` | `nextLayer.weightCount() == nodeCount()` |
      | `optimize(input, learningRate)` | `input.size() == weightCount()` samt `0.0 < learningRate < 1.0` |

    * Blanda inte ihop antalet noder och antalet vikter per nod. Indatan till ett lager har ett 
      värde per vikt, medan referensvärdena har ett värde per nod.
    * Beräknar avsiktligt ingenting:
        * Utdatan förblir `outputValue` oavsett vad som matas in.
        * Felet förblir noll.
    * Ska markeras `override` samt `noexcept`.
* **`setOutput()`:**
    * Sätter varje element i utdatavektorn till det angivna värdet.
    * Ska ha ett enda ingående argument:
        * `outputValue`: värdet som varje element i utdatavektorn ska sättas till (flyttal).
    * Returnerar ingenting, och ska markeras `noexcept`.
    * Ingår **inte** i `Interface`. Den finns enbart på stubben.
    * Det är denna metod som gör det möjligt att styra utdatan för ett helt nätverk i ett test:
        * Nätverket ni bygger under **L07** lagrar sina lager som referenser, så ett anrop till 
          `setOutput()` på det lager nätverket byggdes med ändrar vad nätverket predikterar.
        * Det avslöjar om nätverket läser sitt utgångslager direkt eller har sparat en egen 
          kopia av utdatan. Just den kopian är vad noteringen om `predict()` i **L07** är till 
          för att förhindra.
* **`feedforwardCount()`** samt **`clearFeedforwardCount()`:**
    * `feedforwardCount()` returnerar antalet gånger `feedforward()` har anropats på detta 
      lager, och räknar **varje** anrop, inte bara de som klarade range-checken:
        * Öka räknaren (`myFeedforwardCount`) allra först i `feedforward()`, innan indatans 
          storlek kontrolleras, så att ett avvisat anrop höjer den precis som ett godkänt.
        * Räknaren mäter hur ofta lagret *ombads* göra feedforward, inte hur ofta det gick med 
          på det.
        * Ska markeras `[[nodiscard]]`, `const` samt `noexcept`.
    * `clearFeedforwardCount()` nollställer räknaren.
        * Returnerar ingenting, och ska markeras `noexcept`.
    * Ingen av dem ingår i `Interface`.
    * `train()` i **L07** genomför en feedforward per träningsuppsättning och epok, så det är 
      detta som gör träningsloopen möjlig att fastställa i ett test:
        * Inget annat kan göra det: en loop som kör ett enda pass i stället för samtliga epoker 
          stämmer fortfarande dimensionsmässigt och returnerar fortfarande `true`, så utan en 
          räknare går den inte att skilja från en korrekt loop.

För denna klass ska default-konstruktorn samt copy- och move-konstruktorerna (och tillhörande 
operatorer) raderas.

---

### 4. Kompilering och test
Skriv en `main`-funktion i `main.cpp` som testar stubbklassen via följande steg:
1. Skapa en `ml::dense_layer::Stub`-instans döpt `hiddenLayer` med 3 noder och 2 vikter per nod 
   (defaultvärdet `0.5` som utdata).
2. Skapa en `ml::dense_layer::Stub`-instans döpt `outputLayer` med 1 nod, 3 vikter per nod samt 
   utdatan `0.8`. Antalet vikter i utgångslagret matchar antalet noder i det dolda lagret.
3. Skriv ut antalet noder, antalet vikter per nod samt utdatan för båda lagren.
4. Anropa `hiddenLayer.feedforward()` två gånger; först med 2 insignaler, sedan med 3. Skriv ut 
   returvärdena, lagrets utdata samt feedforward-räknaren.
5. Anropa `outputLayer.backpropagate()` med 1 referensvärde samt `hiddenLayer.backpropagate()` med 
   `outputLayer` som nästa lager. Skriv ut returvärdena.
6. Anropa `hiddenLayer.optimize()` med 2 insignaler två gånger; först med lärhastigheten `0.01`, 
   sedan med `1.5`. Skriv ut returvärdena.
7. Anropa `hiddenLayer.setOutput(0.1)` samt `hiddenLayer.clearFeedforwardCount()`. Skriv ut lagrets 
   utdata samt feedforward-räknaren.

Använd `std::boolalpha` för att skriva ut returvärdena som `true`/`false` i stället för `1`/`0`.

Kompilera och testkör programmet. Ni ska få följande utskrift:

```
--------------------------------------------------------------------------------
Hidden layer: node count: 3, weight count: 2, output: 0.5 0.5 0.5
Output layer: node count: 1, weight count: 3, output: 0.8
--------------------------------------------------------------------------------
Feedforward with 2 inputs: true
Feedforward with 3 inputs: false
Output after feedforward: 0.5 0.5 0.5
Feedforward count: 2
--------------------------------------------------------------------------------
Backpropagate output layer: true
Backpropagate hidden layer: true
Optimize with learning rate 0.01: true
Optimize with learning rate 1.5: false
--------------------------------------------------------------------------------
Output after setOutput(0.1): 0.1 0.1 0.1
Feedforward count after clearFeedforwardCount(): 0
--------------------------------------------------------------------------------
```

Notera följande i utskriften:
* Utdatan är fortfarande `0.5` efter feedforward, eftersom stubben inte beräknar någonting.
* Feedforward-räknaren är `2` trots att det andra anropet avvisades.
* `hiddenLayer.backpropagate(outputLayer)` lyckas eftersom utgångslagrets antal vikter per nod (3) 
  matchar det dolda lagrets antal noder (3). Detta är samma koppling som nätverket i **L07** 
  bygger på.

---

### 5. Enhetstester
Kontrollera er implementation mot testsviten i `exercises/test`. Se testsvitens 
[README](../exercises/test/README.md) för detaljer.

1. Om katalogen `libs/test` i repots rotkatalog är tom, hämta testramverket en gång via 
   kommandot `git submodule update --init`.
2. Bygg och kör testsviten via kommandot `make` i katalogen `exercises/test`.
    * Skriver ni er kod i er egen `ml`-kodbas i stället för i `exercises`, ange sökvägen till den 
      via `make ML_DIR=<sökväg till er ml-katalog>`.
3. Åtgärda eventuella fel och kör testsviten igen, tills samtliga testfall går igenom.

Testsviten kompilerar inte förrän båda headerfilerna finns och deklarerar samtliga metoder som 
testerna anropar. Läs det första kompileringsfelet; det anger oftast vilken metod som saknas eller 
har fel signatur.

Konstruktorns anrop till `std::terminate()` testas inte av testsviten, eftersom det avslutar hela 
testprogrammet. Kontrollera det för hand genom att skapa en stubb med 0 noder.

---
