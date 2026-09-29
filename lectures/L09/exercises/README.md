# L09 - Lösningsförslag: Dense-lager i C++ (del II)
Lösningsförslag till övningsuppgiften i [bilaga B](../appendix/b_exercises.md).
Katalogen innehåller hela `ml`-kodbasen efter denna lektion, dvs. dense-lagrets interface och stubb
(**L06**), nätverket `Shallow` (**L07**) samt klassen `Dense`, som nu är färdig.

Det som tillkommer under **L09** ligger uteslutande i
[dense.cpp](./source/ml/dense_layer/dense.cpp); varken interfacet, stubben, `Shallow` eller
`dense.h` behövde ändras en enda rad.

---

## Vad L09 bygger
**Fyra hjälpfunktioner i en anonym namnrymd.** `initRandom()` och `randomStartVal()` sköter
slumptalen, medan `actFuncOutput()` och `actFuncDelta()` översätter lagrets `ActFunc`-värde till
faktisk matematik.

**Randomiserade startvärden.** Den nya privata metoden `initParams()` fyller bias och vikter med
slumptal i intervallet `[-1.0, 1.0]` i stället för att lämna dem på `0.0`.

**Beräkningarna i de fyra metoderna.** `feedforward()`, båda varianterna av `backpropagate()` samt
`optimize()` behåller sina argumentkontroller från **L08** oförändrade och räknar efter dem.

**Ett nätverk som lär sig.** Med beräkningarna på plats ändrar träningen prediktionerna, och
nätverket i [main.cpp](./source/main.cpp) lär sig XOR-mönstret. Demoprogrammets parametrar är
uppdaterade till 10 dolda noder, lärhastigheten `0.05` och 10 000 epoker, enligt förslaget i
avsnitt 9 i bilaga B.

---

## Filer

| Fil | Innehåll |
|---|---|
| [source/ml/dense_layer/dense.cpp](./source/ml/dense_layer/dense.cpp) | Lektionens enda ändrade implementationsfil: de fyra hjälpfunktionerna, randomiseringen i konstruktorn samt beräkningarna i `feedforward()`, `backpropagate()` och `optimize()`. |
| [source/main.cpp](./source/main.cpp) | Demoprogrammet från **L08**, med uppdaterade värden på `hiddenCount`, `epochCount` och `learningRate`. |
| [include/ml/dense_layer/dense.h](./include/ml/dense_layer/dense.h) | Den privata metoden `initParams()` tillkommer. Allt annat är oförändrat från **L08**; samtliga övriga metoder och medlemsvariabler deklarerades redan då, `myPreActOutput` inkluderad. |
| [include/ml/types.h](./include/ml/types.h), [include/ml/dense_layer/interface.h](./include/ml/dense_layer/interface.h), [include/ml/dense_layer/stub.h](./include/ml/dense_layer/stub.h) | Oförändrade från **L06** och **L08**. |
| [include/ml/neural_network/](./include/ml/neural_network/), [source/ml/neural_network/](./source/ml/neural_network/) | Oförändrade från **L07**: nätverkets interface samt klassen `Shallow`. |
| [Makefile](./Makefile) | Oförändrad från **L08**. |

Katalogen [test](./test/) innehåller lektionens testsvit, som beskrivs i sin
[egen README](./test/README.md).

---

## Hjälpfunktionerna
Samtliga fyra ligger i en anonym namnrymd högst upp i `dense.cpp`. De hör till implementationen,
inte till klassens gränssnitt, och den anonyma namnrymden gör dem osynliga utanför filen. Ingen av
dem rör någon medlemsvariabel; de tar emot det de behöver som argument och returnerar ett värde,
vilket också gör dem lätta att resonera om.

### `initRandom()` och `randomStartVal()`

```cpp
void initRandom() noexcept
{
    // The static variable keeps its value between calls.
    static bool initialized{false};
    if (initialized) { return; }

    // Seed the random generator with the current time, occurs only once.
    std::srand(std::time(nullptr));
    initialized = true;
}

[[nodiscard]] double randomStartVal() noexcept
{
    // Generate a value in range [0.0, 1.0], cast to avoid integer division.
    constexpr double max{static_cast<double>(RAND_MAX)};
    const auto ratio = rand() / max;

    // Rescale to range [-1.0, 1.0], the start values must be able to go negative.
    return 2.0 * ratio - 1.0;
}
```

Den statiska lokala variabeln `initialized` gör att generatorn seedas exakt en gång per
programkörning, oavsett hur många lager som skapas. Samma konstruktion som i regressionsmodellen
från **L03**, och av samma skäl: `std::srand()` nollställer generatorns tillstånd, och tidsfröet har
sekundupplösning, så två seedningar inom samma sekund ger identiska talföljder.

`randomStartVal()` skalar om `std::rand()` från `[0, RAND_MAX]` till `[-1.0, 1.0]`. Omvandlingen av
`RAND_MAX` till `double` är det som gör divisionen till flyttalsdivision; utan den hade
heltalsdivision gett `0` för varje utfall utom det allra största.

### `actFuncOutput()` och `actFuncDelta()`

```cpp
[[nodiscard]] double actFuncOutput(const ActFunc actFunc, const double input) noexcept
{
    switch (actFunc)
    {
        case ActFunc::Relu:
            return 0.0 < input ? input : 0.0;
        case ActFunc::Tanh:
            return std::tanh(input);
        default:
            return input;
    }
}
```

`actFuncDelta()` är uppbyggd likadant, men returnerar derivatan: `1.0` respektive `0.0` för ReLU,
`1.0 - tanh(input) * tanh(input)` för tanh, samt `1.0` för `ActFunc::None`.

Det är här `ActFunc`-värdet från **L08** äntligen används. Lagret innehåller ingen `if`-sats om
aktiveringsfunktioner; det skickar sitt `myActFunc` till hjälpfunktionerna, som väljer beräkning.
En femte aktiveringsfunktion kräver därmed två nya `case`-grenar och ingenting annat.

Båda funktionerna är markerade `[[nodiscard]]`: ett bortkastat returvärde innebär att beräkningen
gjordes i onödan, eftersom funktionerna saknar sidoeffekter.

---

## Randomiseringen i `initParams()`
Konstruktorn kontrollerar sina argument, ger vektorerna sina storlekar och överlåter därefter
startvärdena till en egen privat metod:

```cpp
void Dense::initParams() noexcept
{
    // Seed the random generator, occurs only for the first layer created.
    initRandom();

    // Randomize the trainable parameters, use range [-1.0, 1.0].
    for (std::size_t i{}; i < nodeCount(); ++i)
    {
        myBias[i] = randomStartVal();

        for (std::size_t j{}; j < weightCount(); ++j)
        {
            myWeights[i][j] = randomStartVal();
        }
    }
}
```

Metoden anropas sist i konstruktorn, efter nollkontrollerna och efter att vektorerna har fått sina
storlekar; dessförinnan finns det ingenting att skriva till. Två saker följer av att loopen ligger
i en metod i stället för i konstruktorn:
* **Gränserna hämtas från `nodeCount()` och `weightCount()`**, inte från konstruktorns parametrar
  med samma namn. Metoderna läser vektorernas storlekar, vilket är samma tal, och de är det enda
  som är tillgängligt utanför konstruktorn.
* **Konstruktorn blir kortare och gör en sak i taget:** kontrollera, allokera, initiera. Namnet
  `initParams()` säger dessutom vad sista steget gör, vilket kommentaren annars fick göra.

**Varför startvärdena måste kunna bli negativa.** Med enbart positiva vikter och bias växer varje
ReLU-nods viktade summa med varje insignal som är hög. Insignalen `{1, 1}` ger då alltid den
största summan i varje nod, och därmed den största utsignalen, medan XOR kräver motsatsen: `{1, 1}`
ska ge `0`, lägre än både `{0, 1}` och `{1, 0}`. Ett sådant nätverk kan inte lära sig mönstret,
oavsett antal noder, epoker eller lärhastighet. Symmetriska startvärden kring noll ger dessutom
tanh-lager en start i funktionens känsliga område i stället för i mättnad, där derivatan är nära
noll.

**Varför varje nod får egna slumptal.** Skulle samtliga noder i ett lager starta med identiska
parametrar skulle de få identiska fel och identiska uppdateringar, och därmed förbli identiska.
Lagret hade då i praktiken haft en enda nod.

---

## Metoden `feedforward()`

```cpp
for (std::size_t i{}; i < nodeCount(); ++i)
{
    // Add the bias first.
    auto sum = myBias[i];

    // Add the contribution from the inputs, multiply with the corresponding weights.
    for (std::size_t j{}; j < weightCount(); ++j)
    {
        sum += myWeights[i][j] * input[j];
    }
    // Store the layer output, pre- and post activation function filtering.
    myPreActOutput[i] = sum;
    myOutput[i]       = actFuncOutput(myActFunc, sum);
}
```

Den yttre loopen går genom lagrets noder, den inre genom nodens vikter; en vikt per insignal.
Summan startar på nodens bias, vilket är samma sak som att se biasvärdet som en vikt vars insignal
alltid är 1.

**Både summan och utdatan sparas.** `myPreActOutput[i]` är den viktade summan $s$ innan
aktiveringsfunktionen, och `myOutput[i]` är utdatan $y$ efter den. Anledningen är
backpropagationen nedan: derivatan ska beräknas ur $s$, och $s$ går inte att räkna tillbaka ur $y$
för alla aktiveringsfunktioner. ReLU är inte inverterbar för negativa summor, där samtliga summor
ger utdatan `0.0`.

---

## Metoden `backpropagate()`
Utgångslagret, som jämför mot referensvärden:

```cpp
for (std::size_t i{}; i < nodeCount(); ++i)
{
    // Compute the deviation between the reference value and the predicted output.
    const auto error = reference[i] - myOutput[i];

    // Scale with the derivative, computed from the weighted sum before activation.
    const auto delta = actFuncDelta(myActFunc, myPreActOutput[i]);
    myError[i]       = error * delta;
}
```

Dolda lager, som i stället summerar felen från nästa lager:

```cpp
for (std::size_t i{}; i < nodeCount(); ++i)
{
    double sum{};

    // Sum the error of each node in the next layer, weighted by the connecting weight.
    for (std::size_t j{}; j < nextLayer.nodeCount(); ++j)
    {
        sum += nextLayer.error()[j] * nextLayer.weights()[j][i];
    }
    // Scale with the derivative, just like for the output layer.
    const auto delta = actFuncDelta(myActFunc, myPreActOutput[i]);
    myError[i]       = sum * delta;
}
```

Tre detaljer är värda att stanna vid:
* **Indexeringen `nextLayer.weights()[j][i]`.** Nästa lagers viktmatris är indexerad
  `[nod][vikt]`, och det är nod `j` i nästa lager som håller vikten till nod `i` i detta lager.
  Ordningen på indexen är alltså omvänd mot vad loopvariablerna kan antyda.
* **Derivatan beräknas ur `myPreActOutput[i]`, inte `myOutput[i]`.** För ReLU råkar båda ge samma
  svar, eftersom summan och utdatan har samma tecken, men för tanh blir derivatan fel. Testfallet
  `BackpropagateUsesPreActivationDerivative` fångar just det misstaget.
* **Ingen parameter uppdateras här.** Precis som vid handräkningen i **L05** beräknas samtliga fel
  innan någon vikt ändras; `Shallow` anropar därför båda lagrens `backpropagate()` innan det första
  anropet till `optimize()`. Hade utgångslagret optimerats först hade det dolda lagret fått sina
  fel ur redan justerade vikter.

Att lagret tar emot nästa lager som `const Interface&` är samma poäng som i **L07**: ett dolt lager
behöver bara `error()`, `weights()` och `nodeCount()`, och bryr sig inte om huruvida nästa lager är
en `Dense` eller en `Stub`.

---

## Metoden `optimize()`

```cpp
for (std::size_t i{}; i < nodeCount(); ++i)
{
    // Adjust the bias with the full change rate.
    const auto changeRate = myError[i] * learningRate;
    myBias[i] += changeRate;

    // Adjust each weight with the change rate scaled by its own input.
    for (std::size_t j{}; j < weightCount(); ++j)
    {
        myWeights[i][j] += changeRate * input[j];
    }
}
```

`changeRate` är $\Delta c$ från **L05**, och beräknas en gång per nod i stället för en gång per
vikt. Biasvärdet får hela förändringen, medan varje vikt får den skalad med sin egen insignal; en
vikt vars insignal var noll lämnas därmed orörd, precis som vid handräkningen.

Metoden skriver enbart till `myBias` och `myWeights`. Att `myOutput` lämnas orörd är vad som gör
det säkert att `Shallow::optimize()` optimerar det dolda lagret först och därefter skickar in
`myHiddenLayer.output()` som utgångslagrets indata: värdena är fortfarande de som beräknades under
feedforward, inte några nya.

---

## Bygg och kör
Bygg och kör programmet via följande kommando i denna katalog:

```bash
make
```

En körning kan se ut såsom visas nedan:

```
--------------------------------------------------------------------------------
Predictions before training:
Input: 0.0 0.0, predicted output: -0.0
Input: 0.0 1.0, predicted output: -0.7
Input: 1.0 0.0, predicted output: -0.8
Input: 1.0 1.0, predicted output: -0.9
--------------------------------------------------------------------------------
Predictions after training:
Input: 0.0 0.0, predicted output: 0.0
Input: 0.0 1.0, predicted output: 1.0
Input: 1.0 0.0, predicted output: 1.0
Input: 1.0 1.0, predicted output: -0.0
--------------------------------------------------------------------------------
```

* **Prediktionerna före träningen varierar mellan körningarna**, till skillnad från **L08**, där de
  alltid var `0`. Startvärdena slumpas, så lagren räknar redan innan träningen, om än på
  slumpmässiga parametrar.
* **Prediktionerna efter träningen matchar XOR-mönstret:** `0`, `1`, `1`, `0`.
* **`-0.0` är inget fel.** Utgångslagret använder tanh och kan ge små negativa värden, som
  `%.1f` skriver ut med minustecken. Avvikelsen från `0` är i storleksordningen någon hundradel.

Övriga targets i [makefilen](./Makefile):

```bash
make build  # Bygger programmet utan att köra det.
make run    # Kör programmet utan att bygga om det.
make clean  # Tar bort den byggda binären.
```

---

## Kör testerna

```bash
make -C test
```

Samtliga 57 testfall går igenom: 15 för stubben från **L06**, 15 för nätverket från **L07** samt 27
för `Dense`, varav 11 fanns redan i **L08**. De 16 nya täcker det som denna lektion lägger till:
* **Randomiseringen:** `ConstructedParametersAreRandomized`, `ConstructedParametersCoverBothSigns`
  samt `LayersAreIndependentlyRandomized`.
* **Feedforward:** `FeedforwardComputesWeightedSum`, `FeedforwardAppliesActivationFunction`,
  `DefaultActivationFunctionIsRelu`, `FeedforwardIsDeterministic` samt `FeedforwardDoesNotTrain`.
* **Backpropagation:** `BackpropagateOutputLayerComputesError`,
  `BackpropagateUsesPreActivationDerivative`, `BackpropagateNoneUsesUnitDerivative` samt
  `BackpropagateHiddenLayerComputesError`.
* **Optimering:** `OptimizeUpdatesBiasAndWeights` samt `OptimizeRejectedLeavesParametersUnchanged`.
* **Helheten:** `ComputesThroughInterface` samt `NetworkLearnsXorPattern`, som tränar ett komplett
  nätverk och kontrollerar att det lär sig mönstret.

Eftersom startvärdena slumpas jämför testerna mot värden som räknas fram ur lagrets egna vikter,
i stället för mot fasta tal.

---

## Svar på frågorna i avsnitt 9
Siffrorna nedan kommer från 30 körningar per inställning, med olika frön till slumptalsgeneratorn.
En körning räknas som lyckad då samtliga fyra prediktioner ligger inom `0.3` från sitt
referensvärde. Egna körningar ger inte exakt samma siffror, men samma mönster.

**1. Hur hänger lärhastigheten och antalet epoker ihop?**
Varje träningssteg flyttar parametrarna en sträcka proportionell mot lärhastigheten, så de två
parametrarna byter mot varandra: halverad lärhastighet kräver ungefär dubbelt så många epoker för
att komma lika långt. Behålls antalet epoker blir nätverket helt enkelt mindre färdigtränat. Med 10
dolda noder och 500 epoker gav lärhastigheten `0.05` ett genomsnittligt största fel på `0.25`,
medan `0.025` gav `0.30`; med dubblerat antal epoker var `0.025` tillbaka på `0.22`.

**2. Vad händer när lärhastigheten blir för hög?**
Stegen blir för stora och skjuter förbi minimum, varpå felet studsar fram och tillbaka i stället
för att minska. Träningen blir både osäkrare och mindre exakt: med 10 dolda noder och 10 000 epoker
lyckades 26 av 30 körningar med lärhastigheten `0.05`, 23 med `0.1` och 17 med `0.9`. Fler epoker
hjälper inte, eftersom problemet är stegens storlek och inte deras antal; 3 dolda noder med
lärhastigheten `0.5` lyckades bara 3 gånger av 30 trots 100 000 epoker. En för hög lärhastighet kan
dessutom slå ut ReLU-noder permanent, genom att trycka ner deras viktade summa under noll för
samtliga insignaler. Se [lösningsförslaget till **L05**](../../L05/appendix/c_solutions.md) för det
resonemanget i sin helhet.

**3. Hur påverkar antalet dolda noder hur ofta träningen lyckas?**
Fler noder ger fler oberoende startpunkter, och det räcker att några av dem hamnar bra. XOR kräver
minst två användbara dolda noder; med bara 3 noder är en olycklig dragning, eller en nod som dör
under träningen, ofta nog för att träningen ska misslyckas. Med lärhastigheten `0.05` och 10 000
epoker lyckades 11 av 30 körningar med 3 dolda noder, 25 med 6 noder och 26 med 10 noder. Priset är
mer beräkning per epok, och ett större nätverk än nödvändigt riskerar dessutom att anpassa sig till
träningsdatan i stället för till mönstret; för XOR, där träningsdatan *är* hela sanningstabellen,
märks det senare inte.

---

## Noteringar om implementationen
* **Träningen kan misslyckas, och det är inte ett kodfel.** Även med de rekommenderade
  inställningarna lyckas inte varje körning. Kör om programmet, och vänta minst en sekund emellan,
  eftersom slumptalsgeneratorn seedas med tiden i hela sekunder; två körningar inom samma sekund
  får identiska startvärden och därmed identiskt resultat.
* **`default` i switch-satserna** täcker både `ActFunc::None` och eventuella ogiltiga värden.
  `ActFunc::None` är identitet, så utdatan är summan oförändrad och derivatan `1.0`; ett ogiltigt
  värde får därmed ett definierat beteende i stället för ett odefinierat, utan någon extra gren.
* **`std::rand()` räcker här, men inte överallt.** Fördelningen är jämn nog för startvärden, men
  generatorn är svag och delas av hela programmet. I skarp kod används `<random>`, exempelvis
  `std::mt19937` tillsammans med `std::uniform_real_distribution`, vilket dessutom gör det möjligt
  att ge varje lager en egen generator.
* **Intervallet `[-1.0, 1.0]` är ett medvetet enkelt val.** Riktiga nätverk skalar intervallet efter
  antalet insignaler per nod (Xavier eller He), så att den viktade summan inte växer med lagrets
  bredd. Se [bilaga A](../appendix/a_math_to_code.md).
* **Lagret validerar fortfarande varje anrop.** Kontrollerna från **L08** ligger kvar först i varje
  metod, och beräkningen sker efter dem. Ett avvisat anrop returnerar `false` utan att röra en enda
  medlemsvariabel, vilket testfallet `OptimizeRejectedLeavesParametersUnchanged` kontrollerar.

---

## Nästa steg
* **L10:** Övningstentamen: implementering av neurala nätverk samt tester i C++.

---
