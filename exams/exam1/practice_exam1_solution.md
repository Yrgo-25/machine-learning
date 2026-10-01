# Övningstentamen 1 - Lösningsförslag
Lösningsförslag till [övningstentamen 1](./practice_exam1.md).

**OBS!** Dokumentet avslöjar samtliga sex buggar. Gör ert eget försök först.

Dokumentet är upplagt så att ni kan stämma av en bugg i taget: varje avsnitt nedan är fristående
och innehåller en bugg, dess rättning och det testfall som fångar den. Läs enbart avsnittet för den
metod ni arbetar med, så spolieras inte de övriga. Vill ni kontrollera er kod utan att läsa facit
alls, använd i stället frågorna under *Självkontroll* i
[övningstentamen](./practice_exam1.md). En sammanställning av samtliga sex buggar finns sist i
dokumentet.

---

## Testfallens gemensamma delar
Samtliga testfall nedan ligger i samma anonyma namnrymd och delar konstanterna nedan, så att inga
lösa tal står mitt i testkoden:

```cpp
/** Tolerance for comparisons of computed floating-point values. */
constexpr double Tolerance{1e-9};

/** Layer dimensions used by the test cases. */
constexpr std::uint16_t HiddenCount{3U};
constexpr std::uint16_t NodeCount{2U};
constexpr std::uint16_t WeightCount{2U};
constexpr std::uint16_t NextNodeCount{3U};
constexpr std::uint16_t OutputCount{2U};

/** Learning rate within the valid range (0.0, 1.0). */
constexpr double LearningRate{0.1};

/** Learning rates outside the valid range (0.0, 1.0). */
constexpr double UpperBoundRate{1.0};
constexpr double TooHighRate{1.5};

/** Number of epochs used when training is expected to run. */
constexpr std::uint32_t EpochCount{10U};

/** Output values assigned to the output layer stub. */
const Matrix1d KnownOutput{0.5, -0.25};
```

Lagrets samtliga parametrar går att läsa av: `bias()` ger biasvärdena och `weights()` vikterna, så
ett förväntat värde kan räknas ut direkt ur lagret utan att köra någon av de metoder som testas.

Indatavektorerna deklareras i respektive testfall. De är `const` snarare än `constexpr`, då
`std::vector` inte går att deklarera `constexpr` i C++17.

---

## Bugg 1: Feedforward använder fel nods vikter
**Ursprunglig kod** i `Dense::feedforward()`:

```cpp
sum += myWeights[0U][j] * input[j];
```

**Rättning:**

```cpp
sum += myWeights[i][j] * input[j];
```

**Testfallet som fångar buggen:**

```cpp
TEST(Dense, FeedforwardUsesTheWeightsOfEachNode)
{
    const Matrix1d input{1.0, 2.0};

    Dense layer{HiddenCount, WeightCount, ActFunc::None};
    const auto bias = layer.bias();

    EXPECT_TRUE(layer.feedforward(input));

    for (std::uint16_t i{}; i < layer.nodeCount(); ++i)
    {
        auto expected = bias[i];

        for (std::uint16_t j{}; j < layer.weightCount(); ++j)
        {
            expected += layer.weights()[i][j] * input[j];
        }
        EXPECT_NEAR(layer.output()[i], expected, Tolerance);
    }
}
```

Testet räknar ut hela den viktade summan för varje nod, ur `bias()` och `weights()`, och jämför mot
lagrets utdata. Med den felaktiga koden får samtliga noder nod 0:s summa, så alla utom den första
avviker. `ActFunc::None` används för att utdatan ska vara summan själv, utan filtrering.

**Påverkan på träningen:** Samtliga noder i lagret beräknar samma viktade summa och skiljer sig
därmed bara åt genom sina biasvärden. Backpropagation och optimering uppdaterar visserligen varje
nods egna vikter, men de vikterna används sedan aldrig i feedforward, så lagret beter sig som en
enda nod oavsett hur många noder det har. Ett nätverk som ska lära sig XOR, vilket kräver minst två
oberoende dolda noder, kan då aldrig lyckas.

---

## Bugg 2: Avvikelsen har omvänt tecken
**Ursprunglig kod** i `Dense::backpropagate(const Matrix1d& output)`:

```cpp
const auto deviation = myOutput[i] - output[i];
```

**Rättning:**

```cpp
const auto deviation = output[i] - myOutput[i];
```

**Testfallet som fångar buggen:**

```cpp
TEST(Dense, BackpropagateComputesReferenceMinusOutput)
{
    const Matrix1d input{1.0, 2.0};
    const Matrix1d reference{1.0, 0.0};

    Dense layer{NodeCount, WeightCount, ActFunc::None};
    EXPECT_TRUE(layer.feedforward(input));

    const auto prediction = layer.output();
    EXPECT_TRUE(layer.backpropagate(reference));

    for (std::uint16_t i{}; i < layer.nodeCount(); ++i)
    {
        EXPECT_NEAR(layer.error()[i], reference[i] - prediction[i], Tolerance);
    }
}
```

**Påverkan på träningen:** Formeln $\delta = y_{ref} - y_p$ från **L05** blir $y_p - y_{ref}$, så
felet pekar bort från referensvärdet i stället för mot det. Varje optimeringssteg gör därmed
prediktionen sämre, och parametrarna växer tills utgångslagrets tanh mättas. Symptomet är att
samtliga prediktioner låser sig vid ytterlägena, ofta `1.0` eller `-1.0`, oavsett indata och
oavsett antal epoker.

---

## Bugg 3: Dolda lagrets fel skalas inte med derivatan
**Ursprunglig kod** i `Dense::backpropagate(const Interface& nextLayer)`:

```cpp
myError[i] = sum;
```

**Rättning:**

```cpp
const auto delta = actFuncDelta(myActFunc, myPreActOutput[i]);
myError[i]       = sum * delta;
```

**Testfallet som fångar buggen:**

```cpp
TEST(Dense, BackpropagateScalesHiddenErrorWithTheDerivative)
{
    const Matrix1d input{1.0, -1.0};
    const Matrix1d reference{1.0, 0.0, -1.0};

    Dense hiddenLayer{NodeCount, WeightCount, ActFunc::Tanh};
    Dense nextLayer{NextNodeCount, NodeCount, ActFunc::None};

    EXPECT_TRUE(hiddenLayer.feedforward(input));
    EXPECT_TRUE(nextLayer.feedforward(hiddenLayer.output()));
    EXPECT_TRUE(nextLayer.backpropagate(reference));

    const auto hiddenOutput = hiddenLayer.output();
    EXPECT_TRUE(hiddenLayer.backpropagate(nextLayer));

    for (std::uint16_t i{}; i < hiddenLayer.nodeCount(); ++i)
    {
        double sum{};

        for (std::uint16_t j{}; j < nextLayer.nodeCount(); ++j)
        {
            sum += nextLayer.error()[j] * nextLayer.weights()[j][i];
        }
        // The derivative of tanh is 1 - tanh(s) * tanh(s), i.e. 1 - output * output.
        const auto delta = 1.0 - hiddenOutput[i] * hiddenOutput[i];
        EXPECT_NEAR(hiddenLayer.error()[i], sum * delta, Tolerance);
    }
}
```

Nästa lager har flera noder, så det dolda lagrets fel är en riktig summa över samtliga anslutna
noder och inte ett enda bidrag. Det dolda lagret måste dessutom ha `ActFunc::Tanh` eller
`ActFunc::Relu`: med `ActFunc::None` är derivatan alltid `1.0`, och då ger både den felaktiga
och den rättade koden exakt samma resultat.

**Påverkan på träningen:** Formeln $\Delta e = \delta * y_p'$ tappar faktorn $y_p'$. En ReLU-nod
vars viktade summa är negativ har derivatan noll och ska därmed inte justeras alls, men får nu ett
fel som om den vore aktiv; en tanh-nod i mättnad, där derivatan är nära noll, får en justering som
är många gånger för stor. Träningen blir instabil och det dolda lagret rör sig åt fel håll, medan
utgångslagret, som inte berörs av buggen, ser ut att fungera.

---

## Bugg 4: Vikterna justeras utan sin insignal
**Ursprunglig kod** i `Dense::optimize()`:

```cpp
myWeights[i][j] += changeRate;
```

**Rättning:**

```cpp
myWeights[i][j] += changeRate * input[j];
```

**Testfallet som fångar buggen:**

```cpp
TEST(Dense, OptimizeScalesEachWeightWithItsOwnInput)
{
    // The two inputs differ, so a weight update that ignores them gives a different result.
    const Matrix1d input{2.0, -1.0};
    const Matrix1d reference{1.0, 0.0};

    Dense layer{NodeCount, WeightCount, ActFunc::None};
    EXPECT_TRUE(layer.feedforward(input));
    EXPECT_TRUE(layer.backpropagate(reference));

    const auto error         = layer.error();
    const auto weightsBefore = layer.weights();
    EXPECT_TRUE(layer.optimize(input, LearningRate));

    for (std::uint16_t i{}; i < layer.nodeCount(); ++i)
    {
        for (std::uint16_t j{}; j < layer.weightCount(); ++j)
        {
            const auto expected = weightsBefore[i][j] + error[i] * LearningRate * input[j];
            EXPECT_NEAR(layer.weights()[i][j], expected, Tolerance);
        }
    }
}
```

Insignalen måste ha olika värden per vikt, som `{2.0, -1.0}`. Med insignalen `{1.0, 1.0}` är
`changeRate * input[j]` lika med `changeRate`, och då går även den felaktiga koden igenom.

**Påverkan på träningen:** Formeln $w_j = w_j + \Delta c_n * y_j$ blir $w_j = w_j + \Delta c_n$.
Samtliga vikter i en nod justeras lika mycket, oberoende av vilken insignal de faktiskt fick, och
en vikt vars insignal var noll justeras trots att den inte bidrog till felet. Nätverket lär sig
därmed inte sambandet mellan enskilda insignaler och utdatan; för XOR, där just kombinationen av
insignalerna avgör, blir träningen meningslös.

---

## Bugg 5: Prediktionen kommer från fel lager
**Ursprunglig kod** i `Shallow::predict()`:

```cpp
return myHiddenLayer.output();
```

**Rättning:**

```cpp
return myOutputLayer.output();
```

**Testfallet som fångar buggen:**

```cpp
TEST(Shallow, PredictReturnsTheOutputLayerOutput)
{
    const Matrix1d input{1.0, 0.0};

    Stub hiddenLayer{HiddenCount, WeightCount};
    Stub outputLayer{OutputCount, HiddenCount};
    Shallow network{hiddenLayer, outputLayer};

    outputLayer.setOutput(KnownOutput);
    const auto& prediction = network.predict(input);

    EXPECT_EQ(prediction.size(), KnownOutput.size());

    for (std::uint16_t i{}; i < outputLayer.nodeCount(); ++i)
    {
        EXPECT_NEAR(prediction[i], KnownOutput[i], Tolerance);
    }
}
```

Stubbarna gör testet oberoende av all matematik: utgångslagret returnerar exakt det som
`setOutput()` satte, så ett fel kan bara bero på nätverket. Utgångslagret har två noder, och
samtliga värden jämförs, så testet fångar även ett nätverk som returnerar rätt antal värden men
fel innehåll.

**Påverkan på träningen:** Träningen påverkas inte alls, eftersom `train()` använder den privata
`feedforward()` och inte `predict()`. Det är enbart avläsningen som är fel, vilket gör buggen lätt
att se i demoprogrammet: prediktionen skrivs ut med ett värde per dold nod i stället för ett enda
värde. Ett nätverk kan alltså vara helt korrekt tränat och ändå redovisa fel svar.

---

## Bugg 6: Lärhastigheten kontrolleras bara nedåt
**Ursprunglig kod** i `inputValid()` i `shallow.cpp`:

```cpp
if (0.0 >= learningRate)
```

**Rättning:**

```cpp
if ((0.0 >= learningRate) || (1.0 <= learningRate))
```

**Testfallet som fångar buggen:**

```cpp
TEST(Shallow, TrainRejectsLearningRateOutsideRange)
{
    const Matrix2d trainIn{{0.0, 0.0}, {1.0, 1.0}};
    const Matrix2d trainOut{{0.0}, {1.0}};

    Stub hiddenLayer{HiddenCount, WeightCount};
    Stub outputLayer{OutputCount, HiddenCount};
    Shallow network{hiddenLayer, outputLayer};

    EXPECT_FALSE(network.train(trainIn, trainOut, EpochCount, UpperBoundRate));
    EXPECT_FALSE(network.train(trainIn, trainOut, EpochCount, TooHighRate));
    EXPECT_TRUE(network.train(trainIn, trainOut, EpochCount, LearningRate));
}
```

Kontraktet står i [shallow.h](./code/include/ml/neural_network/shallow.h): lärhastigheten ska ligga
i intervallet `(0.0, 1.0)`. Testet jämför alltså implementationen mot dokumentationen, inte mot sig
själv.

**Påverkan på träningen:** En lärhastighet på `1.0` eller mer accepteras av nätverket, varpå
`Dense::optimize()` avvisar varje enskilt anrop och returnerar `false`. Eftersom `Shallow` inte
kontrollerar lagrens returvärden fortsätter träningen ändå, epok efter epok, utan att en enda
parameter uppdateras, och `train()` returnerar till sist `true`. Symptomet är alltså det värsta
tänkbara: en träning som rapporterar framgång trots att ingenting har hänt.

---

## Uppgift 2: argumentkontroller
Kontrollerna i `Dense` är korrekta i den utlämnade koden, så dessa testfall går igenom direkt;
uppgiften är oberoende av buggjakten. Exempeltestet `Dense.FeedforwardChecksInputSize` täcker
`feedforward()`, så de efterfrågade testfallen är:

```cpp
TEST(Dense, BackpropagateChecksReferenceSize)
{
    Dense layer{NodeCount, WeightCount};
    EXPECT_FALSE(layer.backpropagate(Matrix1d(NodeCount - 1U, 1.0)));
    EXPECT_FALSE(layer.backpropagate(Matrix1d(NodeCount + 1U, 1.0)));
    EXPECT_TRUE(layer.backpropagate(Matrix1d(NodeCount, 1.0)));
}

TEST(Dense, BackpropagateChecksNextLayerDimensions)
{
    Dense layer{NodeCount, WeightCount};
    Dense mismatched{NodeCount, static_cast<std::uint16_t>(NodeCount + 1U)};
    Dense matching{NodeCount, NodeCount};

    EXPECT_FALSE(layer.backpropagate(mismatched));
    EXPECT_TRUE(layer.backpropagate(matching));
}

TEST(Dense, OptimizeChecksArguments)
{
    const Matrix1d validInput(WeightCount, 1.0);
    Dense layer{NodeCount, WeightCount};

    EXPECT_FALSE(layer.optimize(validInput, 0.0));
    EXPECT_FALSE(layer.optimize(validInput, 1.0));
    EXPECT_FALSE(layer.optimize(validInput, 1.5));
    EXPECT_FALSE(layer.optimize(Matrix1d(WeightCount + 1U, 1.0), LearningRate));
    EXPECT_TRUE(layer.optimize(validInput, LearningRate));
}
```

Godkänn även andra uppdelningar, exempelvis ett testfall per kontroll, så länge samtliga fyra
metoders avvisande och accepterande vägar täcks. `Shallow::train()` saknar däremot en av sina
kontroller; den hör till bugg 6 och bedöms där.

---

## Sammanställning av buggarna

| Nr | Fil | Rad | Bugg |
|---|---|---|---|
| 1 | `source/ml/dense_layer/dense.cpp` | 85 | `feedforward()` läser vikterna ur nod 0 för samtliga noder. |
| 2 | `source/ml/dense_layer/dense.cpp` | 111 | `backpropagate()` (utgångslager) beräknar avvikelsen med omvänt tecken. |
| 3 | `source/ml/dense_layer/dense.cpp` | 142 | `backpropagate()` (dolt lager) skalar inte felet med aktiveringsfunktionens derivata. |
| 4 | `source/ml/dense_layer/dense.cpp` | 177 | `optimize()` justerar varje vikt utan att skala med viktens egen insignal. |
| 5 | `source/ml/neural_network/shallow.cpp` | 109 | `predict()` returnerar det dolda lagrets utdata i stället för utgångslagrets. |
| 6 | `source/ml/neural_network/shallow.cpp` | 67 | `train()` saknar den övre gränsen för lärhastigheten. |

Radnumren avser den utlämnade koden, innan någon rättning har gjorts.

---

## Uppgift 6 och 7: edge cases samt konvergens (VG)
Båda uppgifterna är oberoende av buggjakten och går igenom även med den utlämnade koden, med ett
undantag: konvergenstestet misslyckas tills buggarna i `Dense` är rättade.

**Edge cases.** Ett lager med en enda nod och en enda vikt, `Dense{1U, 1U}`, ska bete sig som
vilket lager som helst; `nodeCount()` och `weightCount()` ska vara 1, och beräkningarna ska stämma
mot samma formler. Träningsdata med olika många indata och utdata ska använda det minsta antalet:
`train()` med tre indata och två utdata ska träna på två uppsättningar, returnera `true` och inte
läsa utanför datan.

**Konvergens.** Ett testfall som kräver att nätverket lär sig XOR-mönstret varje gång är inte
pålitligt: med tio dolda noder, lärhastigheten `0.05` och 1000 epoker lär sig nätverket mönstret i
ungefär åtta fall av tio, vilket betyder att ett sådant testfall slår fel var femte körning trots
korrekt kod. Två formuleringar som däremot höll i samtliga körningar vid provkörning:
* **Felet minskade:** medelabsolutfelet efter träningen är lägre än före, vilket höll i 200 av 200
  körningar.
* **Flera nätverk, majoriteten lyckas:** träna fem nätverk och kräv att minst tre av dem når ett
  medelabsolutfel under `0.1`, vilket höll i 40 av 40 grupper.

Godkänn vilken formulering som helst som är robust mot startvärdena, och underkänn den som kräver
att en enskild körning lyckas. Att studenten kan motivera valet är hela poängen med uppgiften.

---

## Kontroll efter rättning
Kör testerna i katalogen [code/test](./code/test):

```bash
make
```

Samtliga testfall ska gå igenom, exempeltesterna inkluderade. Kör därefter demoprogrammet i
katalogen [code](./code):

```bash
make
```

Nätverket ska nu prediktera XOR-mönstret, dvs. `{0.0}`, `{1.0}`, `{1.0}` och `{0.0}`. Startvärdena
slumpas, så någon enstaka körning kan ändå misslyckas; kör om programmet, och vänta minst en sekund
emellan, eftersom slumptalsgeneratorn seedas med tiden i hela sekunder.

---
