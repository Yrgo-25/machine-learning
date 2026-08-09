# Komplett konvolutionellt neuralt nätverk
Detta är en komplett, fungerande C++17-implementation av ett konvolutionellt neuralt nätverk (CNN)
som tränas för att klassificera fyra handritade 4×4-pixelmönster (siffrorna 0–3).

Nedan visas de fyra träningsmönstren ([source/main.cpp](./source/main.cpp)), där 1 är en fylld pixel och 0 är tom:

```
Siffra 0:      Siffra 1:      Siffra 2:      Siffra 3:
1 1 1 1        0 1 0 0        1 0 0 1        0 0 1 0
1 0 0 1        0 1 0 0        0 1 1 0        0 0 1 0
1 0 0 1        0 1 0 0        0 1 1 0        1 1 1 1
1 1 1 1        0 1 0 0        1 0 0 1        0 0 1 0
```

---

## Struktur

```
cnn_demo/
├── Makefile
├── source/
│   ├── main.cpp
│   └── ml/
│       ├── cnn/cnn.cpp
│       ├── conv_layer/{conv,max_pool}.cpp
│       ├── dense_layer/dense.cpp
│       ├── flatten_layer/flatten.cpp
│       ├── factory/factory.cpp
│       ├── random/generator.cpp
│       └── utils.cpp
└── include/ml/
    ├── types.h, utils.h
    ├── act_func/{interface,relu,tanh,none,type}.h
    ├── cnn/{interface,cnn}.h
    ├── conv_layer/{interface,conv,max_pool,stub}.h
    ├── dense_layer/{interface,dense,stub}.h
    ├── flatten_layer/{interface,flatten,stub}.h
    ├── factory/{interface,factory,stub}.h
    └── random/{interface,generator}.h
```

Varje lagertyp (`conv_layer`, `dense_layer`, `flatten_layer`) följer samma mönster: ett `Interface`
som definierar kontraktet, en konkret klass som gör den riktiga beräkningen (`Conv`, `MaxPool`,
`Dense`, `Flatten`), samt en `Stub`-variant som bara validerar dimensioner och returnerar
`true`/`false` utan att räkna något. `ml::factory::Factory` bygger de riktiga lagren,
`ml::factory::Stub` bygger stub-varianterna - vilken som används väljs vid körning via
`ml::factory::create(bool stub = false)`, inte via kompileringsflaggor.

**Datatyper** (`ml/types.h`): `Matrix1d = std::vector<double>`, `Matrix2d = std::vector<Matrix1d>`,
`Matrix3d = std::vector<Matrix2d>`.

---

## Lagren

### `conv_layer::Conv`
En enkel conv-lagerimplementation med en enda kernel och en bias, med "same"-padding (utdatan får
samma spatiala storlek som indatan).
* **Konstruktor:** tar emot indatastorlek, kernelstorlek (1–11) och en aktiveringsfunktion.
  Initierar kernel och bias med slumpmässiga startvärden.
* **`feedforward()`:** nollpaddar indatan, och beräknar för varje utdatacell
  `summa = bias + Σ kernel[i][j] * paddad_indata[...]`, följt av aktiveringsfunktionen.
* **`backpropagate()`:** tar emot felgradienter från nästa lager, ackumulerar bias- och
  kernelgradienter samt beräknar gradienter för den paddade indatan, som sedan "avpaddas" till
  `inputGradients()` för föregående lager.
* **`optimize()`:** uppdaterar kernel och bias med `gradient * lärhastighet`.

### `conv_layer::MaxPool`
Icke-överlappande maxpooling, definierad i samma namnrymd (`ml::conv_layer`) eftersom den delar
interface med `Conv`.
* **Konstruktor:** tar emot indatastorlek och poolstorlek (indatastorleken måste vara jämnt delbar
  med poolstorleken).
* **`feedforward()`:** väljer maxvärdet i varje icke-överlappande block.
* **`backpropagate()`:** letar upp vilken cell i respektive block som gav maxvärdet vid
  feedforward, och routar hela felgradienten dit - övriga celler får ingen gradient.
* **`optimize()`:** gör ingenting (returnerar bara `true`) - lagret saknar tränbara parametrar.

### `flatten_layer::Flatten`
Formar om en 2D-matris till en 1D-vektor (och tillbaka vid backpropagation). Saknar tränbara
parametrar och har därför ingen `optimize()`-metod alls i sitt interface.

### `dense_layer::Dense`
Ett vanligt fullt uppkopplat lager med vikter och bias.
* **`feedforward()`:** `summa = bias[i] + Σ vikt[i][j] * indata[j]` per nod, följt av
  aktiveringsfunktionen.
* **`backpropagate()`:** beräknar `fel[i] = (outputGradients[i] - myOutput[i]) *
  aktiveringsfunktion.delta(föraktiveringssumma[i])`. Se anteckning nedan om hur
  `outputGradients` tolkas olika beroende på om lagret är utgångslagret eller ett dolt lager.
* **`optimize()`:** uppdaterar vikter och bias med `fel * lärhastighet * indata`.

---

## `ml::cnn::Cnn` - hur lagren hänger ihop
Konstruktorn skapar och kopplar ihop lagren via den injicerade fabriken:
1. Ett `Conv`-lager och ett `MaxPool`-lager läggs i samma lista, `myConvLayers` (i den ordningen).
2. Ett `Flatten`-lager skapas utifrån maxpool-lagrets utdatastorlek.
3. Ett `Dense`-lager skapas utifrån flatten-lagrets utdatastorlek.

**`feedforward(input)`:** `input → Conv → MaxPool → Flatten → Dense → prediktion`. Varje lager
matas med föregående lagers `output()`.

**`backpropagate(target)`:** går samma väg baklänges. Målvektorn (`target`) skickas direkt in i det
sista dense-lagrets `backpropagate()` - det är där `(outputGradients[i] - myOutput[i])` faktiskt
blir `(mål[i] - prediktion[i])`. Varje tidigare lager matas sedan med nästa lagers
`inputGradients()`: dense-lagren i omvänd ordning, sedan flatten-lagret, sedan `MaxPool` och
till sist `Conv`.

**`optimize(lärhastighet)`:** uppdaterar samtliga lager med de gradienter som redan beräknats av
`backpropagate()`. Dense-lagren behöver även föregående lagers `output()` (det värde som faktiskt
användes vid feedforward) som indata till viktuppdateringen.

**`train(trainIn, trainOut, epochCount, lärhastighet)`:** ren stokastisk gradientnedstigning
(batchstorlek 1, ingen mini-batching). Träningsordningen slumpas om för varje epok. Om
`feedforward`, `backpropagate` eller `optimize` misslyckas för något exempel avbryts träningen
omedelbart.

**`predict(input)`:** kör `feedforward()` och returnerar sista dense-lagrets `output()`.

---

## Aktiveringsfunktioner

| Funktion | `output(x)` | `delta(x)` |
|---|---|---|
| `Relu` | `x` om `x > 0`, annars `0` | `1` om `x > 0`, annars `0` |
| `Tanh` | `tanh(x)`, intervall `[-1, 1]` | `1 - tanh(x)²` |
| `None` | `x` (identitet) | `1` |

`delta()` beräknas alltid utifrån föraktiveringssumman (värdet innan aktiveringsfunktionen
applicerades), inte utifrån det redan aktiverade utdatavärdet.

---

## Slumptalsgenerator
`random::Generator` är en singleton som initierar `std::srand` en gång vid första användning.
Används dels för att slumpa initiala vikt- och biasvärden (via `randomStartVal()` i `Conv` och
`Dense`), dels för att slumpa om träningsordningen varje epok.

---

## Bygg och kör

```bash
make
```

Detta bygger och kör programmet direkt (`build`-target följt av `run`-target). `clean` städar bort
den byggda binären.

---

## Implementationsdetaljer värda att känna till
* **`-DSTUB`-flaggan i Makefilen är för närvarande utan effekt.** Kommentaren i Makefilen antyder
  att den byter till stub-implementationerna, men ingen `#ifdef STUB` finns någonstans i koden som
  reagerar på flaggan. Stub-klasserna existerar och går att använda (via
  `ml::factory::create(true)`), men `main.cpp` anropar alltid `ml::factory::create()` utan
  argument, vilket ger den riktiga implementationen. I den här demon används alltså aldrig
  stub-lagren.
* **`Dense::backpropagate()` återanvänder samma formel för utgångslagret och dolda lager.** För det
  sista dense-lagret är `outputGradients` bokstavligen målvektorn, så
  `outputGradients[i] - myOutput[i]` blir `mål - prediktion`. För tidigare dense-lager är det som
  skickas in i stället föregående lagers redan uträknade `inputGradients()` - alltså inte en
  målvektor. Samma formel tillämpas ändå på båda fallen. Det är så koden faktiskt fungerar, inte
  ett fel som behöver rättas till - men värt att känna till när man läser eller resonerar kring
  backpropagation-koden.

  ---
