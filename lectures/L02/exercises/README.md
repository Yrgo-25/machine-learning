# L02 - Lösningsförslag: Linjär regression i C++
Linjär regressionsmodell som predikterar enligt `y = kx + m` och tränas med en fast lärhastighet.

Koden är uppdelad i ett interface, `ml::lin_reg::Interface`, samt en underklass, `ml::lin_reg::Fixed`, som implementerar interfacet.

---

## Struktur
```
exercises/
├── Makefile
├── include/
│   └── ml/
│       ├── lin_reg/
│       │   ├── fixed.h        # Klassen Fixed, deklaration.
│       │   └── interface.h    # Interfacet Interface.
│       └── types.h            # Matrisalias: Matrix1d, Matrix2d, Matrix3d.
└── source/
    ├── main.cpp               # Demo: tränar en modell för y = 2x + 2.
    └── ml/
        └── lin_reg/
            └── fixed.cpp      # Klassen Fixed, implementation.
```

---

## Interfacet `ml::lin_reg::Interface`
Interfacet definierar vad varje linjär regressionsmodell måste kunna, oavsett hur den tränas:
* **`~Interface()`:** Virtuell destruktor, så att en modell kan raderas via en referens eller pekare till interfacet.
* **`predict(input)`:** Rent virtuell prediktionsmetod, markerad `const` samt `noexcept`.

Nyttan syns i `main.cpp`: funktionen `testModel()` tar emot en `const ml::lin_reg::Interface&` och behöver aldrig veta vilken sorts modell den predikterar med. Samma funktion fungerar därför oförändrad för `Adaptive`, den modell med adaptiv lärhastighet som ni implementerar under **P03**.

---

## Klassen `ml::lin_reg::Fixed`
Ärver `Interface` via publikt arv och är markerad `final`, så att den inte kan ärvas vidare. Klassen döps till `Fixed` eftersom lärhastigheten är fast under hela träningen.

### Medlemsvariabler

| Variabel | Typ | Innehåll |
|---|---|---|
| `myTrainIn` | `const Matrix1d&` | Referens till träningsdatans indata. |
| `myTrainOut` | `const Matrix1d&` | Referens till träningsdatans utdata. |
| `mySetCount` | `const std::size_t` | Antalet fullständiga träningsuppsättningar. |
| `myBias` | `double` | Modellens biasvärde, dvs. m i `y = kx + m`. |
| `myWeight` | `double` | Modellens viktvärde, dvs. k i `y = kx + m`. |

Träningsdatan lagras som referenser i stället för kopior; modellen äger inte datan. Det är därmed den som skapar modellen som ansvarar för att träningsdatan finns kvar under hela träningen, och att den inte ändras under tiden.

`mySetCount` sätts till `std::min(trainIn.size(), trainOut.size())`. Träningsdata med fler indata än utdata, eller tvärtom, saknar fullständiga uppsättningar bortom den kortaste av de två vektorerna, så överskottsvärdena är oanvändbara.

### Metoder
* **`Fixed(trainIn, trainOut)`:** Initierar samtliga medlemsvariabler, där `myBias` och `myWeight` båda sätts till 0.0; modellen predikterar alltså 0 för varje indata tills den har tränats. Om `mySetCount` är 0 skrivs ett felmeddelande ut till `stderr`, varpå `std::terminate()` anropas: en modell utan träningsdata kan aldrig tränas, och konstruktorn kan inte returnera någon felkod till anroparen.
* **`predict(input)`:** Returnerar `myWeight * input + myBias`.
* **`train(epochCount, learningRate)`:** Returnerar `false` direkt om `epochCount` är 0, eller om `learningRate` ligger utanför intervallet `(0.0, 1.0)`; till skillnad från konstruktorn kan `train()` rapportera ogiltiga argument via sitt returvärde. Annars tränas modellen i angivet antal epoker, och `true` returneras.
* **`optimize(input, output, learningRate)`:** Privat metod som justerar `myBias` samt `myWeight` för en enskild träningsuppsättning.

Defaultkonstruktorn samt kopierings- och förflyttningskonstruktorerna med tillhörande operatorer är raderade. En modell är knuten till den träningsdata den skapades med, och referensmedlemmarna går inte att tilldela om i efterhand.

---

## Träningsalgoritmen
`train()` itererar genom samtliga träningsuppsättningar en gång per epok, och anropar `optimize()` för varje uppsättning. Optimeringen består av tre steg:

1. Prediktera med aktuell indata: `prediction = myWeight * input + myBias`.
2. Beräkna felet som differensen mellan referensvärdet och prediktionen: `error = output - prediction`.
3. Justera modellens parametrar utefter felet:
    * `myBias   += error * learningRate`
    * `myWeight += error * learningRate * input`

Lärhastigheten avgör hur stor del av felet som korrigeras i varje steg. Ett för lågt värde ger långsam träning, medan ett värde på 1.0 eller mer korrigerar med minst hela felet varje gång, vilket får träningen att oscillera eller divergera i stället för att konvergera. Därav intervallkontrollen i `train()`.

---

## Bygg och kör
Bygg och kör programmet via följande kommando i denna katalog:

```bash
make
```

Övriga targets i [makefilen](./Makefile):

```bash
make build  # Bygger programmet utan att köra det.
make run    # Kör programmet utan att bygga om det.
make clean  # Tar bort den byggda binären.
```

`main.cpp` tränar en modell på `y = 2x + 2` i 60 epoker med lärhastigheten 0.22, och predikterar sedan med varje indata i träningsdatan. Utskriften ska se ut såsom visas nedan:

```
----------------------------------------
Input: 0, output: 2
Input: 1, output: 4
Input: 2, output: 6
Input: 3, output: 8
Input: 4, output: 10
----------------------------------------
```

---

## Att notera
* Namnen på medlemsvariabler och lokala variabler behöver inte stämma exakt med bilaga B; här används exempelvis `myTrainIn` och `myTrainOut` i stället för `myTrainInput` och `myTrainOutput`. Det är beteendet som ska stämma.
* Antalet epoker och lärhastigheten i `main.cpp` är fritt valda. Fler epoker med en lägre lärhastighet ger samma resultat, men tar längre tid.
* Specialfallet då indatan är noll hanteras inte här; det behandlas i **L03**.

---

## Nästa steg
* **L03:** Optimering då indatan är noll, samt randomiserad träningsordning inför varje epok.
* **P03:** `ml::lin_reg::Adaptive`, en modell med adaptiv lärhastighet, i ett inbyggt system.

---
