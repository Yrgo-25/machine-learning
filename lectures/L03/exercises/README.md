# L03 - Lösningsförslag: Randomiserad linjär regression i C++
Linjär regressionsmodell som predikterar enligt `y = kx + m` och tränas med en fast lärhastighet, nu med hantering av indatan noll samt randomiserad träningsordning.

Koden bygger vidare på [lösningsförslaget från **L02**](../../L02/exercises/README.md). Interfacet `ml::lin_reg::Interface` är oförändrat; all ny funktionalitet ligger i underklassen `ml::lin_reg::Fixed`.

---

## Ändringar jämfört med L02
| Uppgift | Ändring |
|---|---|
| 1 | `optimize()` sätter biasvärdet direkt då indatan är noll. |
| 2 | Nytt alias `MatrixU32` i `ml/types.h`. |
| 3 | Ny funktion `initRandom()` i en anonym namnrymd i `fixed.cpp`. |
| 4 | Ny medlemsvariabel `myTrainOrder`; `mySetCount` har tagits bort. |
| 5 | Ny privat metod `shuffle()`. |
| 6 | `train()` blandar träningsordningen inför varje epok och itererar via `myTrainOrder`. |

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
│       └── types.h            # Matrisalias: Matrix1d, Matrix2d, Matrix3d, MatrixU32.
└── source/
    ├── main.cpp               # Demo: tränar en modell för y = 2x + 2.
    └── ml/
        └── lin_reg/
            └── fixed.cpp      # Klassen Fixed, implementation.
```

---

## Aliaset `MatrixU32`
`ml/types.h` utökas med ett alias för `std::vector<std::uint32_t>`:

```cpp
/** Matrix holding unsigned integers. */
using MatrixU32 = std::vector<std::uint32_t>;
```

Aliaset används för träningsordningen, som består av index i träningsdatan; index är alltid heltal utan tecken, till skillnad från själva träningsdatan, som lagras som flyttal i `Matrix1d`. Headerfilen `<cstdint>` inkluderas för `std::uint32_t`.

---

## Klassen `ml::lin_reg::Fixed`
Klassen ärver fortfarande `Interface` via publikt arv och är markerad `final`. Namnet `Fixed` syftar alltjämt på att lärhastigheten är fast under hela träningen; det är träningsordningen, inte lärhastigheten, som randomiseras.

### Medlemsvariabler

| Variabel | Typ | Innehåll |
|---|---|---|
| `myTrainOrder` | `MatrixU32` | Index till träningsuppsättningarna, i den ordning de ska tränas. |
| `myTrainIn` | `const Matrix1d&` | Referens till träningsdatans indata. |
| `myTrainOut` | `const Matrix1d&` | Referens till träningsdatans utdata. |
| `myBias` | `double` | Modellens biasvärde, dvs. m i `y = kx + m`. |
| `myWeight` | `double` | Modellens viktvärde, dvs. k i `y = kx + m`. |

`myTrainOrder` är den enda av medlemsvariablerna som modellen äger; träningsdatan lagras precis som i **L02** som referenser, så det är den som skapar modellen som ansvarar för att datan finns kvar och är oförändrad under hela träningen.

Medlemsvariabeln `mySetCount` från **L02** har tagits bort. `myTrainOrder` innehåller exakt ett index per fullständig träningsuppsättning, så `myTrainOrder.size()` ger samma information; att lagra antalet på två ställen innebär bara en risk att de hamnar i otakt.

### Metoder
* **`Fixed(trainIn, trainOut)`:** Beräknar antalet fullständiga träningsuppsättningar som `std::min(trainIn.size(), trainOut.size())`, skriver ut ett felmeddelande till `stderr` och anropar `std::terminate()` om antalet är 0, fyller därefter `myTrainOrder` med indexen `0, 1, 2 ... N-1` och initierar slumptalsgeneratorn via `initRandom()`. `myBias` och `myWeight` sätts båda till 0.0.
* **`predict(input)`:** Returnerar `myWeight * input + myBias`, oförändrad från **L02**.
* **`train(epochCount, learningRate)`:** Returnerar `false` direkt om `epochCount` är 0, eller om `learningRate` ligger utanför intervallet `(0.0, 1.0)`. Annars blandas träningsordningen inför varje epok, varpå samtliga träningsuppsättningar tränas i den blandade ordningen.
* **`optimize(input, output, learningRate)`:** Privat metod som justerar `myBias` samt `myWeight` för en enskild träningsuppsättning, nu med indatan noll som specialfall.
* **`shuffle()`:** Privat metod som blandar `myTrainOrder` i slumpvis ordning.

Defaultkonstruktorn samt kopierings- och förflyttningskonstruktorerna med tillhörande operatorer är fortsatt raderade.

---

## Optimering då indatan är noll
Då indatan är noll gäller `y = k * 0 + m = m`. Referensvärdet *är* alltså biasvärdet, och inget värde på viktvärdet påverkar en prediktion gjord med indatan noll:

```cpp
// m == yref if x == 0.
if (0.0 == input)
{
    myBias = output;
    return;
}
```

Två saker vinns på detta:
* **Snabbare konvergens:** Biasvärdet sätts direkt i stället för att närma sig referensvärdet stegvis, en bråkdel av felet per epok.
* **Ingen onödig beräkning:** Viktvärdets korrigering `error * learningRate * input` blir ändå noll då indatan är noll, så viktvärdet lämnas orört.

Effekten syns i `main.cpp`: 20 epoker räcker nu för att träna modellen till samma resultat som krävde 60 epoker i **L02**.

---

## Randomiserad träningsordning
Tränas modellen alltid i samma ordning riskerar den att anpassa sig till just den sekvensen; den lär sig mönstret i ordningen snarare än i datan. Genom att blanda ordningen inför varje epok exponeras modellen för datan i varierande ordning, vilket ger en mer robust träning. För en linjär modell är vinsten begränsad, eftersom dess felyta är kvadratisk med exakt ett minimum; nyttan blir betydligt större i de djupare nätverken från **L05** och framåt, där blandningen även hjälper parametrarna att ta sig ur dåliga områden av felytan.

### Initiering av slumptalsgeneratorn
`initRandom()` implementeras som en fri funktion i en anonym namnrymd i `fixed.cpp`; den rör inga medlemsvariabler, och den anonyma namnrymden gör den osynlig utanför den egna filen:

```cpp
void initRandom() noexcept
{
    // Only initialize the random generator once.
    static bool initialized{false};
    if (initialized) { return; }

    // Initialize the random generator with the current timestamp as seed.
    std::srand(std::time(nullptr));
    initialized = true;
}
```

Den statiska lokala variabeln `initialized` skapas en enda gång, första gången funktionen anropas, och behåller sitt värde mellan anropen. Därmed initieras slumptalsgeneratorn precis en gång per programkörning, oavsett hur många modeller som skapas. Det är önskvärt av två skäl:
* `std::srand()` nollställer generatorns tillstånd; anropas den gång på gång startar talföljden om från början i stället för att fortsätta.
* Fröet `std::time(nullptr)` har en upplösning på en sekund; två initieringar inom samma sekund ger identiskt frö, och därmed exakt samma "slumpmässiga" ordning.

Alternativet till en statisk lokal variabel vore en global variabel eller en medlemsvariabel, men den lokala varianten begränsar synligheten till den enda funktion som faktiskt behöver den.

### Blandning av träningsordningen
`shuffle()` itererar genom `myTrainOrder` och byter plats på varje index `i` och ett slumpmässigt valt index `r`:

```cpp
void Fixed::shuffle() noexcept
{
    // Iterate through all training sets, swap each index i with a random index r.
    for (std::size_t i{}; i < myTrainOrder.size(); ++i)
    {
        const auto r    = std::rand() % myTrainOrder.size();
        const auto temp = myTrainOrder[i];
        myTrainOrder[i] = myTrainOrder[r];
        myTrainOrder[r] = temp;
    }
}
```

Notera att det är indexen som blandas, inte träningsdatan; träningsdatan är `const` och ägs dessutom av anroparen, medan indexvektorn är billig att flytta runt i och kan blandas hur många gånger som helst utan att datan påverkas.

---

## Träningsalgoritmen
`train()` utför följande per epok:

1. Blanda träningsordningen via `shuffle()`.
2. Iterera genom `myTrainOrder` och hämta indatan `myTrainIn[i]` samt referensvärdet `myTrainOut[i]` för varje index `i`.
3. Anropa `optimize()` för varje träningsuppsättning.

Optimeringen för en enskild uppsättning:

1. Om indatan är noll: sätt `myBias = output` och avsluta.
2. Prediktera med aktuell indata: `prediction = myWeight * input + myBias`.
3. Beräkna felet som differensen mellan referensvärdet och prediktionen: `error = output - prediction`.
4. Justera modellens parametrar utefter felet:
    * `myBias   += error * learningRate`
    * `myWeight += error * learningRate * input`

Observera att loopvariabeln `i` i `train()` inte längre är en räknare, utan ett index in i träningsdatan hämtat ur `myTrainOrder`:

```cpp
for (const auto i : myTrainOrder)
{
    const auto input  = myTrainIn[i];
    const auto output = myTrainOut[i];
    optimize(input, output, learningRate);
}
```

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

`main.cpp` tränar en modell på `y = 2x + 2` i 20 epoker med lärhastigheten 0.23, och predikterar sedan med varje indata i träningsdatan. Utskriften ska se ut såsom visas nedan, oavsett hur träningsordningen blandas:

```
----------------------------------------
Input: 0, output: 2
Input: 1, output: 4
Input: 2, output: 6
Input: 3, output: 8
Input: 4, output: 10
----------------------------------------
```

Kör programmet flera gånger; ordningen som modellen tränas i skiljer sig mellan körningarna, men resultatet efter genomförd träning är detsamma.

---

## Att notera
* Namnen på medlemsvariabler och lokala variabler behöver inte stämma exakt med bilaga B; det är beteendet som ska stämma.
* `initRandom()` anropas här i konstruktorn, så att generatorn är initierad innan träningsordningen kan blandas första gången. Ett anrop i början av `train()` fungerar lika bra, eftersom funktionen ändå bara utför något vid sitt första anrop; bilaga B tillåter båda placeringarna.
* `shuffle()` är implementerad som en medlemsmetod utan argument, medan bilaga A blandar en vektor som skickas som argument. Skillnaden är att träningsordningen här är en medlemsvariabel och därmed alltid tillgänglig för metoden.
* Jämförelsen `0.0 == input` är en exakt flyttalsjämförelse. Det fungerar här, eftersom träningsdatan innehåller en exakt nolla, men mot data som beräknats fram vore en jämförelse mot ett litet tröskelvärde säkrare.
* Blandningsalgoritmen ovan är enkel snarare än statistiskt perfekt; alla permutationer blir inte exakt lika sannolika, och `std::rand() % N` ger dessutom en liten skevhet mot de lägsta indexen. Det spelar ingen roll för träningen, men `std::shuffle()` tillsammans med `std::mt19937` från `<random>` är standardbibliotekets riktiga verktyg för samma sak i ren värdkod.
* Slumptalsgeneratorn sås med tiden, vilket gör varje körning unik. Vill man kunna reproducera en körning exakt, exempelvis vid felsökning, kan `std::srand()` i stället anropas med ett fast frö.

---

## Nästa steg
* **L04:** Beräkning av medelfel per epok, för att kunna följa modellens träningsförlopp.
* **P03:** `ml::lin_reg::Adaptive`, en modell med adaptiv lärhastighet, i ett inbyggt system.

---
