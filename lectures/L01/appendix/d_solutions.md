# Bilaga D - Lösningsförslag
Denna bilaga innehåller lösningsförslag till övningsuppgifterna i [bilaga C](./c_exercises.md). Koden är skriven i C++17 och följer kursens kodstandard. Flera av uppgifterna kan lösas på andra sätt; lösningarna nedan utgör alltså förslag snarare än det enda rätta svaret.

---

## Vektorer i C++

### Uppgift 1 - Iteration och hexadecimal utskrift
Nedan visas en `main.cpp` där vektorns innehåll skrivs ut på hexadecimal form via index, range-baserad for-loop samt C++-iterator:

```cpp
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

int main()
{
    std::vector<std::uint8_t> v1{1U, 2U, 3U};

    // a) Iterate via index.
    std::printf("Index:\t\t\t");
    for (std::size_t i{}; i < v1.size(); ++i)
    {
        std::printf("0x%02X, ", v1[i]);
    }

    // b) Iterate via a range-based for-loop.
    std::printf("\nRange-based for loop:\t");
    for (const auto byte : v1)
    {
        std::printf("0x%02X, ", byte);
    }

    // c) Iterate via a C++ iterator.
    std::printf("\nC++ iterator:\t\t");
    for (auto it = v1.begin(); it != v1.end(); ++it)
    {
        std::printf("0x%02X, ", *it);
    }
    std::printf("\n");
    return 0;
}
```

Samtliga tre loopar skriver ut samma innehåll, alltså `0x01`, `0x02` samt `0x03`. Utskriften sker med en rad per loop, föregången av en etikett som visar vilken iterationsmetod som används.

---

### Uppgift 2 - Vanliga vektormetoder
Nedan visas samtliga operationer i den ordning de anges i uppgiften:

```cpp
#include <cstdint>
#include <cstdio>
#include <vector>

int main()
{
    std::vector<std::uint8_t> v1{1U, 2U, 3U};

    // a) Check the size of the vector.
    std::printf("Size: %zu\n", v1.size());

    // b) Add the number 4 at the back of the vector.
    v1.push_back(4U);

    // c) Resize the vector to hold ten elements.
    v1.resize(10U);

    // d) Clear the vector.
    v1.clear();

    // e) Check if the vector is empty.
    std::printf("Empty: %s\n", v1.empty() ? "yes" : "no");

    // f) Remove the last element, if any.
    if (!v1.empty()) { v1.pop_back(); }
    return 0;
}
```

**Kommentarer:**
* `resize(10U)` nollställer nya element. Vid behov kan ett annat värde anges via det andra argumentet, exempelvis `resize(10U, 5U)` för att sätta nya element till 5. Om den nya storleken är mindre än den nuvarande raderas överskjutande element från slutet.
* `empty()` är att föredra framför `0U == size()`; metoden uttrycker avsikten tydligare.
* **OBS!** `pop_back()` på en tom vektor medför odefinierat beteende. Eftersom vektorn töms i deluppgift d) måste anropet i deluppgift f) därmed skyddas av en kontroll, vilket görs ovan. Detta är ett typiskt fel att ha i åtanke; `std::vector` kontrollerar inte detta åt dig.

---

### Uppgift 3 - Referens eller kopia?
Vid enbart läsning ska vektorn passeras som en referens till skrivskyddad data, alltså `const std::vector<double>&`:

```cpp
void print(const std::vector<double>& values) noexcept;
```

**Motivering:**
1. **Ingen onödig kopiering:** En kopia innebär att samtliga element kopieras, vilket dessutom kräver en ny minnesallokering på heapen. Kostnaden växer med vektorns storlek, medan en referens är lika billig oavsett hur många element vektorn innehåller.
2. **Tydlig avsikt:** Nyckelordet `const` visar för läsaren att funktionen enbart läser innehållet, samtidigt som kompilatorn ser till att detta faktiskt efterlevs.
3. **Ingen risk för oavsiktlig ändring:** Utan `const` hade funktionen kunnat ändra anroparens vektor av misstag.

En kopia (*pass-by-value*) är motiverad enbart då funktionen behöver en egen version som den kan ändra fritt utan att påverka anroparens data, exempelvis för att sortera innehållet lokalt.

---

### Uppgift 4 - Funktionen getLargestInt()
Nedan visas en implementering av `getLargestInt()` samt hur denna används för att radera det största talet ur vektorn:

```cpp
#include <algorithm>
#include <cstdio>
#include <vector>

namespace
{
[[nodiscard]] int getLargestInt(const std::vector<int>& numbers) noexcept
{
    // Check if the vector is empty, return 0 if true.
    if (numbers.empty()) { return 0; }
    auto largest = numbers[0U];

    // Store each number exceeding the currently largest number.
    for (const auto& number : numbers)
    {
        if (number > largest) { largest = number; }
    }
    return largest;
}
} // namespace

int main()
{
    std::vector<int> numbers{1, 2, 5, 4, 8, 9, 0, -3};

    // Get the largest number, then find its position in the vector.
    const auto largest = getLargestInt(numbers);
    const auto it      = std::find(numbers.begin(), numbers.end(), largest);

    // Erase the largest number, if found.
    if (it != numbers.end()) { numbers.erase(it); }

    // Print the remaining numbers.
    for (const auto& number : numbers)
    {
        std::printf("%d\n", number);
    }
    return 0;
}
```

Programmet skriver ut talen `1, 2, 5, 4, 8, 0` samt `-3`, ett tal per rad, alltså den ursprungliga vektorn utan talet 9.

**Kommentarer:**
* Variabeln `largest` initieras till vektorns första element, inte till noll. Om den hade initierats till noll hade en vektor med enbart negativa tal felaktigt returnerat 0, trots att inget av elementen är noll.
* `erase()` tar en iterator, inte ett värde. Därför används `std::find()` från `<algorithm>` för att få en iterator till det största talets position.
* Vid dubbletter av det största talet raderas enbart den första instansen.
* Attributet `[[nodiscard]]` medför att kompilatorn varnar om returvärdet ignoreras, vilket är lämpligt för en funktion vars enda syfte är att ta fram ett värde.

---

## Linjär regression

### Uppgift 5 - Träning för hand
Träningsdatan följer formeln y = 3x + 2, vilket innebär att modellens parametrar ska närma sig k = 3 samt m = 2 under träningen:

| x | y  |
|:-:|:--:|
| 0 |  2 |
| 1 |  5 |
| 2 |  8 |
| 3 | 11 |
| 4 | 14 |

Vid start gäller att både vikten k samt biasvärdet m är lika med noll:

$$\begin{cases} k = 0 \\ m = 0 \end{cases}$$

Träning genomförs under en epok, alltså en genomgång av samtliga fem träningsuppsättningar, med lärhastigheten LR = 0.1. För varje träningsuppsättning genomförs följande fyra steg:
1. Prediktera med aktuell indata: $y_p = k * x + m$.
2. Beräkna avvikelsen mot referensvärdet: $\delta = y_{ref} - y_p$.
3. Beräkna justeringsmängden: $\Delta e = \delta * LR$.
4. Justera parametrarna: $m = m + \Delta e$ samt $k = k + \Delta e * x$.

#### Träningsuppsättning 1
$$\begin{cases} x = 0 \\ y_{ref} = 2 \end{cases}$$

$$y_p = k * x + m = 0 * 0 + 0 = 0$$

$$\delta = y_{ref} - y_p = 2 - 0 = 2$$

$$\Delta e = \delta * LR = 2 * 0.1 = 0.2$$

$$m = m + \Delta e = 0 + 0.2 = 0.2$$

$$k = k + \Delta e * x = 0 + 0.2 * 0 = 0$$

Efter den första träningsrundan:

$$\begin{cases} k = 0 \\ m = 0.2 \end{cases}$$

Notera att vikten k är oförändrad, eftersom indatan x är lika med noll. Vid x = 0 utgörs prediktionen enbart av biasvärdet m, oavsett vilket värde vikten k har.

#### Träningsuppsättning 2
$$\begin{cases} x = 1 \\ y_{ref} = 5 \end{cases}$$

$$y_p = k * x + m = 0 * 1 + 0.2 = 0.2$$

$$\delta = y_{ref} - y_p = 5 - 0.2 = 4.8$$

$$\Delta e = \delta * LR = 4.8 * 0.1 = 0.48$$

$$m = m + \Delta e = 0.2 + 0.48 = 0.68$$

$$k = k + \Delta e * x = 0 + 0.48 * 1 = 0.48$$

Efter den andra träningsrundan:

$$\begin{cases} k = 0.48 \\ m = 0.68 \end{cases}$$

#### Träningsuppsättning 3
$$\begin{cases} x = 2 \\ y_{ref} = 8 \end{cases}$$

$$y_p = k * x + m = 0.48 * 2 + 0.68 = 1.64$$

$$\delta = y_{ref} - y_p = 8 - 1.64 = 6.36$$

$$\Delta e = \delta * LR = 6.36 * 0.1 = 0.636$$

$$m = m + \Delta e = 0.68 + 0.636 = 1.316$$

$$k = k + \Delta e * x = 0.48 + 0.636 * 2 = 1.752$$

Efter den tredje träningsrundan:

$$\begin{cases} k = 1.752 \\ m = 1.316 \end{cases}$$

Notera att avvikelsen δ hittills har ökat för varje träningsuppsättning. Detta beror på att referensvärdena växer snabbare än vad modellens parametrar hinner justeras under de första rundorna.

#### Träningsuppsättning 4
$$\begin{cases} x = 3 \\ y_{ref} = 11 \end{cases}$$

$$y_p = k * x + m = 1.752 * 3 + 1.316 = 6.572$$

$$\delta = y_{ref} - y_p = 11 - 6.572 = 4.428$$

$$\Delta e = \delta * LR = 4.428 * 0.1 = 0.4428$$

$$m = m + \Delta e = 1.316 + 0.4428 = 1.7588$$

$$k = k + \Delta e * x = 1.752 + 0.4428 * 3 = 3.0804$$

Efter den fjärde träningsrundan:

$$\begin{cases} k = 3.0804 \\ m = 1.7588 \end{cases}$$

Notera att avvikelsen δ nu har minskat för första gången, samtidigt som vikten k har passerat sitt önskade värde 3 något.

#### Träningsuppsättning 5
$$\begin{cases} x = 4 \\ y_{ref} = 14 \end{cases}$$

$$y_p = k * x + m = 3.0804 * 4 + 1.7588 = 14.0804$$

$$\delta = y_{ref} - y_p = 14 - 14.0804 = -0.0804$$

$$\Delta e = \delta * LR = -0.0804 * 0.1 = -0.00804$$

$$m = m + \Delta e = 1.7588 - 0.00804 = 1.75076$$

$$k = k + \Delta e * x = 3.0804 - 0.00804 * 4 = 3.04824$$

Efter den femte träningsrundan:

$$\begin{cases} k = 3.04824 \\ m = 1.75076 \end{cases}$$

Här är avvikelsen δ negativ för första gången, eftersom prediktionen 14.0804 är något för hög jämfört med referensvärdet 14. Justeringsmängden Δe blir då också negativ, vilket innebär att både k och m minskas i stället för att ökas.

#### Resultat efter en epok
Efter genomförd träning predikterar modellen enligt följande formel:

$$y_p = 3.04824 * x + 1.75076$$

Vikten k ligger mycket nära sitt önskade värde 3, medan biasvärdet m fortfarande ligger en bit från sitt önskade värde 2. Detta är väntat, av två anledningar:
* Vikten k justeras med Δe * x, alltså mer ju större indatan x är, medan biasvärdet m justeras med enbart Δe.
* De två parametrarna tränas samtidigt utifrån samma avvikelse. När vikten k väl är tillräckligt bra blir avvikelsen δ liten, vilket innebär att biasvärdet m därefter justeras långsamt.

Med fler epoker fortsätter båda parametrarna att närma sig sina önskade värden.

---

### Verifiering
I nedanstående tabell visas predikterad utdata $y_p$ samt motsvarande referensvärden $y_{ref}$ för indata $x$ i intervallet [-5, 5]. Predikterad utdata har avrundats till två decimaler.

| $x$ | $y_p$  | $y_{ref}$ |
|:--:|:------:|:---------:|
| -5 | -13.49 | -13       |
| -4 | -10.44 | -10       |
| -3 |  -7.39 |  -7       |
| -2 |  -4.35 |  -4       |
| -1 |  -1.30 |  -1       |
|  0 |   1.75 |   2       |
|  1 |   4.80 |   5       |
|  2 |   7.85 |   8       |
|  3 |  10.90 |  11       |
|  4 |  13.94 |  14       |
|  5 |  16.99 |  17       |

Modellen predikterar med god precision efter en enda epok, men notera att avvikelsen är betydligt större för negativa indata än för positiva:
* Vikten k är något för hög (3.04824 i stället för 3), medan biasvärdet m är för lågt (1.75076 i stället för 2). För positiva indata tar dessa två fel delvis ut varandra, medan de i stället adderas för negativa indata.
* Träningsdatan täcker enbart intervallet [0, 4]. Prediktioner utanför detta intervall utgör extrapolering, vilket generellt är mindre tillförlitligt än prediktioner inom det intervall modellen faktiskt har tränats på.

Detta illustrerar en generell princip inom maskininlärning: en modell är enbart så bra som den träningsdata den har fått, både sett till mängd och sett till vilket intervall datan täcker.

---
