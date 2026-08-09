# Bilaga C - Övningsuppgifter

## Vektorer i C++
**1.** Skapa en fil döpt `main.cpp`. Anta att du har följande vektor:

```cpp
std::vector<std::uint8_t> v1{1U, 2U, 3U};
```

Iterera genom vektorn och skriv ut dess innehåll på hexadecimal form via:
**a)** Index.  
**b)** Range-baserad for-loop.  
**c)** C++-iterator.

**Tips:** Heltal kan skrivas ut på hexadecimal form med `std::printf()`:

```cpp
// Skriver ut '03'.
std::printf("%02X", 3U);

// Skriver ut '0x0F':
std::printf("0x%02X", 5U);
```

---

**2.** Visa via C++-instruktioner i `main()` hur du kan:  
**a)** Kolla vektorns storlek.  
**b)** Lägga till talet `4U` längst bak i vektorn.  
**c)** Sätta vektorns storlek till `10U`.  
**d)** Tömma vektorn.  
**e)** Kolla om vektorn är tom.  
**f)** Ta bort det sista elementet.

---

**3.** Anta att du ska implementera en funktion där du enbart vill läsa innehållet i en vektor. Du kan antingen skicka en referens (*pass-by-reference*) eller en kopia (*pass-by-value*) av vektorn. Vilken metod hade du använt och varför? 

---

**4.** Skapa en funktion döpt `getLargestInt()` som returnerar det största talet i en vektor med heltal av typen `int`. Om vektorn är tom ska `0` returneras.

Använd `getLargestInt()` för att radera det största talet ur följande vektor:

```cpp
std::vector<int> numbers{1, 2, 5, 4, 8, 9, 0, -3};
```

---

**5.** En regressionsmodell ska tränas via de fem träningsuppsättningarna definierade enligt formeln `y = 3x + 2`:

| x | y  |
|:-:|:--:|
| 0 |  2 |
| 1 |  5 |
| 2 |  8 |
| 3 | 11 |
| 4 | 14 |

Anta att modellens bias (m-värde) samt vikt (k-värde) är noll vid start:

$$\begin{cases} k = 0 \\ m = 0 \end{cases}$$

Genomför träning under en epok med en lärhastighet LR på 10 %:

$$LR = 0.1$$

Genomför sedan prediktion för indata bestående av alla heltal inom intervallet [-5, 5].

---
