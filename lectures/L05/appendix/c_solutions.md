# Bilaga C - Lösningsförslag
Fullständiga lösningar till de tre träningsexemplen i [bilaga B](./b_exercises.md). Varje exempel genomförs i fyra steg: feedforward, backpropagation, optimering samt en kontroll via feedforward med de nya parametrarna.

---

## Beteckningar
| Beteckning | Betydelse |
|---|---|
| $x_n$ | Insignal till nätverket. |
| $s_n$ | Viktad summa för nod $n$, dvs. nodens bias plus samtliga insignaler multiplicerade med respektive vikt. |
| $y_n$ | Utsignal från nod $n$: $y_n = \sigma(s_n)$. |
| $\sigma$, $\sigma'$ | Aktiveringsfunktionen samt dess derivata. |
| $\delta_n$ | Avvikelse för nod $n$. |
| $e_n$ | Fel för nod $n$: $e_n = \delta_n * \sigma'(s_n)$. |
| $\Delta c_n$ | Förändringshastighet för nod $n$: $\Delta c_n = e_n * LR$. |
| $LR$ | Lärhastighet. |

Samtliga exempel använder ReLU som aktiveringsfunktion:

$$\sigma(s) = \begin{cases} s, & s > 0 \\ 0, & s \leq 0 \end{cases} \qquad \sigma'(s) = \begin{cases} 1, & s > 0 \\ 0, & s \leq 0 \end{cases}$$

Derivatan är alltså 1 för varje nod vars viktade summa är positiv, och 0 för varje nod vars viktade summa är noll eller negativ.

---

## Arbetsgång
1. **Feedforward:** Beräkna $s$ samt $y = \sigma(s)$ för varje nod, lager för lager från ingången mot utgången.
2. **Backpropagation:** Beräkna avvikelse och fel för varje nod, lager för lager från utgången mot ingången:
    * Utgångslagret: $\delta = Y_{train} - y$ samt $e = \delta * \sigma'(s)$.
    * Dolda lagret: $\delta$ är summan av felet $e$ för varje ansluten nod i nästa lager, multiplicerat med vikten som förbinder noderna; därefter $e = \delta * \sigma'(s)$.
3. **Optimering:** Beräkna $\Delta c = e * LR$ för varje nod och uppdatera sedan nodens parametrar:
    * Bias: $b = b + \Delta c$.
    * Vikt: $w = w + \Delta c * (\text{insignalen via vikten})$.
4. **Kontroll:** Genomför feedforward på nytt med de nya parametrarna och jämför avvikelsen med den ursprungliga.

Två regler är lätta att missa:
* **Samtliga fel beräknas innan någon parameter uppdateras.** Dolda lagrets avvikelser beräknas alltså med utgångslagrets *gamla* vikter, även om utgångslagret redovisas först under optimeringen.
* **En vikt multipliceras med sin egen insignal.** Insignalen är utsignalen från noden i föregående lager (eller $x$ för det första lagret), inte utsignalen från den nod som vikten leder in i.

---

## Exempel 1: Udda antal höga insignaler
![](./images/network1.png)

**Träningsuppsättning:** $X_1X_2 = 01$, $Y = 1$. **Lärhastighet:** $LR = 0{,}1$.

```
b1 = 0.2,  b2 = 0.4,  b3 = 0.6
w1 = 0.5,  w2 = 0.6,  w3 = 0.3
w4 = 0.8,  w5 = 0.1,  w6 = 0.9
```

### 1. Feedforward
$$x_1 = X_1 = 0, \quad x_2 = X_2 = 1$$

$$y_1 = \sigma(b_1 + x_1 w_1 + x_2 w_2) = \sigma(0{,}2 + 0 * 0{,}5 + 1 * 0{,}6) = \sigma(0{,}8) = 0{,}8$$

$$y_2 = \sigma(b_2 + x_1 w_3 + x_2 w_4) = \sigma(0{,}4 + 0 * 0{,}3 + 1 * 0{,}8) = \sigma(1{,}2) = 1{,}2$$

$$y_3 = \sigma(b_3 + y_1 w_5 + y_2 w_6) = \sigma(0{,}6 + 0{,}8 * 0{,}1 + 1{,}2 * 0{,}9) = \sigma(1{,}76) = 1{,}76$$

$$Y = y_3 = 1{,}76$$

Samtliga viktade summor är positiva, så $\sigma'(s) = 1$ för samtliga noder.

### 2. Backpropagation
Utgångsnoden:

$$\delta_3 = Y_{train} - y_3 = 1 - 1{,}76 = -0{,}76$$

$$e_3 = \delta_3 * \sigma'(s_3) = -0{,}76 * 1 = -0{,}76$$

Dolda noder, med de gamla vikterna $w_5$ och $w_6$:

$$\delta_1 = e_3 w_5 = -0{,}76 * 0{,}1 = -0{,}076, \quad e_1 = \delta_1 * \sigma'(s_1) = -0{,}076 * 1 = -0{,}076$$

$$\delta_2 = e_3 w_6 = -0{,}76 * 0{,}9 = -0{,}684, \quad e_2 = \delta_2 * \sigma'(s_2) = -0{,}684 * 1 = -0{,}684$$

### 3. Optimering
Utgångsnoden:

$$\Delta c_3 = e_3 * LR = -0{,}76 * 0{,}1 = -0{,}076$$

$$b_3 = b_3 + \Delta c_3 = 0{,}6 + (-0{,}076) = 0{,}524$$

$$w_5 = w_5 + \Delta c_3 * y_1 = 0{,}1 + (-0{,}076) * 0{,}8 = 0{,}0392$$

$$w_6 = w_6 + \Delta c_3 * y_2 = 0{,}9 + (-0{,}076) * 1{,}2 = 0{,}8088$$

Dold nod 1:

$$\Delta c_1 = e_1 * LR = -0{,}076 * 0{,}1 = -0{,}0076$$

$$b_1 = b_1 + \Delta c_1 = 0{,}2 + (-0{,}0076) = 0{,}1924$$

$$w_1 = w_1 + \Delta c_1 * x_1 = 0{,}5 + (-0{,}0076) * 0 = 0{,}5$$

$$w_2 = w_2 + \Delta c_1 * x_2 = 0{,}6 + (-0{,}0076) * 1 = 0{,}5924$$

Dold nod 2:

$$\Delta c_2 = e_2 * LR = -0{,}684 * 0{,}1 = -0{,}0684$$

$$b_2 = b_2 + \Delta c_2 = 0{,}4 + (-0{,}0684) = 0{,}3316$$

$$w_3 = w_3 + \Delta c_2 * x_1 = 0{,}3 + (-0{,}0684) * 0 = 0{,}3$$

$$w_4 = w_4 + \Delta c_2 * x_2 = 0{,}8 + (-0{,}0684) * 1 = 0{,}7316$$

Vikterna $w_1$ och $w_3$ är oförändrade, eftersom deras insignal $x_1$ är noll; en vikt vars insignal är noll har inte bidragit till felet och justeras därför inte.

Nya parametrar:

```
b1 = 0.1924,  b2 = 0.3316,  b3 = 0.524
w1 = 0.5,     w2 = 0.5924,  w3 = 0.3
w4 = 0.7316,  w5 = 0.0392,  w6 = 0.8088
```

### 4. Kontroll
$$y_1 = \sigma(0{,}1924 + 0 * 0{,}5 + 1 * 0{,}5924) = \sigma(0{,}7848) = 0{,}7848$$

$$y_2 = \sigma(0{,}3316 + 0 * 0{,}3 + 1 * 0{,}7316) = \sigma(1{,}0632) = 1{,}0632$$

$$y_3 = \sigma(0{,}524 + 0{,}7848 * 0{,}0392 + 1{,}0632 * 0{,}8088) = \sigma(1{,}4147) \approx 1{,}415$$

$$\delta_3 = 1 - 1{,}415 = -0{,}415$$

**Resultat:** Avvikelsen minskade från $-0{,}76$ till cirka $-0{,}415$, nästan en halvering efter ett enda träningssteg.

---

## Exempel 2: Inversen av insignalerna
![](./images/network2.png)

**Träningsuppsättning:** $X_1X_2 = 10$, $Y_1Y_2 = 01$. **Lärhastighet:** $LR = 0{,}1$.

```
b1 = 0.1,  b2 = 0.1,  b3 = 0.7
w1 = 0.2,  w2 = 0.9,  w3 = 0.3
w4 = 0.8,  w5 = 0.5,  w6 = 0.4

b4  = 0.1,  b5  = 0.6
w7  = 0.1,  w8  = 0.1,  w9  = 1.0
w10 = 0.3,  w11 = 0.0,  w12 = 0.6
```

### 1. Feedforward
$$x_1 = X_1 = 1, \quad x_2 = X_2 = 0$$

Dolda lagret:

$$y_1 = \sigma(b_1 + x_1 w_1 + x_2 w_2) = \sigma(0{,}1 + 1 * 0{,}2 + 0 * 0{,}9) = \sigma(0{,}3) = 0{,}3$$

$$y_2 = \sigma(b_2 + x_1 w_3 + x_2 w_4) = \sigma(0{,}1 + 1 * 0{,}3 + 0 * 0{,}8) = \sigma(0{,}4) = 0{,}4$$

$$y_3 = \sigma(b_3 + x_1 w_5 + x_2 w_6) = \sigma(0{,}7 + 1 * 0{,}5 + 0 * 0{,}4) = \sigma(1{,}2) = 1{,}2$$

Utgångslagret:

$$y_4 = \sigma(b_4 + y_1 w_7 + y_2 w_8 + y_3 w_9) = \sigma(0{,}1 + 0{,}3 * 0{,}1 + 0{,}4 * 0{,}1 + 1{,}2 * 1{,}0) = \sigma(1{,}37) = 1{,}37$$

$$y_5 = \sigma(b_5 + y_1 w_{10} + y_2 w_{11} + y_3 w_{12}) = \sigma(0{,}6 + 0{,}3 * 0{,}3 + 0{,}4 * 0{,}0 + 1{,}2 * 0{,}6) = \sigma(1{,}41) = 1{,}41$$

$$Y_1 = y_4 = 1{,}37, \quad Y_2 = y_5 = 1{,}41$$

Samtliga viktade summor är positiva, så $\sigma'(s) = 1$ för samtliga noder.

### 2. Backpropagation
Utgångslagret:

$$\delta_4 = Y_{1,train} - y_4 = 0 - 1{,}37 = -1{,}37, \quad e_4 = \delta_4 * \sigma'(s_4) = -1{,}37$$

$$\delta_5 = Y_{2,train} - y_5 = 1 - 1{,}41 = -0{,}41, \quad e_5 = \delta_5 * \sigma'(s_5) = -0{,}41$$

Dolda lagret, där varje nod är ansluten till båda utgångsnoderna:

$$\delta_1 = e_4 w_7 + e_5 w_{10} = -1{,}37 * 0{,}1 + (-0{,}41) * 0{,}3 = -0{,}137 + (-0{,}123) = -0{,}26$$

$$\delta_2 = e_4 w_8 + e_5 w_{11} = -1{,}37 * 0{,}1 + (-0{,}41) * 0{,}0 = -0{,}137$$

$$\delta_3 = e_4 w_9 + e_5 w_{12} = -1{,}37 * 1{,}0 + (-0{,}41) * 0{,}6 = -1{,}37 + (-0{,}246) = -1{,}616$$

$$e_1 = \delta_1 * \sigma'(s_1) = -0{,}26, \quad e_2 = \delta_2 * \sigma'(s_2) = -0{,}137, \quad e_3 = \delta_3 * \sigma'(s_3) = -1{,}616$$

### 3. Optimering
Utgångsnod 4:

$$\Delta c_4 = e_4 * LR = -1{,}37 * 0{,}1 = -0{,}137$$

$$b_4 = 0{,}1 + (-0{,}137) = -0{,}037$$

$$w_7 = w_7 + \Delta c_4 * y_1 = 0{,}1 + (-0{,}137) * 0{,}3 = 0{,}0589$$

$$w_8 = w_8 + \Delta c_4 * y_2 = 0{,}1 + (-0{,}137) * 0{,}4 = 0{,}0452$$

$$w_9 = w_9 + \Delta c_4 * y_3 = 1{,}0 + (-0{,}137) * 1{,}2 = 0{,}8356$$

Utgångsnod 5:

$$\Delta c_5 = e_5 * LR = -0{,}41 * 0{,}1 = -0{,}041$$

$$b_5 = 0{,}6 + (-0{,}041) = 0{,}559$$

$$w_{10} = w_{10} + \Delta c_5 * y_1 = 0{,}3 + (-0{,}041) * 0{,}3 = 0{,}2877$$

$$w_{11} = w_{11} + \Delta c_5 * y_2 = 0{,}0 + (-0{,}041) * 0{,}4 = -0{,}0164$$

$$w_{12} = w_{12} + \Delta c_5 * y_3 = 0{,}6 + (-0{,}041) * 1{,}2 = 0{,}5508$$

Dolda lagret:

$$\Delta c_1 = e_1 * LR = -0{,}026, \quad \Delta c_2 = e_2 * LR = -0{,}0137, \quad \Delta c_3 = e_3 * LR = -0{,}1616$$

$$b_1 = 0{,}1 + (-0{,}026) = 0{,}074, \quad w_1 = 0{,}2 + (-0{,}026) * 1 = 0{,}174, \quad w_2 = 0{,}9 + (-0{,}026) * 0 = 0{,}9$$

$$b_2 = 0{,}1 + (-0{,}0137) = 0{,}0863, \quad w_3 = 0{,}3 + (-0{,}0137) * 1 = 0{,}2863, \quad w_4 = 0{,}8 + (-0{,}0137) * 0 = 0{,}8$$

$$b_3 = 0{,}7 + (-0{,}1616) = 0{,}5384, \quad w_5 = 0{,}5 + (-0{,}1616) * 1 = 0{,}3384, \quad w_6 = 0{,}4 + (-0{,}1616) * 0 = 0{,}4$$

Vikterna $w_2$, $w_4$ och $w_6$ är oförändrade, eftersom deras insignal $x_2$ är noll. Vikten $w_{11}$ uppdateras däremot trots att den var noll; det är insignalen $y_2 = 0{,}4$, inte viktens eget värde, som avgör om vikten justeras.

Nya parametrar:

```
b1 = 0.074,   b2 = 0.0863,  b3 = 0.5384
w1 = 0.174,   w2 = 0.9,     w3 = 0.2863
w4 = 0.8,     w5 = 0.3384,  w6 = 0.4

b4  = -0.037,  b5  = 0.559
w7  = 0.0589,  w8  = 0.0452,   w9  = 0.8356
w10 = 0.2877,  w11 = -0.0164,  w12 = 0.5508
```

### 4. Kontroll
Dolda lagret:

$$y_1 = \sigma(0{,}074 + 1 * 0{,}174 + 0 * 0{,}9) = \sigma(0{,}248) = 0{,}248$$

$$y_2 = \sigma(0{,}0863 + 1 * 0{,}2863 + 0 * 0{,}8) = \sigma(0{,}3726) = 0{,}3726$$

$$y_3 = \sigma(0{,}5384 + 1 * 0{,}3384 + 0 * 0{,}4) = \sigma(0{,}8768) = 0{,}8768$$

Utgångslagret:

$$y_4 = \sigma(-0{,}037 + 0{,}248 * 0{,}0589 + 0{,}3726 * 0{,}0452 + 0{,}8768 * 0{,}8356) = \sigma(0{,}7271) \approx 0{,}727$$

$$y_5 = \sigma(0{,}559 + 0{,}248 * 0{,}2877 + 0{,}3726 * (-0{,}0164) + 0{,}8768 * 0{,}5508) = \sigma(1{,}1072) \approx 1{,}107$$

$$\delta_4 = 0 - 0{,}727 = -0{,}727, \quad \delta_5 = 1 - 1{,}107 = -0{,}107$$

**Resultat:** Båda avvikelserna minskade:
* $Y_1$: från $-1{,}37$ till cirka $-0{,}727$.
* $Y_2$: från $-0{,}41$ till cirka $-0{,}107$.

Nod 3 fick den största korrigeringen i det dolda lagret. Den är starkast kopplad till utgången via $w_9 = 1{,}0$ och $w_{12} = 0{,}6$, och hade därmed störst del i felet.

---

## Exempel 3: Udda antal höga insignaler, med en död nod
![](./images/network3.png)

**Träningsuppsättning:** $X_1X_2 = 11$, $Y = 0$. **Lärhastighet:** $LR = 0{,}1$.

```
b1 = 0.1,   b2 = -0.4,  b3 = -0.2
w1 = 0.5,   w2 = 0.4,   w3 = -0.2
w4 = 0.1,   w5 = 0.7,   w6 = 0.8
```

### 1. Feedforward
$$x_1 = X_1 = 1, \quad x_2 = X_2 = 1$$

$$y_1 = \sigma(b_1 + x_1 w_1 + x_2 w_2) = \sigma(0{,}1 + 1 * 0{,}5 + 1 * 0{,}4) = \sigma(1{,}0) = 1{,}0$$

$$y_2 = \sigma(b_2 + x_1 w_3 + x_2 w_4) = \sigma(-0{,}4 + 1 * (-0{,}2) + 1 * 0{,}1) = \sigma(-0{,}5) = 0$$

$$y_3 = \sigma(b_3 + y_1 w_5 + y_2 w_6) = \sigma(-0{,}2 + 1{,}0 * 0{,}7 + 0 * 0{,}8) = \sigma(0{,}5) = 0{,}5$$

$$Y = y_3 = 0{,}5$$

Den viktade summan för nod 2 är negativ, så $y_2 = 0$ och $\sigma'(s_2) = 0$. För nod 1 och 3 gäller $\sigma'(s) = 1$.

### 2. Backpropagation
Utgångsnoden:

$$\delta_3 = Y_{train} - y_3 = 0 - 0{,}5 = -0{,}5$$

$$e_3 = \delta_3 * \sigma'(s_3) = -0{,}5 * 1 = -0{,}5$$

Dolda noder:

$$\delta_1 = e_3 w_5 = -0{,}5 * 0{,}7 = -0{,}35, \quad e_1 = \delta_1 * \sigma'(s_1) = -0{,}35 * 1 = -0{,}35$$

$$\delta_2 = e_3 w_6 = -0{,}5 * 0{,}8 = -0{,}4, \quad e_2 = \delta_2 * \sigma'(s_2) = -0{,}4 * 0 = 0$$

Nod 2 har en avvikelse, men eftersom derivatan är noll blir felet noll; noden kommer inte att uppdateras i detta steg.

### 3. Optimering
Utgångsnoden:

$$\Delta c_3 = e_3 * LR = -0{,}5 * 0{,}1 = -0{,}05$$

$$b_3 = b_3 + \Delta c_3 = -0{,}2 + (-0{,}05) = -0{,}25$$

$$w_5 = w_5 + \Delta c_3 * y_1 = 0{,}7 + (-0{,}05) * 1{,}0 = 0{,}65$$

$$w_6 = w_6 + \Delta c_3 * y_2 = 0{,}8 + (-0{,}05) * 0 = 0{,}8$$

Dold nod 1:

$$\Delta c_1 = e_1 * LR = -0{,}35 * 0{,}1 = -0{,}035$$

$$b_1 = 0{,}1 + (-0{,}035) = 0{,}065, \quad w_1 = 0{,}5 + (-0{,}035) * 1 = 0{,}465, \quad w_2 = 0{,}4 + (-0{,}035) * 1 = 0{,}365$$

Dold nod 2:

$$\Delta c_2 = e_2 * LR = 0 * 0{,}1 = 0$$

$$b_2 = -0{,}4, \quad w_3 = -0{,}2, \quad w_4 = 0{,}1 \quad \text{(oförändrade)}$$

Även $w_6$ är oförändrad, eftersom dess insignal $y_2$ är noll.

Nya parametrar:

```
b1 = 0.065,  b2 = -0.4,  b3 = -0.25
w1 = 0.465,  w2 = 0.365, w3 = -0.2
w4 = 0.1,    w5 = 0.65,  w6 = 0.8
```

### 4. Kontroll
$$y_1 = \sigma(0{,}065 + 1 * 0{,}465 + 1 * 0{,}365) = \sigma(0{,}895) = 0{,}895$$

$$y_2 = \sigma(-0{,}4 + 1 * (-0{,}2) + 1 * 0{,}1) = \sigma(-0{,}5) = 0$$

$$y_3 = \sigma(-0{,}25 + 0{,}895 * 0{,}65 + 0 * 0{,}8) = \sigma(0{,}33175) \approx 0{,}332$$

$$\delta_3 = 0 - 0{,}332 = -0{,}332$$

### Blev felet mindre?
Ja. Avvikelsen minskade från $-0{,}5$ till cirka $-0{,}332$.

Nätverket har dock ett större problem än storleken på felet: nod 2 är en så kallad **död nod**. Dess viktade summa är negativ för *samtliga* fyra träningsuppsättningar, både före och efter optimeringen:

| $X_1X_2$ | $s_2 = b_2 + x_1 w_3 + x_2 w_4$ | $y_2$ | $\sigma'(s_2)$ |
|:---:|:---:|:---:|:---:|
| 00 | $-0{,}4$ | 0 | 0 |
| 01 | $-0{,}4 + 0{,}1 = -0{,}3$ | 0 | 0 |
| 10 | $-0{,}4 - 0{,}2 = -0{,}6$ | 0 | 0 |
| 11 | $-0{,}4 - 0{,}2 + 0{,}1 = -0{,}5$ | 0 | 0 |

Derivatan är därmed noll oavsett träningsuppsättning, vilket får följande konsekvenser:
* $e_2$ och $\Delta c_2$ blir alltid noll, så $b_2$, $w_3$ och $w_4$ ändras aldrig igen.
* $y_2$ är alltid noll, så inte heller $w_6$ ändras någonsin.
* Noden kan inte återhämta sig av egen kraft, hur många epoker som än genomförs.

Nätverket har i praktiken bara en dold nod kvar, och med en enda dold nod kan det inte lära sig uppgiften. Utsignalen blir då en monoton funktion av den viktade summan $s_1$, men uppgiften kräver hög utsignal för 01 och 10 och låg utsignal för både 00 och 11. Det går inte, eftersom $s_1(01) + s_1(10) = s_1(00) + s_1(11)$; summorna för 01 och 10 kan därför inte båda ligga över, eller båda under, summorna för 00 och 11.

### Vad hade kunnat ändras?
* **Leaky ReLU i stället för ReLU:** Derivatan blir $\alpha$ (exempelvis 0,01) i stället för noll då $s \leq 0$. Felet $e_2$ blir då litet men inte noll, så nod 2 kan tränas tillbaka till liv.
* **Andra startvärden:** Det är framför allt biasvärdet $b_2 = -0{,}4$ som håller noden död. Små slumpmässiga vikter tillsammans med ett bias på noll eller strax över noll gör det osannolikt att en nod startar död för samtliga insignaler.
* **Lägre lärhastighet:** Hade felet i stället ökat, eller bytt tecken, efter optimeringen hade parametrarna justerats för mycket; då är en lägre lärhastighet det första att prova. En för hög lärhastighet kan dessutom knuffa in en nod i det döda området från början.

---

## Att notera
* **Beräkningsordningen.** Gör klart hela backpropagation-steget innan någon parameter uppdateras. Den som uppdaterar $w_5$ och $w_6$ innan $\delta_1$ och $\delta_2$ beräknas får fel värden i det dolda lagret.
* **Noll som insignal kontra noll som vikt.** En vikt vars insignal är noll lämnas orörd ($w_1$ och $w_3$ i exempel 1, $w_2$, $w_4$ och $w_6$ i exempel 2, $w_6$ i exempel 3). En vikt som själv är noll uppdateras precis som övriga vikter ($w_{11}$ i exempel 2).
* **ReLU är obegränsad uppåt.** Utsignaler som $1{,}76$ och $1{,}41$ är fullt möjliga trots att referensvärdena bara är 0 eller 1. Vid binära utsignaler används därför ofta sigmoid i utgångslagret, se [bilaga A](./a_neural_networks.md).
* **Ett träningssteg förbättrar bara den träningsuppsättning som användes.** Övriga uppsättningar drar parametrarna åt andra håll. Därför tränas nätverket över många epoker med samtliga träningsuppsättningar i randomiserad ordning, precis som regressionsmodellen i **L03**.
* **Avrundning.** Behåll minst fyra decimaler i mellanleden. Avrundningsfel fortplantar sig genom lagren och syns tydligast i kontrollsteget.

---

## Nästa steg
* **L06:** Påbörjande av övningsuppgift: Enkelt neuralt nätverk i C++ (del I), samt mer träning av neurala nätverk för hand.

---
