# Dense-lager - Matematik till kod (del II)
Denna bilaga sammanfattar hur ekvationerna för feedforward, backpropagation samt optimering från 
**L05** (se [bilaga A](../../L05/appendix/a_neural_networks.md)) motsvaras av koden ni skriver i 
`Dense` (se [bilaga B](./b_exercises.md) för den fullständiga instruktionen).

---

## Feedforward
$$s = b + \sum_{i=0}^{j} w_i * x_i, \quad y = \delta(s)$$

| Matematik | Kod |
|---|---|
| $b$ | `myBias[i]` |
| $w_i$ | `myWeights[i][j]` |
| $x_i$ | `input[j]` |
| $s$ | `sum` (sparas även i `myPreActivationOutput[i]`, se nedan) |
| $\delta(s)$ | `actFuncOutput(myActFunc, sum)` |
| $y$ | `myOutput[i]` |

`myPreActivationOutput[i]` behövs vid backpropagation nedan, eftersom aktiveringsfunktionens
derivata $y_p'$ ska beräknas utifrån $s$ (den viktade summan innan aktiveringsfunktionen
applicerades); inte utifrån $y$ (utdatan efter aktiveringsfunktionen).

---

## Backpropagation

**Utgångslager:**

$$\delta = y_{ref} - y_p, \quad \Delta e = \delta * y_p'$$

| Matematik | Kod |
|---|---|
| $y_{ref}$ | `reference[i]` |
| $y_p$ | `myOutput[i]` |
| $\delta$ | `err` |
| $y_p'$ | `actFuncDelta(myActFunc, myPreActivationOutput[i])` |
| $\Delta e$ | `myError[i]` |

**Dolt lager:**

$$\delta = \sum_{i=0}^{j} [\Delta e_i * w_i], \quad \Delta e = \delta * y_p'$$

| Matematik | Kod |
|---|---|
| $\Delta e_i$ (nästa lager) | `nextLayer.error()[j]` |
| $w_i$ (nästa lager) | `nextLayer.weights()[j][i]` |
| $\delta$ | `err` |
| $y_p'$ | `actFuncDelta(myActFunc, myPreActivationOutput[i])` |
| $\Delta e$ | `myError[i]` |

---

## Optimering
$$\Delta c_n = \Delta e_n * L, \quad b_n = b_n + \Delta c_n, \quad w_j = w_j + \Delta c_n * y_j$$

| Matematik | Kod |
|---|---|
| $\Delta e_n$ | `myError[i]` |
| $L$ | `learningRate` |
| $b_n$ | `myBias[i]` |
| $w_j$ | `myWeights[i][j]` |
| $y_j$ (föregående lagers output) | `input[j]` |

---

## Hjälpfunktionerna `initRandom()` och `randomStartVal()`
Dessa hör inte till någon specifik ekvation, utan hanterar den slumpmässiga initieringen av vikter och bias som beskrivs i **L05**:
* `initRandom()` säkerställer att slumptalsgeneratorn (`std::rand()`) seedas exakt en gång, oavsett hur många lager som skapas.
* `randomStartVal()` genererar det faktiska slumptalet i intervallet `[0.0, 1.0]` som varje bias- och viktvärde initieras med. Intervallet är slutet i båda ändar: funktionen dividerar med `RAND_MAX`, så ett utfall på exakt `RAND_MAX` ger exakt `1.0`. Notera att detta är ett medvetet enkelt val för kursen; varje parameter startar positiv, med medelvärdet 0.5. Nätverk i produktion initierar symmetriskt kring noll och skalar efter antalet insignaler (Xavier eller He), vilket spelar störst roll för `Tanh`-lager, eftersom en genomgående positiv start placerar dem direkt i mättnad.

---
