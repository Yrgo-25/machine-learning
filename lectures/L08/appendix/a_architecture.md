# Dense-lager - Arkitektur

## Introduktion
* Ett **dense-lager** *(fully connected layer)* är ett lager i ett neuralt nätverk där varje nod tar emot samtliga insignaler från föregående lager.
* Klassen `ml::dense_layer::Dense` (se [bilaga B](./b_exercises.md)) implementerar interfacet `ml::dense_layer::Interface` och representerar ett sådant lager, oavsett om det används som dolt lager eller utgångslager i nätverket.

---

## Uppbyggnad av ett lager
Ett dense-lager med `nodeCount` noder och `weightCount` vikter per nod består av de delar som beskrivs nedan. Dessa motsvarar direkt teorin i **L05** (se [bilaga A](../../L05/appendix/a_neural_networks.md)):

| Medlemsvariabel | Storlek | Motsvarar (se L05) |
|---|---|---|
| `myOutput` | `nodeCount` | $y$ - respektive nods utsignal (efter aktiveringsfunktion) |
| `myPreActivationOutput` | `nodeCount` | $s$ - respektive nods vägda summa (innan aktiveringsfunktion) |
| `myError` | `nodeCount` | $\Delta e$ - respektive nods beräknade fel |
| `myBias` | `nodeCount` | $b$ - respektive nods bias |
| `myWeights` | `nodeCount` x `weightCount` | $w$ - respektive nods vikter |
| `myActFunc` | – | $\delta$ - lagrets aktiveringsfunktion |

---

## Aktiveringsfunktion
Varje lager tilldelas en aktiveringsfunktion av typen `ActFunc` (`Relu` eller `Tanh`) vid konstruktion. Aktiveringsfunktionen tillämpas på samtliga noder i lagret vid feedforward (implementeras i **L09**).

---

## Sammansättning till ett neuralt nätverk
Ett enkelt neuralt nätverk (`ml::neural_network::Shallow`, se **L06-L07**) består av två dense-lager: ett dolt lager samt ett utgångslager. Det dolda lagrets utsignaler utgör utgångslagrets insignaler, vilket innebär att utgångslagrets `weightCount` är lika med det dolda lagrets `nodeCount`.

---
