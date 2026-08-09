# Arkitektur – Interface och klasser

## Översikt
Ni ska implementera ett enkelt neuralt nätverk steg för steg, med följande struktur:

```
ml::dense_layer::Interface   (Interface för dense-lager)
└── ml::dense_layer::Stub    (Dense-lagerstubb)

ml::neural_network::Interface   (Interface för neurala nätverk)
└── ml::neural_network::Shallow (Neuralt nätverk med ett enda dolt lager)
        ├── ml::dense_layer::Interface& myHiddenLayer
        └── ml::dense_layer::Interface& myOutputLayer
```

Under dessa lektioner används `ml::dense_layer::Stub` som placeholder. När en skarp implementation skapas i **L08–L09** ersätts stubben utan att resten av koden behöver ändras.

---

## Dense-lagrets interface
`ml::dense_layer::Interface` definierar kontraktet som alla dense-lager måste uppfylla:

| Metod | Beskrivning |
|---|---|
| `nodeCount()` | Antal noder i lagret |
| `weightCount()` | Antal vikter per nod |
| `output()` | Lagrets utsignaler efter feedforward |
| `error()` | Lagrets beräknade fel efter backpropagation |
| `weights()` | Lagrets vikter (2D: nod × vikt) |
| `feedforward(input)` | Beräknar utsignaler från indatan |
| `backpropagate(reference)` | Beräknar fel utifrån referensvärden (utgångslager) |
| `backpropagate(nextLayer)` | Beräknar fel utifrån nästa lager (dolt lager) |
| `optimize(input, learningRate)` | Uppdaterar bias och vikter |

---

## Nätverkets struktur
`Shallow` håller referenser till två dense-lager och kopplar ihop dem vid prediktion:

```
indata → myHiddenLayer.feedforward(input)
               ↓
       myHiddenLayer.output()
               ↓
       myOutputLayer.feedforward(...)
               ↓
       myOutputLayer.output()  →  prediktion
```

`Shallow` äger inte lagren, utan tar emot dem som referenser via konstruktorn. Detta gör det enkelt 
att byta ut lager utan att ändra nätverksklassen.

---

## Stubbklassen
`ml::dense_layer::Stub` implementerar interfacet men gör ingenting av värde:
* Feedforward sätter alltid utdata till ett fast värde (t.ex. `0.5`).
* Backpropagation och optimering gör ingenting.

Stubben finns till för att nätverket ska gå att kompilera och testköras redan nu. 
En korrekt implementation följer i **L08–L09**.

---
