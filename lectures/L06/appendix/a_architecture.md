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

Strukturen byggs i två steg:
1. **L06:** Dense-lagrets interface samt stubbklassen (`ml::dense_layer`).
2. **L07:** Nätverkets interface samt klassen `Shallow` (`ml::neural_network`), se 
   [L07 bilaga A](../../L07/appendix/a_training_loop.md).

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

## Stubbklassen
`ml::dense_layer::Stub` implementerar interfacet men gör ingenting av värde:
* Feedforward sätter alltid utdata till ett fast värde (t.ex. `0.5`).
* Backpropagation och optimering gör ingenting.

Stubben finns till för att nätverket i **L07** ska gå att kompilera och testköra innan en skarp 
dense-lagerimplementation finns. En korrekt implementation följer i **L08–L09**.

---
