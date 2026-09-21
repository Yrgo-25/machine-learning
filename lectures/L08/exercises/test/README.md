# L08 - Testsvit
Enhetstester för dense-lagret `Dense` från [bilaga B](../../appendix/b_exercises.md), skrivna med
testramverket [`yrgo-test`](https://github.com/yrgo-libs/yrgo-test).

Testsviten innehåller även testerna från **L06–L07**, så den ersätter L07:s testsvit. Det räcker
att köra denna.

Testerna för `Dense` omfattar dimensioner, startvärden för utdata och fel, getters, klassens
egenskaper samt argumentkontrollerna i `feedforward()`, `backpropagate()` och `optimize()`. Själva
beräkningarna testas först i **L09**.

Felmeddelanden som `Input dimension mismatch` och `Invalid learning rate` skrivs ut av de testfall
som medvetet anropar lagret med felaktiga argument. De betyder alltså inte att något test har
misslyckats.

---

## Köra testerna
Hämta först testramverket, en gång, i repots rotkatalog:

```bash
git submodule update --init --recursive
```

Kör därefter följande kommando i denna katalog:

```bash
make
```

Skriver ni er kod i er egen `ml`-kodbas i stället för i `exercises`, ange sökvägen till den:

```bash
make ML_DIR=<sökväg till er ml-katalog>
```

Testsviten kompilerar inte förrän `ml::ActFunc` finns i `ml/types.h`, och klassen `Dense` är
implementerad i `ml/dense_layer/dense.h` samt `source/ml/dense_layer/dense.cpp`, utöver filerna
från **L06–L07**.

**OBS!** Avbryts testsviten med `Segmentation fault` läser `train()` i `Shallow` sannolikt utanför
träningsdatan. Kontrollera att `myTrainSetCount` är det minsta av antalet rader i `trainInput`
och `trainOutput`.

---
