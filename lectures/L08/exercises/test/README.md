# L08 - Testsvit
Enhetstester för dense-lagret `Dense` från [bilaga B](../../appendix/b_exercises.md), skrivna med
testramverket [`yrgo-test`](https://github.com/yrgo-libs/yrgo-test).

Testsviten innehåller även testerna från **L06–L07**, så den ersätter L07:s testsvit. Det räcker
att köra denna.

Testerna för `Dense` omfattar enbart det som implementeras under denna lektion: dimensioner,
startvärden för utdata och fel, getters samt klassens egenskaper. `feedforward()`,
`backpropagate()` och `optimize()` testas först i **L09**, när de implementeras på riktigt.

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

Testsviten kompilerar inte förrän `ml/dense_layer/dense.h` samt `source/ml/dense_layer/dense.cpp`
finns, utöver filerna från **L06–L07**.

**OBS!** Avbryts testsviten med `Segmentation fault` läser `train()` i `Shallow` sannolikt utanför
träningsdatan. Kontrollera att `myTrainSetCount` är det minsta av antalet rader i `trainInput`
och `trainOutput`.

---
