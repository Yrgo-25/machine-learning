# L07 - Testsvit
Enhetstester för dense-lagerstubben samt komponenttester för det neurala nätverket från
[bilaga B](../../appendix/b_exercises.md), skrivna med testramverket
[`yrgo-test`](https://github.com/yrgo-libs/yrgo-test).

Testsviten innehåller även stubbtesterna från **L06**, så den ersätter L06:s testsvit. Det räcker
att köra denna.

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

Testsviten kompilerar inte förrän `ml/neural_network/interface.h`, `ml/neural_network/shallow.h`
samt `source/ml/neural_network/shallow.cpp` finns.

**OBS!** Avbryts testsviten med `Segmentation fault` läser `train()` sannolikt utanför
träningsdatan. Kontrollera att `myTrainSetCount` är det minsta av antalet rader i `trainInput`
och `trainOutput`.

---
