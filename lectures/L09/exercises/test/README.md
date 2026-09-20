# L09 - Testsvit
Enhets- och komponenttester för dense-lagret `Dense` från [bilaga B](../../appendix/b_exercises.md),
skrivna med testramverket [`yrgo-test`](https://github.com/yrgo-libs/yrgo-test).

Testsviten innehåller även testerna från **L06–L08**, så den ersätter L08:s testsvit. Det räcker
att köra denna.

Utöver testerna från **L08** verifieras randomiseringen av bias och vikter, samt att
`feedforward()`, båda varianterna av `backpropagate()` och `optimize()` räknar rätt. Eftersom
startvärdena slumpas jämför testerna mot värden som räknas fram ur lagrets egna vikter, i stället
för mot fasta tal. Testfallet `NetworkLearnsXorPattern` tränar slutligen ett helt nätverk av två
dense-lager på XOR-mönstret och kontrollerar att det faktiskt lär sig.

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

Testsviten kompilerar inte förrän `ml/dense_layer/dense.h` samt `source/ml/dense_layer/dense.cpp`
finns, utöver filerna från **L06–L07**.

**OBS!** Avbryts testsviten med `Segmentation fault` läser `train()` i `Shallow` sannolikt utanför
träningsdatan. Kontrollera att `myTrainSetCount` är det minsta av antalet rader i `trainInput`
och `trainOutput`.

---
