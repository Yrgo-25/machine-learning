# L06 - Testsvit
Enhetstester för dense-lagerstubben från [bilaga B](../../appendix/b_exercises.md), skrivna med 
testramverket [`yrgo-test`](https://github.com/yrgo-libs/yrgo-test).

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

---
